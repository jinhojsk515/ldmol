from xbert import BertConfig, BertForMaskedLM
from transformers import BertModel
import torch
from torch import nn
import torch.distributed
from scheduler import create_scheduler
import copy
from pysmilesutils.augment import MolAugmenter
from torch.utils.data import DataLoader
from dataset import SMILESDataset_pretrain
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import torch.distributed
import argparse
from pathlib import Path
from utils import regexTokenizer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ldmol_autoencoder(pl.LightningModule):
    def __init__(self, cp=None, config=None, loader_len=0, no_train=False, tokenizer=None, use_linear=True, use_pr=False):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []

        self.text_encoder = BertForMaskedLM(config=BertConfig.from_json_file(config['bert_config_decoder']))
        bert_config2 = BertConfig.from_json_file(config['bert_config_encoder'])
        self.text_encoder2 = BertModel(config=bert_config2)

        if cp:
            checkpoint = torch.load(cp, map_location='cpu')
            try:
                state_dict = copy.deepcopy(checkpoint['model'])
            except:
                state_dict = copy.deepcopy(checkpoint['state_dict'])
            for key in list(state_dict.keys()):
                if 'text_encoder.' in key:
                    new_key = key.replace('text_encoder.', '')
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
            msg = self.text_encoder2.load_state_dict(state_dict, strict=False)
            print('inside', msg)
            del state_dict
        for param in self.text_encoder2.parameters():
            param.requires_grad = False

        self.aug = MolAugmenter()
        self.use_linear = use_linear
        if use_linear:
            self.output_dim = 64
            final_dim = bert_config2.hidden_size
            self.encode_prefix = nn.Linear(final_dim, self.output_dim)
            self.decode_prefix = nn.Linear(self.output_dim, final_dim)

        if not no_train:
            self.loader_len = loader_len
            self.warmup_steps = config['schedular']['warmup_epochs']

    def forward(self, text_input_ids, text_attention_mask, text_input_ids2, text_attention_mask2):
        # ================= MLM ================= #
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()[:, 1:]
        with torch.no_grad():
            text_embeds = self.text_encoder2(text_input_ids2, attention_mask=text_attention_mask2, return_dict=True).last_hidden_state
        if self.use_linear:
            text_embeds = self.decode_prefix(self.encode_prefix(text_embeds))
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text_attention_mask,
                                       encoder_hidden_states=text_embeds,
                                       encoder_attention_mask=None,
                                       return_dict=True,
                                       is_decoder=True,
                                       return_logits=True,
                                       )[:, :-1, :]

        loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        loss_mlm = loss_fct(mlm_output.permute((0, 2, 1)), labels)
        return loss_mlm

    def configure_optimizers(self):
        arg_opt = self.config['optimizer']
        optimizer = torch.optim.AdamW(self.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])
        arg_sche = AttrDict(self.config['schedular'])
        scheduler, _ = create_scheduler(arg_sche, optimizer)
        return [optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        print('qqq', metric)

    def training_step(self, train_batch, batch_idx):
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.zero_grad()
        text = train_batch
        # text_input = self.tokenizer(text, padding='longest', truncation=True, max_length=128, return_tensors="pt").to(self.device)
        text_input_ids = self.tokenizer(text, truncation='longest').to(self.device)
        text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        
        # text2 = ['[CLS]' + Chem.MolToSmiles(self.aug([Chem.MolFromSmiles(t[5:])])[0], canonical=False, isomericSmiles=True) if random.random()<0.1 else t for t in text]
        text2 = text
        # text_input2 = self.tokenizer(text2, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(self.device)
        text2_input_ids = self.tokenizer(text2, truncation='longest').to(self.device)
        text2_attention_mask = torch.where(text2_input_ids == 0, 0, 1).to(self.device)

        loss_mlm = self(text_input_ids, text_attention_mask, text2_input_ids, text2_attention_mask)
        loss = loss_mlm
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss_mlm', loss_mlm, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0:
            scheduler.step(self.current_epoch + self.warmup_steps)
        else:
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        self.training_step_outputs.append(torch.tensor([loss_mlm, ]))
        return torch.tensor([loss_mlm, ])

    def on_train_epoch_end(self):    # outputs: collection of returns from 'training_step'
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}')
        self.training_step_outputs.clear()


def main(args, config):
    # data
    print("Creating dataset")
    dataset = SMILESDataset_pretrain(args.data_path, data_length=[0, 10000000], is_train=False, shuffle=True)
    print('#data:', len(dataset))
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=8, shuffle=False, pin_memory=True, drop_last=True)
    tokenizer = regexTokenizer(vocab_path=args.vocab_filename, max_len=127)#newtkn

    # model
    print("Creating model")
    model = ldmol_autoencoder(config=config, cp=args.enc_checkpoint, tokenizer=tokenizer, use_linear=True)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']

        for key in list(state_dict.keys()):
            if 'queue' in key or 'property' in key or '_m' in key:
                del state_dict[key]
            if '_unk' in key:
                new_key = key.replace('_unk', '_mask')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_{epoch}',
                                                       save_top_k=1,
                                                       # every_n_train_steps=10000,
                                                       every_n_epochs=1
                                                       )
    ngpu = 8
    trainer = pl.Trainer(accelerator='gpu', devices=ngpu, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=False), limit_val_batches=0.,
                         # logger=WandbLogger(),
                         )
    trainer.fit(model, data_loader, None, ckpt_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--enc_checkpoint', default='./Pretrain/checkpoint_encoder.ckpt')
    parser.add_argument('--data_path', default='./data/pubchem_10m.txt')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300_sc.txt')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    pretrain_config = {
        'property_width': 768,
        'embed_dim': 256,
        'batch_size': 128,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'schedular': {'sched': 'cosine', 'lr': 0.5e-4, 'epochs': 5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 0.5e-4, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)

