from xbert import BertConfig
from transformers import BertModel
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed
import pytorch_lightning as pl
from scheduler import create_scheduler
import argparse
from pathlib import Path
from dataset import SMILESDataset_pretrain
from pytorch_lightning.strategies import DDPStrategy
from rdkit import Chem
import random
from torch.utils.data import DataLoader
from pysmilesutils.augment import MolAugmenter
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from utils import regexTokenizer


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ldmol_encoder(pl.LightningModule):
    def __init__(self, tokenizer=None, config=None, loader_len=0, no_train=False):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.config = config
        self.tokenizer = tokenizer
        self.training_step_outputs = []

        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config_encoder'])
        self.text_encoder = BertModel(config=bert_config)
        text_width = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.aug = MolAugmenter()

        # create momentum models
        self.text_encoder_m = BertModel(config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        for p in self.text_encoder_m.parameters():      p.requires_grad = False
        for p in self.text_proj_m.parameters():         p.requires_grad = False

        self.model_pairs = [[self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        if not no_train:
            self.temp = nn.Parameter(torch.ones([]) * config['temp'])
            self.warmup_steps = config['schedular']['warmup_epochs']
            self.loader_len = loader_len
            self.momentum = config['momentum']
            self.queue_size = config['queue_size']
            self.register_buffer("text1_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("text2_queue", torch.randn(embed_dim, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

            self.text1_queue = nn.functional.normalize(self.text1_queue, dim=0)
            self.text2_queue = nn.functional.normalize(self.text2_queue, dim=0)

    def forward(self, text1_input_ids, text1_attention_mask, text2_input_ids, text2_attention_mask, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.07, 0.5)

        text1_embeds = self.text_encoder(text1_input_ids, attention_mask=text1_attention_mask, return_dict=True).last_hidden_state
        text1_feat = F.normalize(self.text_proj(text1_embeds[:, 0, :]), dim=-1)
        text2_embeds = self.text_encoder(text2_input_ids, attention_mask=text2_attention_mask, return_dict=True).last_hidden_state
        text2_feat = F.normalize(self.text_proj(text2_embeds[:, 0, :]), dim=-1)
        # get momentum features

        with torch.no_grad():
            self._momentum_update()
            text1_embeds_m = self.text_encoder_m(text1_input_ids, attention_mask=text1_attention_mask, return_dict=True).last_hidden_state
            text1_feat_m = F.normalize(self.text_proj(text1_embeds_m[:, 0, :]), dim=-1)
            text1_feat_all = torch.cat([text1_feat_m.t(), self.text1_queue.clone().detach()], dim=1)

            text2_embeds_m = self.text_encoder_m(text2_input_ids, attention_mask=text2_attention_mask, return_dict=True).last_hidden_state
            text2_feat_m = F.normalize(self.text_proj(text2_embeds_m[:, 0, :]), dim=-1)
            text2_feat_all = torch.cat([text2_feat_m.t(), self.text2_queue.clone().detach()], dim=1)

            sim_21_m = text2_feat_m @ text1_feat_all / self.temp
            sim_12_m = text1_feat_m @ text2_feat_all / self.temp
            # sim_11_m = text1_feat_m @ text1_feat_all / self.temp
            # sim_22_m = text2_feat_m @ text2_feat_all / self.temp

            sim_targets = torch.zeros(sim_21_m.size()).to(self.device)
            sim_targets.fill_diagonal_(1)

            sim_21_targets = alpha * F.softmax(sim_21_m, dim=1) + (1 - alpha) * sim_targets
            sim_12_targets = alpha * F.softmax(sim_12_m, dim=1) + (1 - alpha) * sim_targets
            # sim_11_targets = alpha * F.softmax(sim_11_m, dim=1) + (1 - alpha) * sim_targets
            # sim_22_targets = alpha * F.softmax(sim_22_m, dim=1) + (1 - alpha) * sim_targets

        sim_21 = text2_feat @ text1_feat_all / self.temp
        sim_12 = text1_feat @ text2_feat_all / self.temp
        # sim_11 = text1_feat @ text1_feat_all / self.temp
        # sim_22 = text2_feat @ text2_feat_all / self.temp

        loss_21 = -torch.sum(F.log_softmax(sim_21, dim=1) * sim_21_targets, dim=1).mean()
        loss_12 = -torch.sum(F.log_softmax(sim_12, dim=1) * sim_12_targets, dim=1).mean()
        # loss_11 = -torch.sum(F.log_softmax(sim_11, dim=1) * sim_11_targets, dim=1).mean()
        # loss_22 = -torch.sum(F.log_softmax(sim_22, dim=1) * sim_22_targets, dim=1).mean()

        loss_ita = loss_21 + loss_12 # + loss_11 + loss_22

        self._dequeue_and_enqueue(text1_feat_m, text2_feat_m)

        return loss_ita

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, text1_feat, text2_feat):
        text1_feats = concat_all_gather(text1_feat)
        text2_feats = concat_all_gather(text2_feat)
        # print(text1_feats.shape, text2_feats.shape)
        text1_feats = text1_feats[:64]
        text2_feats = text2_feats[:64]

        batch_size = text1_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.text1_queue[:, ptr:ptr + batch_size] = text1_feats.T
        self.text2_queue[:, ptr:ptr + batch_size] = text2_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

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
        text1 = train_batch
        
        text1 = [t.split('Q') for t in text1]
        tmp = []
        for t in text1:
            tmp += t
        text2 = []
        text1 = []
        for t in tmp:
            try:
                t2 = '[CLS]' + Chem.MolToSmiles(self.aug([Chem.MolFromSmiles(t[5:])])[0], canonical=False, isomericSmiles=True)
                text1.append(t)
                text2.append(t2)
            except:
                print('err', t)
                continue

        # text_input1 = self.tokenizer(text1, padding='longest', truncation=True, max_length=128, return_tensors="pt").to(self.device)
        # text_input2 = self.tokenizer(text2, padding='longest', truncation=True, max_length=128, return_tensors="pt").to(self.device)
        text_input_ids = self.tokenizer(text1, truncation='longest').to(self.device)
        text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        text2_input_ids = self.tokenizer(text2, truncation='longest').to(self.device)
        text2_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        alpha = self.config['alpha'] if self.current_epoch > 0 else self.config['alpha'] * min(1., batch_idx / self.loader_len)

        # loss = self(text_input1.input_ids[:, 1:], text_input1.attention_mask[:, 1:], text_input2.input_ids[:, 1:], text_input2.attention_mask[:, 1:], alpha=alpha)
        loss = self(text_input_ids, text_attention_mask, text2_input_ids, text2_attention_mask, alpha=alpha)
        if loss != torch.tensor(0.):
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.)
            optimizer.step()
        else:
            print('aaaaaaaaaaaa')
        if self.global_rank == 0:
            self.log('lr', optimizer.param_groups[0]["lr"], prog_bar=True)
            self.log('loss', loss, prog_bar=True)
            self.log('temp', self.temp, prog_bar=True)

        step_size = 100
        warmup_iterations = self.warmup_steps * step_size
        if self.current_epoch > 0 and batch_idx == 0:
            scheduler.step(self.current_epoch + self.warmup_steps)
        else:
            if self.current_epoch == 0 and batch_idx % step_size == 0 and batch_idx <= warmup_iterations:
                scheduler.step(batch_idx // step_size)
        self.training_step_outputs.append(torch.tensor([loss]))
        return torch.tensor([loss])

    def on_train_epoch_end(self):    # outputs: collection of returns from 'training_step'
        tmp = torch.stack(self.training_step_outputs[-1000:]).mean(dim=0).tolist()
        if self.global_rank == 0:
            print(f'\n mean loss: {tmp[0]:.4f}')
        self.training_step_outputs.clear()


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def main(args, config):
    # data
    print("Creating dataset")
    dataset = SMILESDataset_pretrain(args.data_path, data_length=[0, 10000000])
    print('#data:', len(dataset))
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=16, shuffle=False, pin_memory=True, drop_last=True)
    tokenizer = regexTokenizer(vocab_path=args.vocab_filename, max_len=127)#newtkn

    # model
    print("Creating model")
    ngpu=1
    model = ldmol_encoder(config=config, tokenizer=tokenizer, loader_len=len(data_loader) // ngpu)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    # training
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=args.output_dir, filename='checkpoint_{epoch}',
                                                       save_top_k=1,
                                                       # every_n_train_steps=10000,
                                                       every_n_epochs=1
                                                       )

    trainer = pl.Trainer(accelerator='gpu', devices=ngpu, precision='16-mixed', max_epochs=config['schedular']['epochs'],
                         callbacks=[checkpoint_callback], strategy=DDPStrategy(find_unused_parameters=True), limit_val_batches=0.)
    trainer.fit(model, data_loader, None, ckpt_path=None)


@torch.no_grad()
def evaluate(args, config):
    # data
    print("Creating dataset")
    dataset = SMILESDataset_pretrain(args.data_path, data_length=[0, 10], is_train=False)
    print('#data:', len(dataset))
    data_loader = DataLoader(dataset, batch_size=config['batch_size'], num_workers=8, shuffle=True, pin_memory=True, drop_last=False)
    tokenizer = regexTokenizer(vocab_path=args.vocab_filename, max_len=127)#newtkn

    # model
    print("Creating model")
    ngpu = 1
    model = ldmol_encoder(config=config, tokenizer=tokenizer, loader_len=len(data_loader) // ngpu)
    print('#parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)
    model = model.to('cuda')
    model.eval()
    for text1 in data_loader:
        # text1 = ['c1ccccc1', 'c1ccccc1O', 'c1ccccc1F', 'c1ccccc1C']
        text1 = [Chem.MolToSmiles(Chem.MolFromSmiles(t[5:]), canonical=True, isomericSmiles=False) for t in text1]
        text1_sc = [Chem.MolToSmiles(random.choice(list(EnumerateStereoisomers(Chem.MolFromSmiles(t)))), canonical=True, isomericSmiles=True) for t in text1]
        text1 = [Chem.MolToSmiles(random.choice(list(EnumerateStereoisomers(Chem.MolFromSmiles(t)))), canonical=True, isomericSmiles=True) for t in text1]
        text1_combined = []
        for i in range(len(text1)):
            text1_combined.append(text1[i])
            text1_combined.append(text1_sc[i])
        print(text1_combined)
        text1 = ['[CLS]' + t for t in text1_combined]
        text2 = ['[CLS]' + Chem.MolToSmiles(model.aug([Chem.MolFromSmiles(t[5:])])[0], canonical=False, isomericSmiles=True) for t in text1]
        # text_input1 = model.tokenizer(text1, padding='longest', truncation=True, max_length=128, return_tensors="pt").to(model.device)
        # text_input2 = model.tokenizer(text2, padding='longest', truncation=True, max_length=128, return_tensors="pt").to(model.device)
        text_input_ids = model.tokenizer(text1, truncation='longest').to(self.device)
        text_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        text2_input_ids = model.tokenizer(text2, truncation='longest').to(self.device)
        text2_attention_mask = torch.where(text_input_ids == 0, 0, 1).to(self.device)
        
        # text1_embeds = model.text_encoder(text_input1.input_ids[:, 1:], attention_mask=text_input1.attention_mask[:, 1:], return_dict=True).last_hidden_state
        text1_embeds = model.text_encoder(text_input_ids, attention_mask=text_attention_mask, return_dict=True).last_hidden_state
        text1_feat = F.normalize(model.text_proj(text1_embeds[:, 0, :]), dim=-1)
        # text2_embeds = model.text_encoder(text_input2.input_ids[:, 1:], attention_mask=text_input2.attention_mask[:, 1:], return_dict=True).last_hidden_state
        text2_embeds = model.text_encoder(text2_input_ids, attention_mask=text2_attention_mask, return_dict=True).last_hidden_state
        text2_feat = F.normalize(model.text_proj(text2_embeds[:, 0, :]), dim=-1)
        sim = text1_feat @ text2_feat.T
        print(sim)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default="")     # "./Pretrain/checkpoint_smauCLIP_sc.ckpt"
    parser.add_argument('--data_path', default='./data/unpaired_200k.txt')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain')
    parser.add_argument('--vocab_filename', default='./vocab_bpe_300_sc.txt')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    pretrain_config = {
        'embed_dim': 256,
        'batch_size': 64,
        'temp': 0.07,
        'queue_size': 16384,
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config_encoder': './config_encoder.json',
        'schedular': {'sched': 'cosine', 'lr': 1e-4, 'epochs': 5, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 5e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 1e-4, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, pretrain_config)
    # evaluate(args, pretrain_config)
