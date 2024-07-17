import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, WordpieceTokenizer
from train_autoencoder import ldmol_autoencoder
from utils import AE_SMILES_decoder, molT5_encoder, AE_SMILES_encoder
from rdkit import Chem
import time
import random
from torch.optim.sgd import SGD


@torch.no_grad()
def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path with --ckpt.")

    # Load model:
    latent_size = 127
    in_channels = 64
    cross_attn = 768
    condition_dim = 1024
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        condition_dim=condition_dim,
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    msg = model.load_state_dict(state_dict, strict=False)
    if rank == 0:   print('DiT from ', ckpt_path, msg)
    model.eval()  # important!

    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'embed_dim': 256,
    }
    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300_sc.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=1000)
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer)
    if args.vae:
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = ae_model.load_state_dict(state_dict, strict=False)
        if rank == 0:   print('autoencoder', args.vae, msg)
    for param in ae_model.parameters():
        param.requires_grad = False
    ae_model = ae_model.to(device)
    ae_model.eval()
    if rank == 0:   print(f'AE #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

    text_encoder = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles').to(device)
    text_tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
    del text_encoder.decoder

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    if rank == 0:
        print(f'text encoder #parameters: {sum(p.numel() for p in text_encoder.parameters())}, #trainable: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}')

    dist.barrier()

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    st = time.time()

    x_smiles = args.input_smiles
    y_s = [args.source_text]
    y_t = [args.target_text]

    # x_smiles = 'c1cncnc1CCC'
    # y_s = ['This molecule has a pyrimidine ring.']
    # y_t = ['This molecule has a pyrimidine ring and acetyl group.']
 
    # x_smiles = 'C[C@H](CCc1ccccc1)Nc1ccc(C#N)cc1F'
    # y_s = ['This molecule contains fluorine.']
    # y_t = ['This molecule contains bromine.']

    # x_smiles = 'CC(C)CC1=CC=CC=C1C(=O)O'
    # y_s = ['This molecule is carboxylic acid.']
    # y_t = ['This molecule is ester.']

    x_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(x_smiles), isomericSmiles=True, canonical=True)
    loss_scale = 0.4
    n_iter = 200
    cfg = 2.
    print(x_smiles, ':', y_s[0], '=>', y_t[0])

    # Sample inputs:
    x_source = AE_SMILES_encoder([x_smiles], ae_model).permute((0, 2, 1)).unsqueeze(-1)
    x_target = x_source.clone()
    x_target.requires_grad = True
    optimizer = SGD(params=[x_target], lr=1e-1)

    y_n = 'no description.'

    biot5_embed_s, pad_mask_s = molT5_encoder(y_s, text_encoder, text_tokenizer, args.description_length, device)
    biot5_embed_t, pad_mask_t = molT5_encoder(y_t, text_encoder, text_tokenizer, args.description_length, device)
    biot5_embed_n, pad_mask_n = molT5_encoder(y_n, text_encoder, text_tokenizer, args.description_length, device)

    y_cond_s = biot5_embed_s.to(device).type(torch.float32)
    pad_mask_s = pad_mask_s.to(device).bool()
    y_cond_t = biot5_embed_t.to(device).type(torch.float32)
    pad_mask_t = pad_mask_t.to(device).bool()
    y_cond_n = biot5_embed_n.to(device).type(torch.float32)
    pad_mask_n = pad_mask_n.to(device).bool()

    # Setup classifier-free guidance:
    model_kwargs_s = dict(y=y_cond_s, pad_mask=pad_mask_s)
    model_kwargs_t = dict(y=y_cond_t, pad_mask=pad_mask_t)
    model_kwargs_n = dict(y=y_cond_n, pad_mask=pad_mask_n)

    for step in range(n_iter):
        # t = torch.randint(50, diffusion.num_timesteps-50, (x.shape[0],), device=device)
        t = random.randint(100, diffusion.num_timesteps - 100) * torch.ones((x_source.shape[0],), device=device).int()

        noise = torch.randn_like(x_target)
        x_target_t = diffusion.q_sample(x_target, t, noise=noise)
        x_source_t = diffusion.q_sample(x_source, t, noise=noise)

        model_output_s = model(x_source_t, t, **model_kwargs_s)
        model_output_s, _ = torch.split(model_output_s, x_source_t.shape[1], dim=1)
        model_output_sn = model(x_source_t, t, **model_kwargs_n)
        model_output_sn, _ = torch.split(model_output_sn, x_source_t.shape[1], dim=1)
        
        model_output_t = model(x_target_t, t, **model_kwargs_t)
        model_output_t, _ = torch.split(model_output_t, x_target_t.shape[1], dim=1)
        model_output_tn = model(x_target_t, t, **model_kwargs_n)
        model_output_tn, _ = torch.split(model_output_tn, x_target_t.shape[1], dim=1)

        model_output_s = model_output_sn + cfg*(model_output_s - model_output_sn)
        model_output_t = model_output_tn + cfg*(model_output_t - model_output_tn)

        grad = (model_output_t - model_output_s).detach()

        #loss = x_target * grad
        #optimizer.zero_grad()
        #(loss_scale * loss).backward()
        #optimizer.step()

        x_target -= grad*loss_scale

        if step % 20 == 0:
            output = x_target.squeeze(-1).permute((0, 2, 1))
            output = AE_SMILES_decoder(output, ae_model, stochastic=False, k=1)
            print(f'{step}\t: {output[0]}')

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    print('time:', time.time() - st)
    print(f'{x_smiles}')
    print(f'{output[0]}')
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--vae", type=str, default="./Pretrain/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--input-smiles", type=str, default="C[C@H](CCc1ccccc1)Nc1ccc(C#N)cc1F")
    parser.add_argument("--source-text", type=str, default="This molecule contains fluorine.")
    parser.add_argument("--target-text", type=str, default="This molecule contains bromine.")
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

