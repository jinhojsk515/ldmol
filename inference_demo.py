import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from tqdm import tqdm
import math
import argparse
from einops import repeat
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, WordpieceTokenizer
from train_autoencoder import ldmol_autoencoder
from utils import AE_SMILES_decoder, molT5_encoder, get_validity
import time
from rdkit import Chem


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
    in_channels = 64  # 64
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
    print('DiT from ', ckpt_path, msg)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'embed_dim': 256,
    }
    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300_sc.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=1000)
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer)
    if args.vae:
        print('LOADING PRETRAINED MODEL..', args.vae)
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = ae_model.load_state_dict(state_dict, strict=False)
        print('autoencoder', msg)
    for param in ae_model.parameters():
        param.requires_grad = False
    del ae_model.text_encoder2
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f'AE #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

    # assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale != 1.0

    text_encoder = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles').to(device)
    text_tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
    del text_encoder.decoder

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print(f'text encoder #parameters: {sum(p.numel() for p in text_encoder.parameters())}, #trainable: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}')

    dist.barrier()
    if rank == 0:
        with open('./generated_molecules.txt', 'w') as f:
            pass

    prompt = args.prompt
    prompt_null = "no dsecription."

    biot5_embed, pad_mask = molT5_encoder([prompt], text_encoder, text_tokenizer, args.description_length, device)
    biot5_embed_null, pad_mask_null = molT5_encoder([prompt_null], text_encoder, text_tokenizer, args.description_length, device)

    biot5_embed = repeat(biot5_embed, '1 L D -> B L D', B=args.per_proc_batch_size)
    pad_mask = repeat(pad_mask, '1 L -> B L', B=args.per_proc_batch_size)
    y_cond = biot5_embed.to(device).type(torch.float32)
    pad_mask_cond = pad_mask.to(device).bool()

    biot5_embed_null = repeat(biot5_embed_null, '1 L D -> B L D', B=args.per_proc_batch_size)
    pad_mask_null = repeat(pad_mask_null, '1 L -> B L', B=args.per_proc_batch_size)
    y_null = biot5_embed_null.to(device).to(torch.float32)
    pad_mask_null = pad_mask_null.to(device).bool()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    st = time.time()
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, 1, device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y = torch.cat([y_cond, y_null], 0)
            pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
            model_kwargs = dict(y=y, pad_mask=pad_mask, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y_cond, pad_mask=pad_mask)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        # print('zzzz', samples.shape)

        samples = samples.squeeze(-1).permute((0, 2, 1))
        samples = AE_SMILES_decoder(samples, ae_model     , stochastic=False, k=1)

        # Save samples to disk as individual .png files
        with open('./generated_molecules.txt', 'a') as f:
            for s in samples:
                f.write(s+'\n')
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print('time:', time.time()-st)
        with open('./generated_molecules.txt', 'r') as f:
            text_out = [m.strip() for m in f.readlines()]
            print(len(text_out))
        val = []
        for l in text_out:
            try:
                if l == "":
                    continue
                mol = Chem.MolFromSmiles(l)
                s = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                val.append(s)
            except:
                continue

        v = get_validity(text_out)
        print(prompt)
        print("="*100)
        print(val)
        print('validity:', v)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--vae", type=str, default="./Pretrain/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--prompt", type=str, default="This molecule contains an amino group.")
    parser.add_argument("--description-length", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--per-proc-batch-size", type=int, default=10)
    parser.add_argument("--cfg-scale",  type=float, default=5.)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
