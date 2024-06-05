# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from einops import repeat
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForCausalLM, AutoTokenizer, BertTokenizer, WordpieceTokenizer, BitsAndBytesConfig
from train_smauCLIP_dec import smauCLIP
from utils import llama2_encoder, SPMM_decoder, molT5_encoder, BioT5_encoder, SPMM_SMILES_encoder, get_biot5_tokenizer, get_validity_uniqueness
from rdkit import Chem, RDLogger
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
import time


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


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
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = 127
    in_channels = 64
    cross_attn = 768
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt #or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    msg = model.load_state_dict(state_dict, strict=False)
    print('DiT from ', ckpt_path, msg)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    spmm_config = {
        'bert_config_text': './config_bert2.json',
        'bert_config_smiles': './config_bert_smiles.json',
        'bert_config_smauclip': './config_bert_smauclip.json',
        'embed_dim': 256,
    }
    tokenizer = BertTokenizer(vocab_file='./vocab_bpe_300_sc.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.wordpiece_tokenizer = WordpieceTokenizer(vocab=tokenizer.vocab, unk_token=tokenizer.unk_token, max_input_chars_per_word=1000)
    spmm = smauCLIP(config=spmm_config, no_train=True, tokenizer=tokenizer)
    if args.vae:
        print('LOADING PRETRAINED MODEL..', args.vae)
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = spmm.load_state_dict(state_dict, strict=False)
        print('spmm', msg)
    for param in spmm.parameters():
        param.requires_grad = False
    del spmm.text_encoder2
    spmm = spmm.to(device)
    spmm.eval()
    print(f'spmm #parameters: {sum(p.numel() for p in spmm.parameters())}, #trainable: {sum(p.numel() for p in spmm.parameters() if p.requires_grad)}')
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0


    if args.text_encoder_name == 'biot5':
        text_encoder = T5ForConditionalGeneration.from_pretrained('QizhiPei/biot5-base-text2mol').to(device)
        text_tokenizer = get_biot5_tokenizer()
        del text_encoder.decoder
    elif args.text_encoder_name == 'molt5':
        text_encoder = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles').to(device)
        text_tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
        del text_encoder.decoder
    elif args.text_encoder_name == 'llama2':
        text_encoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", load_in_8bit=True, trust_remote_code=True, use_cache=True,)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # text_encoder = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", quantization_config=bnb_config, trust_remote_code=True, use_cache=True, )
        text_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        text_tokenizer.pad_token = text_tokenizer.eos_token
    else:
        raise ValueError('text_encoder_name must be biot5 or molt5')
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print(f'text encoder #parameters: {sum(p.numel() for p in text_encoder.parameters())}, #trainable: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}')


    '''
    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    '''
    dist.barrier()
    if rank == 0:
        with open('./generated_molecules.txt', 'w') as f:
            pass

    prompt = args.prompt    # "This molecule is soluble in water."
    prompt_null = "no dsecription."
    if args.text_encoder_name == 'biot5':
        biot5_embed, pad_mask = BioT5_encoder([prompt], text_encoder, text_tokenizer, args.description_length, device)
        biot5_embed_null, pad_mask_null = BioT5_encoder([prompt_null], text_encoder, text_tokenizer, args.description_length, device)
    elif args.text_encoder_name == 'molt5':
        biot5_embed, pad_mask = molT5_encoder([prompt], text_encoder, text_tokenizer, args.description_length, device)
        biot5_embed_null, pad_mask_null = molT5_encoder([prompt_null], text_encoder, text_tokenizer, args.description_length, device)
    else:
        biot5_embed, pad_mask = llama2_encoder([prompt], text_encoder, text_tokenizer, args.description_length, device)
        biot5_embed_null, pad_mask_null = llama2_encoder([prompt_null], text_encoder, text_tokenizer, args.description_length, device)
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
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
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
        # z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        z = torch.randn(n, model.in_channels, latent_size, 1, device=device)
        # y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            # y_null = torch.tensor([1000] * n, device=device)
            #print('a', y_cond.shape, y_null.shape, pad_mask_cond.shape)
            y = torch.cat([y_cond, y_null], 0)
            pad_mask = torch.cat([pad_mask_cond, pad_mask_null], 0)
            #print('aa', y.shape, pad_mask.shape)
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

        # samples = vae.decode(samples / 0.18215).sample
        samples = samples.squeeze(-1).permute((0, 2, 1))
        samples = SPMM_decoder(samples, spmm, stochastic=False, k=1)

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
        sc_specified = []
        for l in text_out:
            try:
                if l == "":
                    continue
                mol = Chem.MolFromSmiles(l)
                sc_specified.append(1 if len(list(EnumerateStereoisomers(mol)))==1 else 0)
                s = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                val.append(s)
            except:
                continue

        v, u = get_validity_uniqueness(text_out)
        print(prompt)
        print("="*100)
        print(val)
        print('validity:', v, 'uniqueness:', u)

        import numpy as np
        print(np.mean(sc_specified))

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, default="../CLDM/Pretrain/checkpoint_smauCLIP_dec_64_sc.ckpt")  # Choice doesn't affect training
    parser.add_argument("--text-encoder-name", type=str, default="llama2")
    parser.add_argument("--prompt", type=str, default="This molecule contains an amino group.")
    parser.add_argument("--description-length", type=int, default=200)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=7.5)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
