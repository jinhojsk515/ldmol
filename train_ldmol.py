# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model

from models import DiT_models
from diffusion import create_diffusion

from transformers import T5ForConditionalGeneration, T5Tokenizer
from train_autoencoder import ldmol_autoencoder
from utils import molT5_encoder, AE_SMILES_encoder, regexTokenizer
from dataset import smi_txt_dataset
import random

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    latent_size = 127
    in_channels = 64  # 64, 1024
    cross_attn = 768
    if args.text_encoder_name == 'llama2':
        condition_dim = 4096
    elif args.text_encoder_name == 'molt5':
        condition_dim = 1024
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        cross_attn=cross_attn,
        condition_dim=condition_dim
    )

    if args.ckpt:
        ckpt_path = args.ckpt #or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
        state_dict = find_model(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=True)
        print('load DiT from ', ckpt_path, msg)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule

    
    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'embed_dim': 256,
    }
    tokenizer = regexTokenizer(vocab_path='./vocab_bpe_300_sc.txt', max_len=127)#newtkn
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True)
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
    del ae_model.text_encoder
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f'AE #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    text_encoder = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles').to(device)
    text_tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
    del text_encoder.decoder

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    print(f'text encoder #parameters: {sum(p.numel() for p in text_encoder.parameters())}, #trainable: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}')


    # Setup data:
    dataset = smi_txt_dataset([
        './data/chebi_20/train_parsed.txt',
        './data/PubchemSTM/train_parsed.txt',
        './data/PCdes/train_parsed.txt'
        './data/unpaired_200k.txt',
    ], data_length=None, shuffle=True, unconditional=False, raw_description=True)
    print('#data:', len(dataset))

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = AE_SMILES_encoder(x, ae_model).permute((0, 2, 1)).unsqueeze(-1)
                
                y = [d if random.random() < 0.95 else dataset.null_text for d in y]
                biot5_embed, pad_mask = molT5_encoder(y, text_encoder, text_tokenizer, args.description_length, device)
                y = biot5_embed.detach().to(device)  # batch*len*768
                
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16*6)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="./Pretrain/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    args = parser.parse_args()
    main(args)
