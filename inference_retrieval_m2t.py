import torch
import torch.distributed as dist
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from tqdm import tqdm
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, WordpieceTokenizer
from train_autoencoder import ldmol_autoencoder
from utils import molT5_encoder, AE_SMILES_encoder
import time
from dataset import smi_txt_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import einops
import random


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
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = ae_model.load_state_dict(state_dict, strict=False)
        if rank == 0:   print('autoencoder', args.vae, msg)
    for param in ae_model.parameters():
        param.requires_grad = False
    del ae_model.text_encoder
    ae_model = ae_model.to(device)
    ae_model.eval()
    if rank == 0:   print(f'E #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

    text_encoder = T5ForConditionalGeneration.from_pretrained('laituan245/molt5-large-caption2smiles').to(device)
    text_tokenizer = T5Tokenizer.from_pretrained("laituan245/molt5-large-caption2smiles", model_max_length=512)
    del text_encoder.decoder

    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    if rank == 0:   print(f'text encoder #parameters: {sum(p.numel() for p in text_encoder.parameters())}, #trainable: {sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)}')

    dist.barrier()

    test_dataset = smi_txt_dataset([args.dataset], data_length=None, shuffle=True, unconditional=False, raw_description=True if args.level == 'paragraph' else False)
    test_dataset.data += test_dataset.data[:args.per_proc_batch_size - len(test_dataset.data) % args.per_proc_batch_size]
    if rank == 0:   print('#data:', len(test_dataset))

    sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        test_dataset,
        batch_size=int(args.per_proc_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    st = time.time()
    sampler.set_epoch(0)
    loader = tqdm(loader, miniters=1) if rank == 0 else loader

    tmp = []
    if rank == 0:
        with open('./generated_molecules_retrieval.txt', 'w') as f:
            pass
    for x_smiles, y in loader:
        # Sample inputs:
        x = AE_SMILES_encoder(x_smiles, ae_model).permute((0, 2, 1)).unsqueeze(-1)
        biot5_embed, pad_mask = molT5_encoder(y, text_encoder, text_tokenizer, args.description_length, device)

        y_cond = biot5_embed.to(device).type(torch.float32)
        pad_mask = pad_mask.to(device).bool()

        # Setup classifier-free guidance:
        model_kwargs = dict(y=y_cond, pad_mask=pad_mask)

        for i in range(x.size(0)):
            x_i = einops.repeat(x[i], 'F L P -> B F L P', B=x.size(0))
            losses = torch.zeros(x.size(0))
            for _ in range(args.n_iter):
                t = random.randint(50, diffusion.num_timesteps-50) * torch.ones((x.shape[0],), device=device).int()
                
                noise = torch.randn_like(x_i[0])
                noise = einops.repeat(noise, 'F L P -> B F L P', B=x.size(0))
                
                x_t = diffusion.q_sample(x_i, t, noise=noise)
                model_output = model(x_t, t, **model_kwargs)
                model_output, _ = torch.split(model_output, x_t.shape[1], dim=1)
                losses += torch.mean(((model_output - noise) ** 2).squeeze(-1), dim=(1, 2)).cpu()
            with open('./generated_molecules_retrieval.txt', 'a') as f:
                f.write(x_smiles[i]+y[i]+'\t'+str(int(torch.argmin(losses).item() == i))+'\n')
            if torch.argmin(losses).item() == i:
                tmp.append([x_smiles[i]]+[y[k]+'\n' for k in torch.topk(losses, 3, largest=False).indices])

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        print('time:', time.time() - st)
        with open('./generated_molecules_retrieval.txt', 'r') as f:
            lines = f.readlines()
            recall = [float(l.strip().split('\t')[1]) for l in lines]
            smiles = [l.split('\t')[0] for l in lines]
        dup = []
        recall2 = []
        for i, s in enumerate(smiles):
            if s in dup:
                continue
            else:
                recall2.append(recall[i])
                dup.append(s)

        print('acc@64:', torch.tensor(recall2).mean().item())
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--vae", type=str, default="./Pretrain/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--dataset", type=str, default="./data/retrieval_test_custom.txt")      # './data/retrieval_test_custom.txt', './data/PCdes/test.txt'
    parser.add_argument("--level", type=str, default="paragraph")      # paragraph, sentence
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)

