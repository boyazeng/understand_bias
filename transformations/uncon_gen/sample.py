"""
Samples a large number of images from a pre-trained DiT model using DDP.
"""
import argparse
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from diffusion import create_diffusion
from models import DiT_models
from tqdm import tqdm

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    utils.init_distributed_mode(args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    print(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=1
    )
    model.to("cuda")
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:

    state_dict = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    if "ema" in state_dict:  # supports checkpoints from train.py
        state_dict = state_dict["ema"]
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    vae.to("cuda")

    if global_rank == 0:
        print(f"Saving .png samples at {os.path.join(SAVE_ROOTS['uncon_gen'], args.dataset, args.split)}")
        os.makedirs(os.path.join(SAVE_ROOTS['uncon_gen'], args.dataset, args.split), exist_ok=True)
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num / global_batch_size) * global_batch_size)
    if global_rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    
    pbar = range(iterations)
    pbar = tqdm(pbar) if global_rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device="cuda")
        y_cls = torch.full((n,), 0, device="cuda")
        
        model_kwargs = dict(y=y_cls)
        sample_fn = model.forward
        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device="cuda"
        )

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * num_tasks + global_rank + total
            Image.fromarray(sample).save(os.path.join(SAVE_ROOTS['uncon_gen'], args.dataset, args.split, f"{index}.png"))
        total += global_batch_size

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-B/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num", type=int, default=None)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--dataset", type=str, default=None, required=True)
    parser.add_argument("--ckpt", type=str, default=None, required=True)
    parser.add_argument("--split", type=str, choices=['train', 'val'], default='val')
    args = parser.parse_args()
    main(args)
