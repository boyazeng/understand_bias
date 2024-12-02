import numpy as np
import torch
import os
import argparse
import random
import json
from diffusers import AutoPipelineForText2Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils

class Caption_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['text_to_image'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.caption_path = os.path.join(SAVE_ROOTS['caption'], 'short', args.dataset, args.split, f"{args.split}.json")
        with open(self.caption_path, 'r') as file:
            self.name_to_caption = json.load(file)
        if args.num is not None:
            self.name_to_caption = [(name, caption) for name, caption in self.name_to_caption.items()][:args.num]
        assert len(self.name_to_caption) == args.num, f"not enough captions in {args.dataset} {args.split} split"

    def __len__(self):
        return len(self.name_to_caption)
    
    def __getitem__(self, i):
        name, caption = self.name_to_caption[i]
        save_path = os.path.join(self.output_dir, f"{name}.png")
        return {"caption": caption, "save_path": save_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default='cc')
    parser.add_argument('--split', type=str, choices=['val', 'train'], default='val')
    parser.add_argument('--num', type=int, default=None)
    args = parser.parse_args()
    
    utils.init_distributed_mode(args)
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    print(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    sd = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    sd.to("cuda")
    sd.enable_vae_slicing()
    sd.enable_vae_tiling()
    sd.enable_attention_slicing()
    
    dataset = Caption_Dataset(args)
    sampler_train = DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=args.seed)
    data_loader = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=1,
        drop_last=False,
        collate_fn=lambda x: x,
        shuffle=False,
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Text-to-Image {args.dataset} {args.split} split'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            batch = batch[0]
            caption = batch["caption"]
            save_path = batch["save_path"]
            result = sd(caption, num_inference_steps=1, guidance_scale=0.0).images[0]
            result = result.resize((256, 256), Image.BICUBIC)
            result.save(save_path)
            metric_logger.synchronize_between_processes()