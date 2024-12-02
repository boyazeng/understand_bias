import argparse
import cv2
import numpy as np
import os
import torch
import random
import json
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

from PIL import Image, PngImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['caption'], args.caption_type, args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.txt')
        try:
            with open(path, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')

            if min(image.size) > 500:
                image = self.preprocess(image)
            image = np.array(image)

            if (image.shape[0]==1 and image.shape[1]==1) or (len(image.shape)==3 and image.shape[2]!=3):
                # Skip through inputs LLaVA can't handle
                print(image.shape)
                return {"image": None, "save_path": save_path}
            return {"image": image, "save_path": save_path}
        except Exception as e:
            print(f"error to open {path} with error {e}")
            return {"image": None, "save_path": save_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="datacomp")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--num', type=int, default=10_000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--caption_type', type=str, choices=["short", "long"], default="short")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    dataset = Image_Dataset(args)
    print("Remaining number of images to process: ", len(dataset))
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        drop_last=False,
        shuffle=False,
        collate_fn=lambda x: x,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained(f"llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        f"llava-hf/llava-1.5-7b-hf",
        quantization_config=quantization_config,
        device_map="cuda",
    )
    if args.caption_type == "short":
        prompt = "USER: <image>\nDescribe this image in one sentence.\nASSISTANT:"
    elif args.caption_type == "long":
        prompt = "USER: <image>\nDescribe this image in one paragraph.\nASSISTANT:"
    else:
        raise ValueError("Invalid caption type")

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Image captioning {args.dataset} {args.split} split'
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            # LLaVA can't handle invalid images, so we skip them
            invalid_mask = np.array([item["image"] is None for item in batch])
            if np.all(invalid_mask):
                continue
            # Remove the invalid images from the batch
            images = [item["image"] for item, invalid in zip(batch, invalid_mask) if not invalid]
            # Iterate through every prompt for the batch of images
            try:
                inputs = processor(images=images, text=[prompt]*len(images), padding=True, return_tensors="pt").to(model.device)
                output = model.generate(**inputs, max_new_tokens=100)
                generated_text = processor.batch_decode(output, skip_special_tokens=True)
            except Exception as e:
                for each in batch:
                    print(f"error to process {each['save_path']} with error {e}")
                continue

            responses = np.array([text.split("ASSISTANT:")[-1] for text in generated_text])
            # Skip the invalid images
            ptr = 0
            for inval, item in zip(invalid_mask, batch):
                if not inval:
                    with open(item["save_path"], 'w', encoding='utf-8') as file:
                        file.write(responses[ptr].encode('utf-8', 'ignore').decode('utf-8'))
                    ptr += 1

            metric_logger.synchronize_between_processes()
    
    # we combine all the captions into a single json file
    output_dir = os.path.join(SAVE_ROOTS['caption'], args.caption_type, args.dataset, args.split)
    captions = {}
    for caption_path in os.listdir(output_dir):
        if caption_path.endswith(".txt"):
            original_name = os.path.basename(caption_path).rstrip(".txt")
            with open(os.path.join(output_dir, caption_path), 'r') as file:
                captions[original_name] = file.read()
    print("Total number of captions: ", len(captions))
    with open(os.path.join(output_dir, f"{args.split}.json"), 'w') as json_file:
        json.dump(captions, json_file)