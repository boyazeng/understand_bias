from argparse import ArgumentParser
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import gc
import glob
import random
import mmcv
import mmcv_custom   # noqa: F401,F403
import mmseg_custom  # noqa: F401,F403
from mmcv.parallel import collate, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.apis.inference import LoadImage
from mmseg.core import get_classes
from mmseg.core.evaluation import get_palette
from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import numpy as np
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


from PIL import Image, PngImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)


import sys
current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['seg'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, interpolation=transforms.InterpolationMode.BICUBIC)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
        try:
            image = Image.open(path).convert("RGB")

            if min(image.size) > 500:
                image = self.preprocess(image) #[W,H,3]
            image = np.array(image, dtype=np.uint8)[:, :, ::-1] #[H,W,BGR]

            return {'image': image, 'save_path': save_path}
        except Exception as e:
            print(f"error to open {path} with error {e}")
            return {'image': None, 'save_path': save_path}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--num', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    print(args)
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    assert args.dataset in IMAGE_ROOTS, f"Dataset {args.dataset} not found in data_path.py"

    # build the model from a config file and a checkpoint file
    cfg = mmcv.Config.fromfile('configs/ade20k/mask2former_beitv2_adapter_large_896_80k_ade20k_ss.py')
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    test_pipeline = cfg.data.test.pipeline
    if test_pipeline[1]['flip']:
        test_pipeline[1]['flip'] = False
    cfg.data.test.pipeline = test_pipeline

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    
    COLOR_MAPPING = get_palette('ade')

    model = init_segmentor(cfg, checkpoint=None, device=f"cuda")
    # no worry about missing keys
    # https://github.com/czczup/ViT-Adapter/issues/74
    checkpoint = load_checkpoint(model, 'mask2former_beitv2_adapter_large_896_80k_ade20k.pth', map_location='cpu')
    model.CLASSES = get_classes('ade')
    model.eval()

    dataset = Image_Dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False, collate_fn=lambda x: x)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f'Semantic Segmentation {args.dataset} {args.split} split'
    for sample in metric_logger.log_every(dataloader, 10, header):
        image = sample[0]['image']
        save_path = sample[0]['save_path']
        try:
            # this might raise OOM error
            result = inference_segmentor(model, image)
            mask = result[0]

            h, w = mask.shape
            seg_image = 255 * np.ones((h, w, 3), dtype=np.uint8)
            for k in range(len(model.CLASSES)):
                seg_image[mask == k] = COLOR_MAPPING[k]

            seg_image = Image.fromarray(seg_image.astype('uint8'), 'RGB')
            seg_image.save(save_path)
            
            del image
            del result
            gc.collect()
            torch.cuda.empty_cache()
            metric_logger.synchronize_between_processes()  
        except RuntimeError as e:
            print(e, save_path)
            continue