import argparse
import gc
import glob
import json
import matplotlib.colors as mcolors
import numpy as np
import os
import random
import sys
import torch
import torch.backends.cudnn as cudnn
import torch.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import MetadataCatalog
from detectron2.data.transforms import ResizeShortestEdge
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.utils.visualizer import Visualizer
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms

from PIL import Image, PngImagePlugin, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024**2)

current_file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("understand_bias") + 1]))
from data_path import IMAGE_ROOTS, SAVE_ROOTS
import transformations.trans_utils as utils


def rgb_to_mpl_color(rgb_tuple):
    # Normalize the RGB values to 0-1
    normalized_rgb = tuple(x/255 for x in rgb_tuple)
    return mcolors.to_rgba(normalized_rgb)

class Image_Dataset(Dataset):
    def __init__(self, args):
        self.output_dir = os.path.join(SAVE_ROOTS['object_det'], args.dataset, args.split)
        os.makedirs(self.output_dir, exist_ok=True)
        self.root = os.path.join(IMAGE_ROOTS[args.dataset], args.split)
        self.paths = glob.glob(os.path.join(self.root, "*.jpg")) + \
            glob.glob(os.path.join(self.root, "*.png")) + glob.glob(os.path.join(self.root, "*.JPEG"))
        if args.num is not None:
            self.paths = self.paths[:args.num]
        assert len(self.paths) == args.num, f"not enough images in {args.dataset} {args.split} split"
        self.preprocess = transforms.Resize(500, interpolation=transforms.InterpolationMode.BICUBIC)
        self.transform = ResizeShortestEdge(short_edge_length=1024, max_size=1024)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        save_path = os.path.join(self.output_dir, os.path.splitext(os.path.basename(path))[0] + '.png')
        try:
            image = Image.open(path).convert("RGB")

            if min(image.size) > 500:
                image = self.preprocess(image) #[W,H,3]
            image = np.array(image, dtype=np.uint8) #[H,W,3]
            h, w, c = image.shape

            t = self.transform.get_transform(image)
            image = t.apply_image(image)
            image = torch.tensor(image).permute(2, 0, 1)
            return {'image': image, 'height': h, 'width': w, 'save_path': save_path}
        except Exception as e:
            print(f"error to open {path} with error {e}")
            return {'image': None, 'height': 0, 'width': 0, 'save_path': save_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[utils.get_args_parser()], add_help=False)
    parser.add_argument('--dataset', type=str, default="cc")
    parser.add_argument('--split', type=str, choices=["train", "val"], default="val")
    parser.add_argument('--batch_size', type=int, default=1)
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

    det_checkpoint = os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("object_det") + 1], f"VitDet_huge_LVIS.pkl")
    det_config = os.path.join(os.sep, *current_file_path.split(os.sep)[:current_file_path.split(os.sep).index("object_det") + 1], f"detectron2/projects/ViTDet/configs/LVIS/cascade_mask_rcnn_vitdet_h_100ep.py")

    cudnn.benchmark = True
    cfg = LazyConfig.load(det_config)
    cfg.train.init_checkpoint = det_checkpoint
    for i in range(3):
        cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5 # there are three
    
    vitdet = instantiate(cfg.model)
    DetectionCheckpointer(vitdet).load(det_checkpoint)
    vitdet.to("cuda")
    vitdet.eval()

    with open(f"LVIS_color_map.json", "r") as f:
        color_dict = json.load(f)
    all_colors = list(color_dict.values())
    all_classes = list(color_dict.keys())

    dataset = Image_Dataset(args)
    sampler_train = DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False, seed=42,
        )
    data_loader = DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=16,
        collate_fn=lambda x: x,
        drop_last=False,
    )

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('box_empty_rate', utils.SmoothedValue(window_size=100, fmt='{value:.2f}'))
    header = f'Object detection {args.dataset} {args.split} split'
    with torch.no_grad():
        for input_batch in metric_logger.log_every(data_loader, 10, header):
            tmp = []
            for i_d in input_batch:
                # skip images that can't be loaded
                if i_d['image'] is None:
                    continue
                tmp.append(i_d)
            if len(tmp) == 0:
                metric_logger.update(box_empty_rate=1)
                metric_logger.synchronize_between_processes()
                continue
            input_batch = tmp
            
            # perform object detection
            output_batch = vitdet(input_batch)
            
            box_empty_rate = 0
            for i_d, o_d in zip(input_batch, output_batch):
                pred_bboxes = o_d['instances'].pred_boxes.tensor
                classes = o_d['instances'].pred_classes.tolist()
                scores = o_d['instances'].scores.tolist()
                # save white image for images that have no objects detected
                if len(pred_bboxes) == 0:
                    # nothing detected
                    box_empty_rate += 1
                    
                    image = Image.fromarray(255*np.ones((i_d['height'], i_d['width'], 3), dtype=np.uint8))
                    image.save(os.path.join(i_d['save_path']))
                else:
                    # save detection results
                    pred_bboxes = pred_bboxes.cpu().numpy().tolist()
                    vis = Visualizer(255*np.ones((i_d['height'], i_d['width'], 3), dtype=np.uint8))
            
                    for bbox, c in zip(pred_bboxes, classes):
                        color = rgb_to_mpl_color(all_colors[c])
                        # Draw rectangle and text annotations
                        vis.draw_box(bbox[:4], alpha=1, edge_color=color)

                        x_center = (bbox[0] + bbox[2]) / 2
                        y = min(bbox[1], bbox[3])
                        vis.draw_text(all_classes[c], (x_center, y), color=color)

                    # see demo here https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
                    # we don't invert rgb channel since we use pil
                    image = Image.fromarray(vis.get_output().get_image()[:, :, :])
                    image.save(os.path.join(i_d['save_path']))
            
            metric_logger.update(box_empty_rate=box_empty_rate/len(output_batch))
            # free memory
            del input_batch
            del output_batch
            gc.collect()
            torch.cuda.empty_cache()
            metric_logger.synchronize_between_processes()

                    