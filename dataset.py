import glob
import json
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from data_path import IMAGE_ROOTS, SAVE_ROOTS

class SimpleDataset(Dataset):
    def __init__(self, root, name, trans, num_samples, transformation=None, patch_size=16, freq_thres=40, filter_order=2, filter_ideal=True):
        self.paths = glob.glob(os.path.join(root, "*.jpg")) + \
            glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.JPEG"))
        self.paths = sorted(self.paths)
        if num_samples is not None:
            self.paths = self.paths[:num_samples]
        self.name = name
        self.transformation = transformation
        self.patch_size = patch_size
        if os.path.exists("randperm_pixel.pth"):
            self.rand_perm_pixel = torch.load("randperm_pixel.pth")
        else:
            self.rand_perm_pixel = torch.randperm(224 * 224)
            torch.save(self.rand_perm_pixel, "randperm_pixel.pth")
        if os.path.exists(f"randperm_patch{patch_size}.pth"):
            self.rand_perm_patch = torch.load(f"randperm_patch{patch_size}.pth")
        else:
            self.rand_perm_patch = torch.randperm(224 // patch_size * 224 // patch_size)
            torch.save(self.rand_perm_patch, f"randperm_patch{patch_size}.pth")
        if self.transformation == "hpf":
            if not filter_ideal:
                self.highpass_filter = ButterworthHighPass(224, 224, freq_thres, filter_order)
                print(f"using butterworth high-pass filter of threshold {freq_thres} and filter order {filter_order}")
            else:
                self.highpass_filter = IdealHighPass(224, 224, freq_thres)
                print(f"using ideal high-pass filter of threshold {freq_thres}")
        elif self.transformation == "lpf":
            if not filter_ideal:
                self.lowpass_filter = ButterworthLowPass(224, 224, freq_thres, filter_order)
                print(f"using butterworth low-pass filter of threshold {freq_thres} and filter_order {filter_order}")
            else:
                self.lowpass_filter = IdealLowPass(224, 224, freq_thres)
                print(f"using ideal low-pass filter of threshold {freq_thres}")
        self.trans = trans
        print(f"SimpleDataset(Name: {self.name}, Samples: {len(self.paths)}, Transformation: {self.transformation})")        
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        while True:
            path = self.paths[i]
            try:
                if self.transformation == "sam" or self.transformation == "canny":
                    with open(path, 'rb') as f:
                        img = Image.open(f).convert('L')
                    return self.trans(img).repeat(3, 1, 1)
                img = pil_loader(path)
                if self.transformation == "mean_rgb":
                    return self.trans(calc_mean_rgb(img))
                if self.transformation == "hpf":
                    return HighPassFilter(self.trans(img), self.highpass_filter).repeat(3, 1, 1)
                if self.transformation == "lpf":
                    return LowPassFilter(self.trans(img), self.lowpass_filter).repeat(3, 1, 1)
                if "pixel_shuffle" in self.transformation:
                    return shuffle_image_pixels(self.trans(img), fixed=("fixed" in self.transformation), rand_perm_pixel=self.rand_perm_pixel)
                if "patch_shuffle" in self.transformation:
                    return shuffle_image_patches(self.trans(img), fixed=("fixed" in self.transformation), patch_size=self.patch_size, rand_perm_patch=self.rand_perm_patch)
                return self.trans(img)
            except Exception as e:
                print("exception in __getitem__: ", e)
            i = np.random.choice(np.arange(len(self.paths)))

class VAEDataset(Dataset):
    def __init__(self, root, original_root, name, trans, num_samples):
        self.paths = glob.glob(os.path.join(root, "*.jpg")) + \
            glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.JPEG"))
        self.paths = sorted(self.paths)
        self.original_paths = glob.glob(os.path.join(original_root, "*.jpg")) + \
            glob.glob(os.path.join(original_root, "*.png")) + glob.glob(os.path.join(original_root, "*.JPEG"))
        self.original_paths = sorted(self.original_paths)
        if num_samples is not None:
            self.paths = self.paths[:num_samples]
            self.original_paths = self.original_paths[:num_samples]
        self.name = name
        self.trans = trans
        print(f"VAEDataset(Name: {self.name}, Samples: {len(self.paths)})")
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        while True:
            path = self.paths[i]
            original_path = self.original_paths[i]
            try:
                img = pil_loader(path)
                original_img = pil_loader(original_path)
                width, height = original_img.size
                resize_back = transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BICUBIC)
                img = resize_back(img)
                return self.trans(img)
            except Exception as e:
                print("exception in __getitem__: ", e)
            i = np.random.choice(np.arange(len(self.paths)))

class CaptionDataset(Dataset):
    def __init__(self, root, name, num_samples):
        self.name = name
        self.captions = list(json.load(open(root, "r")).values())
        if num_samples is not None:
            self.captions = self.captions[:num_samples]
        print(f"CaptionDataset(Name: {self.name}, Samples: {len(self.captions)})")

    def __getitem__(self, i):
        return self.captions[i]

    def __len__(self):
        return len(self.captions)

class CompositeDataset(Dataset):
    def __init__(self, names, datasets):
        self.datasets = datasets
        
        indices = [np.random.permutation(len(ds)) for ds in datasets]
        self.indices = np.concatenate(indices, axis = 0)

        targets = [np.full(len(ds), i) for i, ds in enumerate(datasets)]
        self.targets = np.concatenate(targets, axis = 0)

        print("Names of the datasets:", names)
        print("Length of each dataset:", [len(ds) for ds in datasets])

    def __len__(self):
        return sum([len(d) for d in self.datasets])

    def __getitem__(self, i):
        if len(self.targets.shape) > 1:
            y = self.targets[i]
        else:
            y = self.targets[i].item()
        x = self.datasets[y][self.indices[i]]
        return x, y

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def IdealHighPass(h, w, d0):
    x = np.arange(w) - w // 2
    y = np.arange(h) - h // 2
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    H = np.where(d > d0, 1, 0)
    return H
def IdealLowPass(h, w, d0):
    x = np.arange(w) - w // 2
    y = np.arange(h) - h // 2
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    H = np.where(d <= d0, 1, 0)
    return H

def ButterworthHighPass(h, w, d0, n):
    x = np.arange(w) - w // 2
    y = np.arange(h) - h // 2
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    d[d == 0] = 1e-5
    H = 1 / (1 + (d0 / d)**(2 * n))
    return H
def ButterworthLowPass(h, w, d0, n):
    x = np.arange(w) - w // 2
    y = np.arange(h) - h // 2
    x, y = np.meshgrid(x, y)
    d = np.sqrt(x**2 + y**2)
    d[d == 0] = 1e-5
    H = 1 / (1 + (d / d0)**(2 * n))
    return H

grayscale_transform = transforms.Grayscale(num_output_channels=1)
def HighPassFilter(image, highpass_filter) -> torch.Tensor:
    image = grayscale_transform(image).numpy()
    f = fft2(image)
    fshift = fftshift(f)
    filtered = highpass_filter * fshift
    ishift = ifftshift(filtered)
    img_back = ifft2(ishift)

    img_back = np.real(img_back)
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-5)
    return torch.from_numpy(img_back.astype(np.float32))
def LowPassFilter(image, lowpass_filter) -> torch.Tensor:
    image = grayscale_transform(image).numpy()
    f = fft2(image)
    fshift = fftshift(f)
    filtered = lowpass_filter * fshift
    ishift = ifftshift(filtered)
    img_back = ifft2(ishift)

    img_back = np.real(img_back)
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back) + 1e-5)
    return torch.from_numpy(img_back.astype(np.float32))

def calc_mean_rgb(image: Image.Image) -> Image.Image:
    data = np.array(image)
    mean_values = data.mean(axis=(0, 1), keepdims=True)
    return Image.fromarray(np.full(data.shape, mean_values, dtype=data.dtype))

def shuffle_image_pixels(img, fixed, rand_perm_pixel) -> torch.Tensor:
    c, h, w = img.shape
    total_pixels = h * w
    pixel_perm = rand_perm_pixel if fixed else torch.randperm(total_pixels)
    flat_img = img.view(c, total_pixels)
    shuffled_flat_img = flat_img[:, pixel_perm]
    shuffled_img = shuffled_flat_img.view(c, h, w)
    return shuffled_img

def shuffle_image_patches(img, fixed, patch_size, rand_perm_patch) -> torch.Tensor:
    num_patches = 224 // patch_size
    image_patches = img.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    image_patches = image_patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, 3, patch_size, patch_size)
    patch_perm = rand_perm_patch if fixed else torch.randperm(num_patches ** 2)
    shuffled_image_patches = image_patches[patch_perm]
    shuffled_image = shuffled_image_patches.view(num_patches, num_patches, 3, patch_size, patch_size)
    shuffled_image = shuffled_image.permute(2, 0, 3, 1, 4).contiguous().view(3, 224, 224)
    return shuffled_image

def build_dataset(is_train, args):
    transform = build_transform(is_train and (not args.eval), args)
    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.transformation in ["reference", "mean_rgb", "hpf", "lpf"] or "shuffle" in args.transformation:
        names = args.data_names.split(',')
        roots = [os.path.join(IMAGE_ROOTS[name], 'train' if is_train else 'val') for name in names]
        dataset = [SimpleDataset(root, name, transform, args.num_samples, transformation=args.transformation, patch_size=args.patch_size, freq_thres=args.freq_thres, filter_order=args.filter_order, filter_ideal=args.filter_ideal) \
                   for root, name in zip(roots, names)]
        nb_classes = len(dataset)
        dataset = CompositeDataset(names, dataset)
    elif args.transformation in ["text_to_image", "uncon_gen", "depth", "hog", "sift", "sam", "seg", "object_det", "canny"]:
        names = args.data_names.split(',')
        roots = [os.path.join(SAVE_ROOTS[args.transformation], name, 'train' if is_train else 'val') for name in names]
        dataset = [SimpleDataset(root, name, transform, args.num_samples, transformation=args.transformation, patch_size=args.patch_size, freq_thres=args.freq_thres, filter_order=args.filter_order, filter_ideal=args.filter_ideal) \
                   for root, name in zip(roots, names)]
        nb_classes = len(dataset)
        dataset = CompositeDataset(names, dataset)
    elif args.transformation == 'vae':
        names = args.data_names.split(',')
        roots = [os.path.join(SAVE_ROOTS['vae'], name, 'train' if is_train else 'val') for name in names]
        original_roots = [os.path.join(IMAGE_ROOTS[name], 'train' if is_train else 'val') for name in names]
        dataset = [VAEDataset(root, original_root, name, transform, args.num_samples) \
                   for root, original_root, name in zip(roots, original_roots, names)]
        nb_classes = len(dataset)
        dataset = CompositeDataset(names, dataset)
    elif args.transformation == "caption":
        names = args.data_names.split(',')
        roots = [os.path.join(SAVE_ROOTS['caption'], args.caption_type, name, 'train.json' if is_train else 'val.json') 
                 for name in names]
        dataset = [CaptionDataset(root, name, args.num_samples) for root, name in zip(roots, names)]
        nb_classes = len(dataset)
        dataset = CompositeDataset(names, dataset)
    else:
        raise NotImplementedError()
    print("Number of classes = %d" % nb_classes)
    
    import time
    time.sleep(5)
    return dataset, nb_classes


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    if is_train:
        if "canny" in args.transformation or "seg" in args.transformation:
            return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            ])
        elif args.transformation in ["hpf", "lpf", "sam", "depth", "hog", "sift"]:
            return transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        else:
            # randomaug with random resized crop
            return create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
    else:
        t = [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]

        if args.transformation not in ["hpf", "lpf", "canny", "sam", "depth", "hog", "sift"]:
            t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)