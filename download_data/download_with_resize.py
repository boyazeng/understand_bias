import argparse
import csv
import os
import random
import requests
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from requests.exceptions import Timeout
from torchvision import transforms
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    '--tsv_path',
    type=str,
    default="",
    help=('path to the tsv file'))
parser.add_argument(
    '--data_dir',
    type=str,
    default="",
    help=('dir to store data'))
args = parser.parse_args()

# Create the root folder
os.makedirs(args.data_dir, exist_ok=True)

# Seed for reproducibility
random.seed(42)

# Read all image URLs into a list
all_urls = []
with open(args.tsv_path, "r") as tsvfile:
    reader = csv.reader(tsvfile, delimiter='\t')
    all_urls = [row[0] for row in reader]
    print(f"Try to download total {len(all_urls)} images from urls")

# shuffle all urls so that each directory contains random images
random.shuffle(all_urls)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
}

SUCCESSFUL_COUNT = 0
trans = transforms.Resize(500, interpolation=transforms.InterpolationMode.BICUBIC)

def download_image(i, img_url, progress_bar):
    global SUCCESSFUL_COUNT
    progress_bar.update(1)
    try:
        img_data = requests.get(img_url, headers=headers, timeout=10).content

        # Group every 10k download trial into a subfolder
        subfolder = str(i // 10000)
        subfolder_path = os.path.join(args.data_dir, subfolder)

        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        image_path = os.path.join(subfolder_path, f"image_{i}.png")

        with open(image_path, "wb") as img_file:
            img_file.write(img_data)

        try:
            with Image.open(image_path) as img:
                img.load()  # Load image data to verify the image
                # resize the image to shorter side 500
                if min(img.size) > 500:
                    if img.mode == "CMYK":
                        img = img.convert("RGB")
                    img = trans(img)
                    output_path = os.path.splitext(image_path)[0] + '.png'
                    img.save(output_path, 'PNG')
                    os.remove(image_path)
            SUCCESSFUL_COUNT += 1
            
            progress_bar.set_description(f"Successful: {SUCCESSFUL_COUNT}")

        except Exception as e:
            print(f"Image index {i} failed to verify. Deleting. url: {img_url}. Error: {e}")
            os.remove(image_path)

    except Exception as e:
        print(f"An error occurred for image index {i}: {e}, url: {img_url}")
        pass

# Initialize the tqdm progress bar
progress_bar = tqdm(total=len(all_urls), desc='Downloading images', leave=True)

# Download the images in parallel
with futures.ThreadPoolExecutor(max_workers=80) as executor:
    executor.map(download_image, range(len(all_urls)), all_urls, [progress_bar for _ in range(len(all_urls))])

progress_bar.close()