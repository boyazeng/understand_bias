import os
import shutil
import random
from tqdm import tqdm
import argparse


def split_data(args):
    train_dir = os.path.join(args.dest_dir, 'train')
    val_dir = os.path.join(args.dest_dir, 'val')

    # Create train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Recursively find all .jpg files within the data directory
    image_paths = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
                
    # Shuffle the list of image paths
    random.shuffle(image_paths)

    # Make sure there are enough images
    assert len(image_paths) >= args.num_train + args.num_val, "Not enough images to sample from."

    # Select 110k images
    selected_images = image_paths[:args.num_train + args.num_val]

    # Copy the first 100k images to the train directory
    for img_path in tqdm(selected_images[:args.num_train], desc="Moving train images"):
        shutil.move(img_path, train_dir)

    # Copy the remaining 10k images to the val directory
    for img_path in tqdm(selected_images[args.num_train:], desc="Moving val images"):
        shutil.move(img_path, val_dir)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--dest_dir', type=str, default='')
    parser.add_argument('--num_train', type=int, default=1_000_000)
    parser.add_argument('--num_val', type=int, default=10_000)
    args = parser.parse_args()
    split_data(args)