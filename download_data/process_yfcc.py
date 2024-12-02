from tqdm import tqdm
import argparse
import random
import csv
from concurrent import futures
import os
import threading as th
import sqlite3
import pandas as pd
from queue import Queue
from hashlib import md5
import itertools
from urllib.parse import urlparse
from queue import Full, Empty

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--meta_dir', type=str, default="", 
                    help='Directory for the metadata (SQLite database file)')
parser.add_argument('--num_samples', type=int, default=0, 
                    help='Number of samples to extract')
parser.add_argument('--option', type=str, default="", choices=["flickr", "aws"], 
                    help='Option for downloading the dataset')
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(42)

TOTAL_NUM = 100_000_000

sample_indices = random.sample(range(TOTAL_NUM), args.num_samples)

def generate_rows_from_db(path):
    conn = sqlite3.connect(path)
    # get column names
    # some settings that hopefully speed up the queries
    conn.execute(f'PRAGMA query_only = YES')
    conn.execute(f'PRAGMA journal_mode = OFF')
    conn.execute(f'PRAGMA locking_mode = EXCLUSIVE')
    conn.execute(f'PRAGMA page_size = 4096')
    conn.execute(f'PRAGMA mmap_size = {4*1024*1024}')
    conn.execute(f'PRAGMA cache_size = 10000')
    # retrieve rows in order
    yield from conn.execute(f'select * from yfcc100m_dataset')

class Yielder(th.Thread):
    def __init__(self, gen, queue, end, error):
        super().__init__()
        self.daemon = True
        self.running = True
        self.gen = gen
        self.queue = queue
        self.end = end
        self.error = error

    def run(self):
        try:
            for obj in self.gen:
                if not self.running:
                    break
                while True:
                    try:
                        self.queue.put(obj, timeout=1)
                        break
                    except Full:
                        pass
            else:
                self.queue.put(self.end, timeout=1)
        finally:
            self.queue.put(self.error, timeout=1)

    def stop(self):
        self.running = False

def yield_threaded(gen):
    """
    Run a generator in a background thread and yield its
    output in the current thread.

    Parameters:
        gen: Generator to yield from.
    """
    end = object()
    error = object()
    queue = Queue(maxsize=3)
    yielder = Yielder(gen, queue, end, error)
    try:
        yielder.start()
        while True:
            try:
                obj = queue.get(timeout=1)
                if obj is end:
                    break
                if obj is error:
                    raise RuntimeError()
                yield obj
            except Empty:
                pass
    finally:
        yielder.stop()


def create_aws_url(url):
    BYTE_MAP = {'%02x' % v: '%x' % v for v in range(256)}
    h = md5(url.encode('utf-8')).hexdigest()
    file_name = ''.join(BYTE_MAP[h[x:x+2]] for x in range(0, 32, 2))
    first_three = file_name[:3]
    second_three = file_name[3:6]
    return os.path.join("https://multimedia-commons.s3-us-west-2.amazonaws.com/data/images/", first_three, second_three, file_name) + ".jpg"

def create_flickr_url(row):
    """
    Transforms a Flickr URL to
    "https://live.staticflickr.com/{server-id}/{id}_{o-secret}_o.{o-format}"
    See here for https://www.flickr.com/services/api/misc.urls.html
    """

    # Splitting the URL to get the necessary parts
    return os.path.join("https://live.staticflickr.com", str(row["serverid"]), f"{row['photoid']}_{row['secretoriginal']}_o.jpg")

# Parse the rows in parallel
rows = tqdm(yield_threaded(itertools.islice(generate_rows_from_db(os.path.join(args.meta_dir, "yfcc100m_dataset.sql")), args.num_samples)), total = args.num_samples)
df = pd.DataFrame(rows, columns=[
        "photoid", "uid", "unickname", "datetaken", "dateuploaded", "capturedevice", "title", "description",
        "usertags", "machinetags", "longitude", "latitude", "accuracy", "pageurl", "downloadurl", "licensename",
        "licenseurl", "serverid", "farmid", "secret", "secretoriginal", "ext", "marker",
    ])


print("sample urls from dataframe")
sample_indices = random.sample(range(len(rows)), args.num_samples)
if args.option == "flickr":
    urls = df.iloc[sample_indices].apply(create_flickr_url, axis = 1)
elif args.option == "aws":
    urls = df['downloadurl'].iloc[sample_indices].apply(create_aws_url)

# Save sampled rows to TSV
output_file = os.path.join(args.meta_dir, "samples.tsv")
urls.to_csv(output_file, sep='\t', index=False, header=False)

print(f"Sampled data saved to {output_file}")
