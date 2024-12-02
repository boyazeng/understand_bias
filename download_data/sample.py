import pandas as pd
import argparse
import os
import random
from tqdm import tqdm
import csv
parser = argparse.ArgumentParser()
parser.add_argument(
    '--meta_dir',
    type=str,
    default="",
    help=('dir for the meta data (parquet)'))
parser.add_argument(
    '--num_samples',
    type=int,
    default=0,
    help='number of samples from the full set')
parser.add_argument(
    '--data',
    type=str,
    choices=["cc", "datacomp"],
    help="which dataset we are sampling"
)
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(42)

output_file = os.path.join(args.meta_dir, 'samples.tsv')
# Define the folder path and initialize variables
total_entries_dict = {"cc": 12_423_374, "datacomp": 1_387_173_656} # Total number of entries in all files
total_entries = total_entries_dict[args.data]
sampled_data = pd.DataFrame()  # DataFrame to hold sampled data

# Check if the folder exists
assert os.path.exists(args.meta_dir), "Folder not found."

# Generate random sample indices
sample_indices = random.sample(range(total_entries), args.num_samples)

sampled_data = pd.DataFrame()
current_index = 0
for file_name in tqdm(sorted(os.listdir(args.meta_dir)), desc="Load metadata files"):
        file_path = os.path.join(args.meta_dir, file_name)
        
        if args.data == "cc":
            with open(file_path, "r") as file:
                reader = csv.reader(file, delimiter='\t')
                all_urls = [row[0] for row in reader]
            df = pd.DataFrame({"URL": all_urls})
        elif args.data == "datacomp":
            df = pd.read_parquet(file_path, columns=["url"])
            df.rename({"url": "URL"}, axis = 1, inplace=True)

        file_entries = len(df)
        # Calculate which sample indices fall into the current file
        relevant_indices = [i - current_index for i in sample_indices if current_index <= i < current_index + file_entries]
        # Sample the data
        if relevant_indices:
            df_sample = df.iloc[relevant_indices]
            sampled_data = pd.concat([sampled_data, df_sample])

        current_index += file_entries

# Extract the 'URL' column and save it to a TSV file

url_data = sampled_data["URL"]
sampled_data.to_csv(output_file, sep='\t', index=False, header=False)
print(f"Sampled URL data saved to {output_file} with {len(url_data)} urls")
    