# CC

Below is the instruction to download the CC dataset.

## Download the dataset

Fill the root where you want to store the dataset. We recommend to download 1.5M images since some of downloads might fail. 

```
root="" # the root of the dataset

meta_dir="${root}/meta"
data_dir="${root}/data"
mkdir -p $meta_dir
mkdir -p $data_dir

# download the metadata file
wget -O ${meta_dir}/cc12m.tsv https://storage.googleapis.com/conceptual_12m/cc12m.tsv

python sample.py --meta_dir $meta_dir --data cc --num_samples 1500000
python download_with_resize.py --tsv_path ${meta_dir}/samples.tsv --data_dir $data_dir
```