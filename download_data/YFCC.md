# YFCC

Below is the instruction to download the YFCC dataset.

## Download the dataset

Fill the root where you want to store the dataset. We recommend to download 1.5M images since some of downloads might fail. It requires about ~400GB disk space to store 1M images.

There are two options to download the dataset:
1. Download from Flickr directly (`--option flickr`). This option recently fails due to the rate limit of Flickr website. Our original results are based on this option.
2. Download from Amazon S3 (`--option aws`). This option is more stable but the resized method is different from the original paper. They by default store the images resized to longer edge of 500px (not shorter edge).

```
root="" # the root of the dataset

meta_dir="${root}/meta"
data_dir="${root}/data"
mkdir -p $meta_dir
mkdir -p $data_dir

# download the metadata file
wget -O ${meta_dir}/yfcc100m_dataset.sql https://multimedia-commons.s3-us-west-2.amazonaws.com/tools/etc/yfcc100m_dataset.sql

python process_yfcc.py --meta_dir $meta_dir --option $option --num_samples 1500000 
python download_with_resize.py --tsv_path ${meta_dir}/samples.tsv --data_dir $data_dir
```

## Acknowledgement
The download script is adapted from [here](https://gitlab.com/jfolz/yfcc100m/-/tree/master?ref_type=heads).
