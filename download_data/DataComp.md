# DataComp

Below is the instruction to download the DataComp dataset.

## Download the dataset

Fill the root where you want to store the dataset. We recommend to download 1.5M images since some of downloads might fail. 

Note that our default setting downloads all 2663 metadata files for DataComp-1B and sample urls from the combined metadata file. It takes about 1 day to download all the metadata files and parse the urls. To speed up the process, you can download parts of the metadata file and sample urls from these metadata files.

```
root="" # the root of the dataset

meta_dir="${root}/meta"
data_dir="${root}/data"
mkdir -p $meta_dir
mkdir -p $data_dir

# download the metadata file
for i in {0000..2663}; do
    filename="${i}.parquet"original filename
    wget -O "${meta_dir}/${filename}" "https://huggingface.co/datasets/mlfoundations/datacomp_1b/resolve/refs%2Fconvert%2Fparquet/default/train/${filename}"
done

python sample.py --meta_dir $meta_dir --data datacomp --num_samples 1500000
python download_with_resize.py --tsv_path ${meta_dir}/samples.tsv --data_dir $data_dir
```