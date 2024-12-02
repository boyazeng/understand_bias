## Instructions for downloading the datasets

### Install environment
Please use the same environment for dataset classification.

### Downloading the datasets
Please check `YFCC.md`, `CC.md`, and `Datacomp.md` for downloading the YFCC, CC, and Datacomp datasets.

Each downloaded dataset will have the following structure:
```
root/
    data/
        0/
            *.png
        1/
            *.png
        ...
    meta/
        metadata_file
        samples.tsv # sampled urls
```
Note in each directory, there are at most 10K images. This ensures that when you open any editor on the `data` folder, your editor will not get stuck due to the large number of files.

### Split the dataset into train/val
Replace `$data_dir` with the path to the dataset and `$dest_dir` with the path to the destination directory.
By default, we use 1M / 10K images to form the train / val sets.
This scirpt keeps remaining images (other than the train / val sets) in the original directory.
```
python split.py --data_dir $data_dir --dest_dir $dest_dir --num_train $num_train --num_val $num_val
```
Your dataset will have the following structure:
```
$dest_dir/
    train/
        *.png
    val/
        *.png
```
