## Caption

### Install environment
```
conda create -n bias_caption
conda activate bias_caption
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -c conda-forge transformers accelerate
pip install bitsandbytes pillow scikit-image opencv-python
```

### Perform transformation
By default, we use 4 bit quantization to generate captions with a batch size of 4. This ensures that the model can be loaded on a GPU with 11GB memory. If you have a GPU with more memory, you can use full precision inference or/and a larger batch size.

Set `dataset` to the name of the dataset in `data_path.py` and `caption_type` to `short` or `long`.

To transform the training set:
```
python transform.py --dataset $dataset --caption_type $caption_type --split train --num 1_000_000
```

To transform the validation set:
```
python transform.py --dataset $dataset --caption_type $caption_type --split val --num 10_000
```

