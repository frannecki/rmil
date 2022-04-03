# Multiple Instance Learning for cervical biopsy cancer grading
## Summary
This repo combines multiple instance learning on bags, supervised 
learning on small dataset and self-supervised learning for auxiliary tasks

## Usage
### Dataset Preparation
#### Multiple instance learning
```sh
# The dataset for multiple instance learning is organized in bags.
# Each bag includes a group of patches from a single whole slide image
python preprocess/dataset_split_mil.py
```

#### supervised learning on small dataset
```sh
# The dataset is placed in folder $data_root_dir_aux (`data/aux` in default)
# The dataset should be organized as follows
# $data_root_dir
# ├── 0
# ├── 1
# ├── 2
# └── 3
python preprocess/dataset_split_aux.py --data_root_dir=$data_root_dir_aux
```

#### self-supervised learning
```sh
# Assuming the images are in the folder $data_root_dir_ssl (`data/ssl` in default)
python preprocess/dataset_generation_ssl.py --data_root_dir=$data_root_dir_ssl
```

### Training and testing
Please use `python main.py -h` to view usages.
```sh
# For training
python main.py --model_dir=models/models --log_dir=logs/logs --attn [--aux --split_path_aux data/train_test_split_aux.json] [--ssl --split_path_ssl data/train_test_split_ssl.json] | tee -a logs/logs.log

# For testing, specify `--testonly` to skip training
python main.py --model_dir=models/models --attn --testonly
```
