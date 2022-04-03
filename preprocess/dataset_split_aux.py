import os
import os.path as osp
import json
import argparse
from typing import Dict
from sklearn.model_selection import train_test_split

import sys
sys.path.append("./")
from rmil import config

def dataset_split(data_root_dir: str) -> Dict:
    """Example dataset structure"""
    # $data_root_dir
    # ├── 0
    # ├── 1
    # ├── 2
    # └── 3
    train_images, val_images = [], []
    for category in range(config.NUM_CLASSES):
        category_dir = osp.join(data_root_dir, f"{category}")
        images = [
            {
                "filepath": osp.join(category_dir, filename),
                "label": category
            } for filename in os.listdir(category_dir)]
        train, val = train_test_split(images, test_size=.2)
        train_images.extend(train)
        val_images.extend(val)
    split = {"train": train_images, "val": val_images}
    return split


parser = argparse.ArgumentParser()
parser.add_argument("--data_root_dir", type=str, default="data/aux",
                    help='image data root directory')
parser.add_argument("--output_path", type=str,
                    default=config.TRAIN_TEST_SPLIT_AUX,
                    help='filepath of train/val split config')


if __name__ == '__main__':
    args = parser.parse_args()
    split = dataset_split(args.data_root_dir)
    with open(args.output_path, 'w') as f:
        json.dump(split, f)
