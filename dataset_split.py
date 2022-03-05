import os
import json
import glob
import argparse
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import train_test_split
import config


def dataset_split(train_labels_csv: str) -> List[Dict]:
    r"""Get train tile filenames based on train metadata
    
    Args:
        train_labels_csv:   filepath of train metadata csv
    """
    df = pd.read_csv(train_labels_csv)
    filenames = df["filename"].values
    slide_ids = [filename.split('.')[0] for filename in filenames]
    annotations = []
    for cls in range(config.NUM_CLASSES):
        annotations.append(df[str(cls)].values)
    labels = []
    for i, filename in enumerate(filenames):
        label = -1
        for cls in range(config.NUM_CLASSES):
            if annotations[cls][i] == 1:
                label = cls
                break
        assert label != -1
        labels.append(label)
    
    slides = []
    for slide_id, label in zip(slide_ids, labels):
        slides.append({"id": slide_id,
                       "label": label})
    train_slides, test_slides = train_test_split(slides, test_size=0.3)
    val_slides, test_slides = train_test_split(test_slides, test_size=0.5)
    split = {"train": train_slides, "val": val_slides,
                        "test": test_slides}
    return split


parser = argparse.ArgumentParser()
parser.add_argument("--train_labels_csv", type=str,
                    default=config.TRAIN_LABELS_CSV,
                    help='filepath of train labels (csv)')
parser.add_argument("--output_path", type=str,
                    default=config.TRAIN_TEST_SPLIT,
                    help='filepath of train/val/test split config')


if __name__ == '__main__':
    args = parser.parse_args()
    train_test_dict = dataset_split(args.train_labels_csv)
    with open(args.output_path, 'w') as f:
        json.dump(train_test_dict, f)