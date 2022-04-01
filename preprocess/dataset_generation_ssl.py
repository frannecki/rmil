import os
import os.path as osp
import json
import argparse
import random
from sklearn.model_selection import train_test_split

from rmil.ssl_ import config

parser = argparse.ArgumentParser()
parser.add_argument('--meta_data_path', type=str,
                    default=config.META_DATA_PATH,
                    help="train test split path for self-supervised learning")
parser.add_argument('--data_root_dir', type=str, default="data/ssl",
                    help="root directory of data for self-supervised learning")
args = parser.parse_args()

if __name__ == '__main__':
    filenames = os.listdir(args.data_root_dir)
    filenames = random.sample(filenames, int(len(filenames) * 0.5))
    filepaths = [osp.join(args.data_root_dir, filename)
                 for filename in filenames]
    transform_ids = [random.randint(0, len(config.IMAGE_SIZES-1))
                     for _ in range(len(filenames))]
    data = [{"filepath": filepath, "label": tfm_id} for
            filepath, tfm_id in zip(filepaths, transform_ids)]
    train_data, val_data = train_test_split(data, test_size=.2)
    with open(args.meta_data_path, "w") as f:
        json.dump({'train': train_data, 'val': val_data}, f)
