import os
import os.path as osp
import json
import config
import random


if __name__ == '__main__':
    filenames = os.listdir(config.DATA_ROOT_DIR)
    filenames = random.sample(filenames, int(len(filenames) * 0.5))
    filepaths = [osp.join(config.DATA_ROOT_DIR, filename)
                 for filename in filenames]
    transform_ids = [random.randint(0, 4) for i in range(len(filenames))]
    labels = [int(tfm_id == 2) for tfm_id in transform_ids]
    data = [{"filepath": filepath, "tfm_id": tfm_id, "label": label} for
            filepath, tfm_id, label in zip(filepaths, transform_ids, labels)]
    indices = [i for i in range(len(filenames))]
    indices_train = random.sample(indices, int(0.7 * len(filenames)))
    indices_train = sorted(indices_train)
    result_split = {"train": [], "val": []}
    i, j = 0, 0
    while i < len(indices):
        if j < len(indices_train) and i == indices_train[j]:
            result_split["train"].append(data[i])
            j += 1
        else:
            result_split["val"].append(data[i])
        i += 1
    with open("train_test_split.json", "w") as f:
        json.dump(result_split, f)
