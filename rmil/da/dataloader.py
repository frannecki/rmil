import json
import torch.utils.data as data

from .. import LabeledDataset


class DomainDataset(LabeledDataset):
    pass


def get_da_data_loaders(meta_data_path, transform, transform_test,
                        batch_size=8, num_workers=4):
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    train_set = DomainDataset(transform, meta_data["train"])
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=True)
    val_set = DomainDataset(transform_test, meta_data["val"])
    val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return train_loader, val_loader
