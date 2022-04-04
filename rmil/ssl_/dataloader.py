import json
import PIL.Image as Image
import torch.utils.data as data
from torchvision import transforms as transforms
from typing import List, Dict

from .. import LabeledDataset
from . import config


class SslDataset(LabeledDataset):
    r"""Dataset class with region-level annotations"""

    def __init__(self, pre_transforms, transform, meta_data: List[Dict]):
        r"""Constructor for annotated region dataset

        Args:
            meta_data: dataset metadata
            transform: transform to apply on images
            crop_size: size of center crop
            image_size: the size to resize the images to
        """
        super(SslDataset, self).__init__(transform, meta_data)
        self.pre_transforms = pre_transforms

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = Image.open(filepath)
        image = self.pre_transforms[self.labels[idx]](image)
        image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.filepaths)


def get_ssl_data_loaders(meta_data_path, transform, transform_test,
                         crop_size=96, batch_size=8, num_workers=4):
    with open(meta_data_path, "r") as f:
        meta_data = json.load(f)
    pre_transforms = [
        transforms.Compose([
            transforms.Resize((image_size, image_size))
        ]) for image_size in config.IMAGE_SIZES]
    train_set = SslDataset(pre_transforms, transform, meta_data["train"])
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=True)
    val_set = SslDataset(pre_transforms, transform_test, meta_data["val"])
    val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return train_loader, val_loader
