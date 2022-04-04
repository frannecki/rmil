import os
import os.path as osp
import json
import copy
import torch
from tqdm import tqdm
import PIL.Image as Image
import torch.utils.data as data
from typing import List, Dict, Tuple


def get_train_data_tiles(split_path: str) -> Tuple[List[Dict]]:
    r"""Load train/val/test split result

    Args:
        split_path：path to split json file
    """
    tasks = ["train", "val", "test"]
    with open(split_path, 'r') as f:
        data = json.load(f)
    slides = []
    for task in tasks:
        slides.append(data[task])
    return tuple(slides)


class MILBagDataset(data.Dataset):
    r"""Bag dataset for multiple-instance learning
    """
    def __init__(self, data_root_dir, slides,
                 topk, transform, cat=False):
        r"""Constructor for MILBagDataset

        Args:
            data_root_dir: the path of the images of train/val/test datasets
            slides: metadata of the whole slide images for training
        """
        super(MILBagDataset, self).__init__()
        self.cat, self.topk, self.transform = cat, topk, transform
        self.data_root_dir, self.labels, self.tiles = data_root_dir, [], []
        for slide in tqdm(slides):
            filename_tiles = os.listdir(osp.join(data_root_dir, slide["id"]))
            filename_tiles = sorted(
                filename_tiles,
                key=lambda x: int(x.split('_')[-1].split('.')[0]))
            filename_tiles = filename_tiles[:self.topk]
            filenames = copy.deepcopy(filename_tiles)
            while len(filenames) < self.topk:
                filenames.extend(filename_tiles[:(self.topk - len(filenames))])
            filenames = [osp.join(data_root_dir, slide["id"], filename)
                         for filename in filenames]
            self.tiles.append(filenames)
            self.labels.append(slide['label'])

    def __getitem__(self, idx):
        images = []
        filenames = self.tiles[idx]
        for filepath in filenames:
            image = Image.open(filepath)
            image = self.transform(image)
            images.append(image)
        if self.cat:
            group = torch.cat(images, dim=0)
        else:
            group = torch.stack(images, dim=0)
        return group, self.labels[idx]

    def __len__(self):
        return len(self.labels)


def get_bag_dataloaders(data_root_dir, split_path, topk,
                        transform, transform_test,
                        batch_size=1, num_workers=1, cat=False):
    train_slides, val_slides, test_slides = get_train_data_tiles(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides, topk,
                             transform=transform, cat=cat)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True)
    valset = MILBagDataset(data_root_dir, val_slides, topk,
                           transform=transform_test, cat=cat)
    valloader = data.DataLoader(valset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
    testset = MILBagDataset(data_root_dir, test_slides, topk,
                            transform=transform_test, cat=cat)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return trainloader, valloader, testloader


def get_bag_datasets(data_root_dir, split_path, topk, crop_size,
                     transform, transform_test, cat=False):
    train_slides, val_slides, test_slides = get_train_data_tiles(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides, topk,
                             crop_size, transform=transform, cat=cat)
    valset = MILBagDataset(data_root_dir, val_slides, topk,
                           crop_size, transform=transform_test, cat=cat)
    testset = MILBagDataset(data_root_dir, test_slides, topk,
                            crop_size, transform=transform_test, cat=cat)
    return trainset, valset, testset


class LabeledDataset(data.Dataset):
    r"""Dataset class with region-level annotations
    (for supervised learning and ssl)
    """
    def __init__(self, transform, meta_data: List[Dict]):
        r"""Constructor for annotated region dataset

        Args:
            transform: transform to apply on images
        """
        super(LabeledDataset, self).__init__()
        self.transform = transform
        self.filepaths = [sample["filepath"] for sample in meta_data]
        self.labels = [sample["label"] for sample in meta_data]

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = Image.open(filepath)
        image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.filepaths)


def get_labeled_data_loaders(meta_data, transform, transform_test,
                             batch_size=8, num_workers=2):
    train_set = LabeledDataset(transform, meta_data["train"])
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   num_workers=num_workers, shuffle=True)
    val_set = LabeledDataset(transform_test, meta_data["val"])
    val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return train_loader, val_loader
