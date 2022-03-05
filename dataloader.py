import glob
import json
import copy
import torch
from tqdm import tqdm
import PIL.Image as Image
import torch.utils.data as data
from typing import List, Dict, Tuple
import torchvision.transforms as transforms


def get_train_data(split_path: str) -> Tuple[List[Dict]]:
    r"""Load train/val/test split result

    Args:
        split_path: path to the json file containing train/val/test
            data directories
    """
    with open(split_path, 'r') as f:
        data = json.load(f)
    train_slides = data["train"]
    val_slides = data["val"]
    test_slides = data["test"]
    return train_slides, val_slides, test_slides


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
    return *slides,


class MILBagDataset(data.Dataset):
    r"""Bag dataset for multiple-instance learning
    """
    def __init__(self, data_root_dir, slides, topk, image_crop_size,
                 transform=None, image_size=256, cat=False):
        r"""Constructor for MILBagDataset

        Args:
            data_root_dir: the path of the images of train/val/test datasets
            slides: metadata of the whole slide images for training
        """
        self.data_root_dir = data_root_dir
        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(image_crop_size),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ])
        self.image_size = (image_size, image_size)
        self.cat = cat
        self.topk = topk
        self.labels = []
        self.tiles_for_slides = []
        for slide in tqdm(slides):
            filename_tiles = glob.glob(
                f"{self.data_root_dir}/{slide['id']}_*.jpeg")
            filename_tiles = sorted(
                filename_tiles,
                key=lambda x: int(x.split('_')[-1].split('.')[0]))[:self.topk]
            if(isinstance(filename_tiles[0], list)):
                break
            filenames = copy.deepcopy(filename_tiles)
            while len(filenames) < self.topk:
                filenames.extend(filename_tiles[:(self.topk-len(filenames))])
            filenames = filenames[:self.topk]
            self.tiles_for_slides.append(filenames)
            self.labels.append(slide['label'])

    def __getitem__(self, idx):
        images = []
        filenames = self.tiles_for_slides[idx]
        for filepath in filenames:
            image = Image.open(filepath)
            image = self.transform(image)
            images.append(image)
        if self.cat:
            img = torch.cat(images, dim=0)
        else:
            img = torch.stack(images, dim=0)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)


class MILBagMultiScaleDataset(MILBagDataset):
    def __init__(self, data_root_dir, slides, topk, pages,
                 image_crop_size, image_size=256, cat=False):
        super(MILBagMultiScaleDataset, self).__init__(
            data_root_dir, slides, topk,
            image_crop_size, None, image_size, cat)
        image_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ]
        tfms = image_transforms
        crop_sizes = [image_crop_size * 2**(len(pages)-1-i)
                      for i in range(len(pages))]
        self.transforms = [transforms.Compose([transforms.CenterCrop(
            crop_size), transforms.Resize(crop_size // 2)] + tfms) for
            crop_size in crop_sizes]

        self.tiles_for_slides = []
        for slide in tqdm(slides):
            filename_tiles = slide['tiles']
            if(len(filename_tiles) == 0):
                break
            filename_group = copy.deepcopy(filename_tiles)
            while len(filename_group[0]) < self.topk:
                for group, filenames in zip(filename_group, filename_tiles):
                    group.extend(filenames[:(self.topk-len(filenames))])
            for i in range(len(filename_group)):
                filename_group[i] = filename_group[i][:self.topk]
            self.tiles_for_slides.append(filename_group)

    def __getitem__(self, idx):
        pyramid = []
        slide = self.slides[idx]
        for i, filenames in enumerate(self.tiles_for_slides[idx]):
            images = []
            for filepath in filenames:
                image = Image.open(filepath)
                image = self.transforms[i](image)
                images.append(image)
            if self.cat:
                img = torch.cat(images, dim=0)
            else:
                img = torch.stack(images, dim=0)
            pyramid.append(img)
        return *pyramid, slide["label"]


def get_bag_dataloaders(data_root_dir, split_path, topk, crop_size,
                        transform=None, transform_test=None,
                        batch_size=1, num_workers=1, cat=False):
    train_slides, val_slides, test_slides = get_train_data_tiles(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides, topk,
                             crop_size, transform=transform, cat=cat)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True)
    valset = MILBagDataset(data_root_dir, val_slides, topk, crop_size,
                           transform=transform_test, cat=cat)
    valloader = data.DataLoader(valset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
    testset = MILBagDataset(data_root_dir, test_slides, topk, crop_size,
                            transform=transform_test, cat=cat)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return trainloader, valloader, testloader


def get_bag_datasets(data_root_dir, split_path, topk, crop_size,
                     transform=None, transform_test=None, cat=False):
    train_slides, val_slides, test_slides = get_train_data_tiles(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides, topk,
                             crop_size, transform=transform, cat=cat)
    valset = MILBagDataset(data_root_dir, val_slides, topk,
                           crop_size, transform=transform_test, cat=cat)
    testset = MILBagDataset(data_root_dir, test_slides, topk,
                            crop_size, transform=transform_test, cat=cat)
    return trainset, valset, testset


def get_bag_multi_scale_dataloaders(data_root_dir, split_path, topk, pages,
                                    crop_size, batch_size=1, num_workers=1,
                                    cat=False):
    train_slides, val_slides, test_slides = get_train_data_tiles(split_path)
    trainset = MILBagMultiScaleDataset(data_root_dir, train_slides, topk,
                                       pages, crop_size, cat=cat)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True)
    valset = MILBagMultiScaleDataset(data_root_dir, val_slides, topk,
                                     pages, crop_size, cat=cat)
    valloader = data.DataLoader(valset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
    testset = MILBagMultiScaleDataset(data_root_dir, test_slides, topk,
                                      pages, crop_size, cat=cat)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return trainloader, valloader, testloader


class LabeledDataset(data.Dataset):
    r"""Dataset class with region-level annotations"""

    def __init__(self, image_info, pre_transforms,
                 transform=None, crop_size=96, image_size=96):
        r"""Constructor for annotated region dataset

        Args:
            root_dir: data root directory
            transform: transform to apply on images
            crop_size: size of center crop
            image_size: the size to resize the images to
        """
        super(LabeledDataset, self).__init__()
        self.pre_transforms = pre_transforms
        self.filepaths = [info["filepath"] for info in image_info]
        self.tfm_ids = [info["tfm_id"] for info in image_info]
        # self.labels = [info["label"] for info in image_info]
        self.labels = [int(tfm_id == 0) for tfm_id in self.tfm_ids]

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(crop_size),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        image = Image.open(filepath)
        image = self.pre_transforms[self.tfm_ids[idx]](image)
        image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.filepaths)


def get_labeled_data_loader(image_info, pre_transforms,
                            transform=None, crop_size=96, image_size=96,
                            batch_size=8, num_workers=4, shuffle=False):
    dataset = LabeledDataset(image_info, pre_transforms, transform,
                             crop_size, image_size)
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=shuffle)
    return loader
