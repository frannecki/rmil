import os
import json
import glob
import copy
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image
from typing import List, Dict, Tuple


def get_train_data(split_path: str) -> Tuple[List[Dict]]:
    r"""Load train/val/test split result

    Args:
        split_path：path to split json file
    """
    with open(split_path, 'r') as f:
        data = json.load(f)
    slides = [data[task] for task in ["train", "val", "test"]]
    return slides


class MILBagDataset(data.Dataset):
    r"""Bag dataset for multiple-instance learning
    """
    def __init__(self, data_root_dir, slides, topk,
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
                raise RuntimeError(f"Invalid patch files for {slide['id']}")
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


class AnnotatedRegionDataset(data.Dataset):

    class_names = ['benign', 'low-level', 'high-level', 'cancer']

    def __init__(self, root_dir, transform=None,
                 crop_size=256, image_size=96):
        super(AnnotatedRegionDataset, self).__init__()
        with open("../slide_with_related_patches.json", 'r') as f:
            registered_slides = json.load(f)
        registered_slides = set([slide['id'] for slide
                                in registered_slides])
        self.root_dir, self.filenames, self.labels = root_dir, [], []
        for i, class_name in enumerate(self.class_names):
            filenames = os.listdir(os.path.join(root_dir, class_name))
            filenames = [filename for filename in filenames if
                         '_'.join(filename.split('_')[:3]) in
                         registered_slides or '_'.join(filename.split('_')[:4])
                         in registered_slides]
            filenames = [os.path.join(class_name, filename) for filename
                         in filenames if filename.endswith('.jpeg')]
            self.filenames += filenames
            self.labels += [i] * len(filenames)

        self.transform = transform
        self.crop_size = crop_size
        self.image_size = image_size
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.CenterCrop(self.crop_size),
                transforms.Resize((image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(os.path.join(self.root_dir, filename))
        image = self.transform(image)
        return image, self.labels[idx]

    def __len__(self):
        return len(self.filenames)


def get_bag_dataloaders(data_root_dir, split_path, topk,
                        transform=None, transform_test=None,
                        batch_size=1, num_workers=1, cat=False):
    train_slides, val_slides, test_slides = get_train_data(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides,
                             topk, cat=cat, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True)
    valset = MILBagDataset(data_root_dir, val_slides,
                           topk, cat=cat, transform=transform_test)
    valloader = data.DataLoader(valset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
    testset = MILBagDataset(data_root_dir, test_slides,
                            topk, cat=cat, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return trainloader, valloader, testloader


def get_bag_datasets(data_root_dir, split_path, topk, transform=None,
                     transform_test=None, cat=False):
    train_slides, val_slides, test_slides = get_train_data(split_path)
    trainset = MILBagDataset(data_root_dir, train_slides,
                             topk, cat=cat, transform=transform)
    valset = MILBagDataset(data_root_dir, val_slides,
                           topk, cat=cat, transform=transform_test)
    testset = MILBagDataset(data_root_dir, test_slides,
                            topk, cat=cat, transform=transform_test)
    return trainset, valset, testset


def get_annotated_region_loader(root_dir, crop_size, image_size,
                                transform=None, batch_size=1,
                                num_workers=4, shuffle=False):
    dataset = AnnotatedRegionDataset(root_dir, transform,
                                     crop_size, image_size)
    loader = data.DataLoader(dataset, batch_size=batch_size,
                             num_workers=num_workers, shuffle=shuffle)
    return loader
