import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as transforms


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
        self.labels = [info["label"] for info in image_info]

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
