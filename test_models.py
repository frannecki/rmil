import random
import torch
import torch.utils.data as data
import PIL.Image as Image

from rmil.rmil import main, get_transforms, get_options
from rmil.dataloader import LabeledDataset, get_labeled_data_loaders


def generate_random_metadata(filepath, number, n_classes):
    return [
        {
            "filepath": filepath,
            "label": random.randint(0, n_classes-1)
        } for _ in range(number)
    ]


METADATA_MIL = {
    "train": generate_random_metadata(["test/image.jpeg"] * 5, 80, 4),
    "val": generate_random_metadata(["test/image.jpeg"] * 5, 10, 4),
    "test": generate_random_metadata(["test/image.jpeg"] * 5, 10, 4),
}

METADATA_AUX = {
    "train": generate_random_metadata("test/image.jpeg", 80, 4),
    "val": generate_random_metadata("test/image.jpeg", 20, 4),
}

METADATA_SSL = {
    "train": generate_random_metadata("test/image.jpeg", 80, 5),
    "val": generate_random_metadata("test/image.jpeg", 20, 5),
}


class MILBagDatasetMock(LabeledDataset):
    r"""Mocked bag dataset for multiple-instance learning
    """
    def __init__(self, transform, meta_data):
        super(MILBagDatasetMock, self).__init__(transform, meta_data)

    def __getitem__(self, idx):
        images = []
        filenames = self.filepaths[idx]
        for filename in filenames:
            image = Image.open(filename)
            image = self.transform(image)
            images.append(image)
        img = torch.stack(images, dim=0)
        return img, self.labels[idx]


def get_bag_dataloaders(meta_data, transform, transform_test,
                        batch_size=1, num_workers=1):
    train_set = MILBagDatasetMock(transform, meta_data["train"])
    trainloader = data.DataLoader(train_set, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=True)
    val_set = MILBagDatasetMock(transform_test, meta_data["val"])
    valloader = data.DataLoader(val_set, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
    test_set = MILBagDatasetMock(transform_test, meta_data["test"])
    testloader = data.DataLoader(test_set, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False)
    return trainloader, valloader, testloader


if __name__ == '__main__':
    args = get_options()
    print(args)
    transform_train, transform_test = get_transforms(args)
    dataloader_mil = get_bag_dataloaders(METADATA_MIL, transform_train,
                                         transform_test)

    dataloaders_aux = get_labeled_data_loaders(METADATA_AUX, transform_train,
                                               transform_test, 4, 4)
    dataloaders_ssl = get_labeled_data_loaders(METADATA_SSL, transform_train,
                                               transform_test, 4, 4)
    main(args, dataloader_mil, dataloaders_aux, dataloaders_ssl)
