import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from dataloader import get_labeled_data_loader
import config
from model import SSClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, trainloader, valloader, criterion, optimizer, writer):
    model.train()
    running_loss = .0
    correct, total = 0, 0
    for idx, (input, tgts) in enumerate(trainloader):
        input = input.to(device)
        tgts = tgts.to(device)
        outs = model(input)
        preds = torch.argmax(outs, axis=1)
        total += input.shape[0]
        correct += torch.sum(preds == tgts).item()
        loss = criterion(outs, tgts)
        running_loss += loss.item() * input.shape[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = correct / total
    avg_loss = running_loss / total
    writer.add_scalars("loss", {"train": avg_loss})
    writer.add_scalars("acc", {"train": acc})

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for idx, (input, tgts) in enumerate(valloader):
            input = input.to(device)
            tgts = tgts.to(device)
            outs = model(input)
            preds = torch.argmax(outs, axis=1)
            total += input.shape[0]
            correct += torch.sum(preds == tgts).item()
    acc = correct / total
    writer.add_scalars("acc", {"eval": acc})
    return acc


if __name__ == '__main__':
    # data
    image_sizes = [96, 144, 192, 288, 384]
    pre_transforms = [transforms.Compose([transforms.Resize(image_size)])
                      for image_size in image_sizes]
    transform_train = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )
    transform_val = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )
    with open("train_test_split.json", "r") as f:
        images = json.load(f)
    train_images, val_images = images["train"], images["val"]
    trainloader = get_labeled_data_loader(
        train_images, pre_transforms, transform_train, batch_size=128)
    valloader = get_labeled_data_loader(
        val_images, pre_transforms, transform_val, batch_size=128)

    # model
    model = SSClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter("./logs")

    best_acc = .0
    for i in range(config.EPOCHS):
        acc = train_model(model, trainloader, valloader,
                          criterion, optimizer, writer)
        print("Accuracy for epoch {}: {:.4f}".format(i, acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "checkpoint.pth")
