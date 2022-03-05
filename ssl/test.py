import json
import torch
import torchvision.transforms as transforms
from dataloader import get_labeled_data_loader
from model import SSClassifier


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_model(model, valloader):
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
    return acc


if __name__ == '__main__':
    # data
    image_sizes = [96, 144, 192, 288, 384]
    pre_transforms = [transforms.Compose([transforms.Resize(image_size)])
                      for image_size in image_sizes]
    transform_val = transforms.Compose([
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )
    with open("train_test_split.json", "r") as f:
        images = json.load(f)
    val_images = images["val"]
    valloader = get_labeled_data_loader(
        val_images, pre_transforms, transform_val, batch_size=128)

    # model
    model = SSClassifier().to(device)

    model.load_state_dict(torch.load("checkpoint.pth"))
    acc = eval_model(model, valloader)
    print("Accuracy: {:.4f}".format(acc))
