import os
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import config
from model import build_attn_mil
from dataloader import get_bag_dataloaders
from utils import train_model, evaluate_model
from utils import train_model_on_labeled_data
from criterion import GradeLoss, GradedCrossEntropyLoss

import json
import ssl.config as config_ss
import ssl.dataloader as dataloader_ss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='MIL')
# data
parser.add_argument('--data_root_dir', type=str, default=config.DATA_ROOT_DIR,
                    help='root directory of tiles cropped from wsi')
parser.add_argument('--train_labels_csv', type=str,
                    default=config.TRAIN_LABELS_CSV,
                    help='filepath of train labels (csv)')
parser.add_argument('--split_path', type=str, default=config.TRAIN_TEST_SPLIT,
                    help='train/val/test split result path (json)')
parser.add_argument('--split_path_ss', type=str,
                    default=config.TRAIN_TEST_SPLIT_SS,
                    help='train/val/test split result path (json) for '
                         'self supervised learning')
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                    help='mini batch size')
parser.add_argument('--batch_size_ss', type=int,
                    default=config.BATCH_SIZE_SS,
                    help='mini batch size for self supervised learning')
parser.add_argument('--meta_data_path_ss', type=str,
                    default=config_ss.META_DATA_PATH,
                    help='path to the metadata json file '
                         'for self-supervised learning')
parser.add_argument('--image_crop_size', type=int,
                    default=config.IMAGE_CROP_SIZE,
                    help='size to center crop images for self '
                         'supervised learning')
parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                    help='size to resize image to')
# model
parser.add_argument('--backbone', type=str, default=config.BACKBONE,
                    help='backbone (resnet18/resnet34/resnet50)')
parser.add_argument('--avgpool_size', type=int, default=config.AVGPOOL_SIZE,
                    help='average pooling size for resnet classifier')
parser.add_argument('--avgpool_size_attn', type=int,
                    default=config.AVGPOOL_SIZE_ATTN,
                    help='average pooling size for attention block')
parser.add_argument('--num_classes', type=int, default=config.NUM_CLASSES,
                    help='number of classes')
parser.add_argument('--out_features', type=int, default=config.OUT_FEATURES,
                    help='output feature dimension of network')
parser.add_argument('--out_features_ss', type=int,
                    default=len(config_ss.IMAGE_SIZES),
                    help='output feature dimension for self-supervised '
                         'learning')
parser.add_argument('--criterion', type=str, default='graded',
                    help='loss functions for weight update. '
                         'Must be among ["graded", "cross_entropy", '
                         '"combined"]')
parser.add_argument('--attn', action='store_true')
# pipeline
parser.add_argument('--trained_epochs', type=int, default=0,
                    help='number of trained epochs')
parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                    help='number of epochs')
parser.add_argument('--trained_epochs_ss', type=int, default=0,
                    help='number of trained epochs for '
                         'self supervised learning')
parser.add_argument('--epochs_ss', type=int, default=config.EPOCHS_SS,
                    help='number of epochs for self-supervised learning')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--topk', type=int, default=config.TOPK,
                    help='how many top k tiles to consider (default: 10)')
parser.add_argument('--checkpoint_path', type=str, default='checkpoint',
                    help='checkpoint name prefix')
parser.add_argument("--metr", type=str, default=config.METR,
                    help='metric based on which to load pretrained model. '
                         'Empty for accuracy. `score` for score')
parser.add_argument('--save_step', type=int, default=10,
                    help='number of steps to save model')
parser.add_argument('--log_step', type=int, default=50,
                    help='number of batches to show log')
parser.add_argument('--log_step_ss', type=int, default=20,
                    help='number of batches to show log for '
                         'self supervised learning')
parser.add_argument('--model_dir', type=str, default='./models/models')
parser.add_argument('--log_dir', type=str, default='./logs/logs')
parser.add_argument('--log_dir_ss', type=str, default='./logs/logs_ss')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrain', action='store_true',
                    help="Pretrain with self supervised learning task")
parser.add_argument('--testonly', action='store_true')


def init_self_supervision_data(args):
    pre_transforms = [transforms.Compose([transforms.Resize(image_size)])
                      for image_size in config_ss.IMAGE_SIZES]
    transform_train = transforms.Compose([
        transforms.CenterCrop(args.image_crop_size),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )
    transform_val = transforms.Compose([
        transforms.CenterCrop(args.image_crop_size),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )
    with open(args.meta_data_path_ss, "r") as f:
        images = json.load(f)
    train_images, val_images = images["train"], images["val"]
    trainloader = dataloader_ss.get_labeled_data_loader(
        train_images, pre_transforms, transform_train,
        batch_size=args.batch_size_ss)
    valloader = dataloader_ss.get_labeled_data_loader(
        val_images, pre_transforms, transform_val,
        batch_size=args.batch_size_ss)
    return trainloader, valloader


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    print(args)
    model, model_ss = build_attn_mil(args)
    ckpt_path = os.path.join(args.model_dir,
                             f"{args.checkpoint_path}_{args.metr}.pth")
    model = model.to(device)
    try:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        best_acc, best_score = ckpt["acc"], ckpt["score"]
        print("Loaded pretrained model:", ckpt_path)
        print("Accuracy: {:.4f}  Score: {:.4f}".format(best_acc, best_score))
    except Exception:
        print("No pretrained model.")
        best_acc, best_score = 0., 0.

    transform_train = transforms.Compose([
        transforms.CenterCrop(args.image_crop_size),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(args.image_crop_size),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not args.testonly:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4, weight_decay=1e-4)
        assert args.criterion in ["graded", "cross_entropy", "combined"]
        if args.criterion == "graded":
            criterion = GradeLoss()
        elif args.criterion == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            criterion = GradedCrossEntropyLoss()

        if args.pretrain:
            pre_transforms = [transforms.Compose(
                [transforms.Resize(image_size)])
                for image_size in config.IMAGE_SIZES_SS]
            train_model_on_labeled_data(args, model, criterion, optimizer,
                                        pre_transforms, transform_train,
                                        transform_test)
            ckpt_path = os.path.join(args.model_dir,
                                     f"{args.checkpoint_path}.pth")
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            print("Accurary for self supervised learning:", ckpt.get("acc"))
            model.fine_tune()

        trainloader, valloader, testloader = get_bag_dataloaders(
            args.data_root_dir, args.split_path, args.topk,
            args.image_crop_size, transform_train, transform_test,
            args.batch_size, args.workers)

        # self-supervised learning
        model_ss = model_ss.to(device)
        trainloader_ss, valloader_ss = init_self_supervision_data(args)
        criterion_ss = torch.nn.CrossEntropyLoss()

        acc, score = evaluate_model(args, model, testloader)
        print("Performance on test set: Accuracy {:.4f}, Score {:.4f}".format(
            acc, score))

        print("number of training and testing batches",
              len(trainloader), len(valloader))
        print("number of training and testing batches for "
              "self-supervised learning",
              len(trainloader_ss), len(valloader_ss))

        train_model(args, model, model_ss, criterion, criterion_ss,
                    trainloader, valloader, trainloader_ss,
                    valloader_ss, optimizer, best_acc, best_score)

    ckpt_path = os.path.join(args.model_dir,
                             f"{args.checkpoint_path}_{args.metr}.pth")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    acc, score = evaluate_model(args, model, testloader)
    print("Performance on test set: Accuracy {:.4f}, Score {:.4f}"
          .format(acc, score))
