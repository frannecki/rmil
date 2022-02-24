import os
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import config
from model import build_attn_mil
from dataloader import get_bag_dataloaders, get_annotated_region_loader
from utils import train_model, evaluate_model
from criterion import GradeLoss, GradedCrossEntropyLoss, FocalLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_function(log_dir='log', purge_step=0):
    """function for log."""
    return SummaryWriter(log_dir, purge_step=purge_step)


parser = argparse.ArgumentParser(description='MIL')
# model
parser.add_argument('--data_root_dir', type=str, default=config.DATA_ROOT_DIR,
                    help='root directory of tiles cropped from wsi')
parser.add_argument('--annotated_regions_root_dir', type=str,
                    default=config.ANNOTATED_REGIONS_ROOT_DIR,
                    help='root directory of annotated regions')
parser.add_argument('--train_labels_csv', type=str,
                    default=config.TRAIN_LABELS_CSV,
                    help='filepath of train labels (csv)')
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
parser.add_argument('--criterion', type=str, default='graded',
                    help='loss functions for weight update. '
                         'Must be among ["graded", "cross_entropy", '
                         '"combined"]')
parser.add_argument('--attn', dest='attn', action='store_true')
# data
parser.add_argument('--split_path', type=str, default=config.TRAIN_TEST_SPLIT,
                    help='train/val/test split result path (json)')
parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                    help='mini batch size')
parser.add_argument('--batch_size_region', type=int,
                    default=config.BATCH_SIZE_REGION,
                    help='mini batch size for region training')
parser.add_argument('--image_crop_size', type=int,
                    default=config.IMAGE_CROP_SIZE,
                    help='size to center crop annotated regions')
parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                    help='size to resize image to')
parser.add_argument('--trained_epochs', type=int, default=0,
                    help='number of trained epochs')
parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                    help='number of epochs')
parser.add_argument('--trained_epochs_region', type=int, default=0,
                    help='number of trained epochs for annotated regions')
parser.add_argument('--epochs_region', type=int, default=config.EPOCHS_REGION,
                    help='number of epochs for training '
                         'with annotated regions')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--topk', type=int, default=config.TOPK,
                    help='how many top k tiles to consider (default: 10)')
# pipeline
parser.add_argument('--checkpoint_path', type=str, default='checkpoint',
                    help='checkpoint name prefix')
parser.add_argument("--metr", type=str, default=config.METR,
                    help='metric based on which to load pretrained model. '
                         'Empty for accuracy. `score` for score')
parser.add_argument('--save_step', type=int, default=10,
                    help='number of steps to save model')
parser.add_argument('--log_step', type=int, default=10,
                    help='number of batches to show log')
parser.add_argument('--log_step_region', type=int, default=100,
                    help='number of batches to show log for annotated regions')
parser.add_argument('--model_dir', type=str, default='./models')
parser.add_argument('--log_dir', type=str, default='./log')
parser.add_argument('--log_dir_region', type=str, default='./log_region')
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
parser.add_argument('--pretrain', dest='pretrain', action='store_true')
parser.add_argument('--testonly', dest='testonly', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    print(args)
    model = build_attn_mil(args).to(device)
    ckpt_path = os.path.join(args.model_dir,
                             f"{args.checkpoint_path}_{args.metr}.pth")
    try:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        best_acc, best_score = ckpt["acc"], ckpt["score"]
        print("Loaded pretrained model:", ckpt_path)
        print("Accuracy: {:.4f}  Score: {:.4f}".format(best_acc, best_score))
    except Exception:
        print("No pretrained model.")
        best_acc, best_score = 0., 0.

    transform = transforms.Compose([
        transforms.CenterCrop(args.image_crop_size),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
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

    trainloader, valloader, testloader = get_bag_dataloaders(
        args.data_root_dir, args.split_path, args.topk,
        transform, transform_test, args.batch_size, args.workers
    )

    if not args.testonly:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4, weight_decay=1e-4)
        if args.criterion == "graded":
            criterion = GradeLoss()
        elif args.criterion == "cross_entropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif args.criterion == "focal_loss":
            criterion = FocalLoss(4, device)
        elif args.criterion == "combined":
            criterion = GradedCrossEntropyLoss()
        else:
            raise NotImplementedError(
                f"The loss function {args.criterion} is not yet implemented")

        writer = log_function(f'{args.log_dir}', purge_step=0)

        if args.pretrain:
            trainloader_anno = get_annotated_region_loader(
                f"{args.annotated_regions_root_dir}/train",
                args.image_crop_size, args.image_size, transform,
                args.batch_size_region, args.workers, shuffle=True)

            valloader_anno = get_annotated_region_loader(
                f"{args.annotated_regions_root_dir}/val",
                args.image_crop_size, args.image_size,
                transform_test, args.batch_size_region, args.workers)
        else:
            trainloader_anno, valloader_anno = None, None

        for epoch in range(args.trained_epochs, args.epochs):
            acc, score = train_model(args, model, criterion, trainloader,
                                     valloader, trainloader_anno,
                                     valloader_anno, optimizer, epoch, writer)
            if acc > best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(),
                            "acc": best_acc,
                            "score": best_score,
                            "args": args},
                           os.path.join(args.model_dir,
                                        f"{args.checkpoint_path}_acc.pth"))

            if score > best_score:
                best_score = score
                torch.save({"model": model.state_dict(),
                            "acc": best_acc,
                            "score": best_score,
                            "args": args},
                           os.path.join(args.model_dir,
                                        f"{args.checkpoint_path}_score.pth"))

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])

    testloader_anno = get_annotated_region_loader(
        f"{args.annotated_regions_root_dir}/test",
        args.image_crop_size, args.image_size,
        transform_test, args.batch_size_region, args.workers)
    acc, score = evaluate_model(args, model, testloader)
    print("Performance on test set: Accuracy {:.4f}, Score {:.4f}"
          .format(acc, score))
