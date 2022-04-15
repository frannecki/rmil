import os
import os.path as osp
import json
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from . import config
from .model import build_attn_mil
from .model import build_feature_backbone
from .model import build_naive_model

from .dataloader import get_bag_dataloaders, get_labeled_data_loaders
from .utils import train_model, evaluate_model
from .criterion import GradeLoss, GradedCrossEntropyLoss, CrossEntropyLoss

from .ssl_ import build_ssl_model
from .ssl_ import get_ssl_data_loaders
from .ssl_ import config as config_ssl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_options():
    parser = argparse.ArgumentParser(description='MIL')
    # data
    parser.add_argument('--data_root_dir', type=str,
                        default=config.DATA_ROOT_DIR,
                        help='root directory of tiles cropped from wsi')
    parser.add_argument('--train_labels_csv', type=str,
                        default=config.TRAIN_LABELS_CSV,
                        help='filepath of train labels (csv)')
    parser.add_argument('--split_path', type=str,
                        default=config.TRAIN_TEST_SPLIT,
                        help='train/val/test split result path (json)')
    parser.add_argument('--split_path_ssl', type=str,
                        default=config_ssl.META_DATA_PATH,
                        help='train/val split json path for '
                             'self supervised learning')
    parser.add_argument('--split_path_aux', type=str,
                        default=config.TRAIN_TEST_SPLIT_AUX,
                        help='train/val split json path for '
                             'supervised learning')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='mini batch size')
    parser.add_argument('--batch_size_ssl', type=int,
                        default=config_ssl.BATCH_SIZE,
                        help='mini batch size for self supervised learning')
    parser.add_argument('--batch_size_aux', type=int,
                        default=config.BATCH_SIZE_AUX,
                        help='mini batch size for supervised learning')
    parser.add_argument('--image_crop_size', type=int,
                        default=config.IMAGE_CROP_SIZE,
                        help='size to center crop images for self '
                             'supervised learning')
    parser.add_argument('--image_size', type=int, default=config.IMAGE_SIZE,
                        help='size to resize image to')
    # model
    parser.add_argument('--backbone', type=str, default=config.BACKBONE,
                        help='backbone (resnet18/resnet34/resnet50)')
    parser.add_argument('--avgpool_size', type=int,
                        default=config.AVGPOOL_SIZE,
                        help='average pooling size for resnet classifier')
    parser.add_argument('--avgpool_size_attn', type=int,
                        default=config.AVGPOOL_SIZE_ATTN,
                        help='average pooling size for attention block')
    parser.add_argument('--num_classes', type=int, default=config.NUM_CLASSES,
                        help='number of classes')
    parser.add_argument('--out_features', type=int,
                        default=config.OUT_FEATURES,
                        help='output feature dimension of network')
    parser.add_argument('--criterion', type=str, default="graded",
                        help='loss function for multiple instance learning')
    parser.add_argument('--lambda_aux', type=float, default=config.LAMBDA_AUX,
                        help='loss function weight for supervised learning')
    parser.add_argument('--lambda_ssl', type=float, default=config.LAMBDA_SSL,
                        help='loss function weight for self supervised learning')
    parser.add_argument('--attn_features', type=int,
                        default=config.ATTN_FEATURES)
    parser.add_argument('--attn', action='store_true')
    # pipeline
    parser.add_argument('--trained_epochs', type=int, default=0,
                        help='number of trained epochs')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='number of epochs')
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
    parser.add_argument('--model_dir', type=str, default='./models/models')
    parser.add_argument('--log_dir', type=str, default='./logs/logs')
    parser.add_argument('--no-pretrained', dest='pretrained',
                        action='store_false',
                        help="Do not use pretrained models")
    parser.add_argument('--aux', action='store_true',
                        help="Train with supervised learning task with"
                        " auxiliary dataset (annotated patches)")
    parser.add_argument('--reg', action='store_true',
                        help="Use auxiliary dataset with registration")
    parser.add_argument('--ssl', action='store_true',
                        help="Train with self supervised learning task")
    parser.add_argument('--testonly', action='store_true')
    return parser.parse_args()


def get_transforms(args):
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
    return transform_train, transform_test


def get_mil_data(args, transform_train, transform_test):
    trainloader, valloader, testloader = get_bag_dataloaders(
        args.data_root_dir, args.split_path, args.topk,
        transform_train, transform_test, args.batch_size, args.workers)
    return trainloader, valloader, testloader


def get_ssl_data(args, transform_train, transform_test):
    trainloader, valloader = get_ssl_data_loaders(
        args.split_path_ssl, transform_train, transform_test,
        batch_size=args.batch_size_ssl)
    return trainloader, valloader


def get_aux_data(args, transform_train, transform_test):
    with open(args.split_path_aux, "r") as f:
        meta_data = json.load(f)
    trainloader, valloader = get_labeled_data_loaders(
        meta_data, transform_train, transform_test, args.batch_size)
    return trainloader, valloader


def main(args, dataloaders_mil, dataloaders_aux=None, dataloaders_ssl=None):
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    #########################################
    # model
    backbone, backbone_out_features = build_feature_backbone(args)
    model = build_attn_mil(args, backbone, backbone_out_features)
    ckpt_path = os.path.join(args.model_dir,
                             f"{args.checkpoint_path}_{args.metr}.pth")
    model = model.to(device)
    if osp.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])
        best_acc, best_score = ckpt["acc"], ckpt["score"]
        print("Loaded pretrained model:", ckpt_path)
        print("Accuracy: {:.4f}  Score: {:.4f}".format(best_acc, best_score))
    else:
        print("No pretrained model.")
        best_acc, best_score = 0., 0.

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

        trainloader, valloader, testloader = dataloaders_mil
        acc, score = evaluate_model(args, model, testloader)
        print("Initial performance on test set: Accuracy {:.4f}, Score {:.4f}"
              .format(acc, score))

        print("Number of training, validating and testing batches:",
              len(trainloader), len(valloader), len(testloader))

        subtasks = []

        if dataloaders_aux:
            # supervised learning subtask based on annotated regions
            model_naive = build_naive_model(args, backbone,
                                            backbone_out_features,
                                            args.num_classes)
            model_naive = model_naive.to(device)
            trainloader_aux, valloader_aux = dataloaders_aux
            criterion_aux = CrossEntropyLoss(weight=args.lambda_aux)
            print("Number of training and testing batches for supervised "
                  "learning", len(trainloader_aux), len(valloader_aux))
            subtasks.append({
                "task": "aux",
                "model": model_naive,
                "trainloader": trainloader_aux,
                "valloader": valloader_aux,
                "criterion": criterion_aux
            })
        if dataloaders_ssl:
            # self-supervised learning subtask
            model_ssl = build_ssl_model(args, backbone, backbone_out_features)
            model_ssl = model_ssl.to(device)
            trainloader_ssl, valloader_ssl = dataloaders_ssl
            criterion_ssl = CrossEntropyLoss(weight=args.lambda_ssl)
            print("Number of training and testing batches for self-supervised "
                  "learning", len(trainloader_ssl), len(valloader_ssl))
            subtasks.append({
                "task": "ssl",
                "model": model_ssl,
                "trainloader": trainloader_ssl,
                "valloader": valloader_ssl,
                "criterion": criterion_ssl
            })

        # train
        for epoch in range(args.trained_epochs, args.epochs):
            acc, score = train_model(args, model, criterion, trainloader,
                                     valloader, subtasks, optimizer, epoch)
            if acc > best_acc:
                print("Better acc: {:.4f}. Saving checkpoint...".format(acc))
                best_acc = acc
                torch.save({
                    "model": model.state_dict(), "acc": best_acc,
                    "score": score, "args": args},
                           osp.join(args.model_dir,
                                    f"{args.checkpoint_path}_acc.pth"))
            if score > best_score:
                print("Better score: {:.4f}. Saving checkpoint..."
                      .format(score))
                best_score = score
                torch.save({
                    "model": model.state_dict(), "acc": acc,
                    "score": best_score, "args": args},
                           os.path.join(args.model_dir,
                                        f"{args.checkpoint_path}_score.pth"))

    # evaluation
    ckpt_path = os.path.join(args.model_dir,
                             f"{args.checkpoint_path}_{args.metr}.pth")
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])
    acc, score = evaluate_model(args, model, testloader)
    print("Performance on test set: Accuracy {:.4f}, Score {:.4f}"
          .format(acc, score))
