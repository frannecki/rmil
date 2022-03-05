import os
import json
import torch
from tensorboardX import SummaryWriter
from dataloader import get_labeled_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-5  # Avoid division by zero
ERROR_TABLE = [[.0, .1, .7, 1.], [.1, .0, .3, .7],
               [.7, .3, .0, .3], [1., .7, .3, .0]]
ERROR_TABLE = torch.Tensor(ERROR_TABLE).to(device)


def log_function(log_dir='log', purge_step=0):
    """function for log."""
    return SummaryWriter(log_dir, purge_step=purge_step)


def predict_staged(logits):
    probs = torch.sigmoid(logits).sum(1)
    preds = probs.round()
    return preds


def calc_score(pred_table):
    score = 1 - torch.sum(ERROR_TABLE * pred_table.float()) \
        .item() / torch.sum(pred_table.float()).item()
    return score


def train_model(args, model, model_ss, criterion, criterion_ss,
                train_loader, val_loader, train_loader_ss,
                val_loader_ss, optimizer, best_acc, best_score):
    r"""Train model on wsi image tiles"""
    classes = torch.arange(args.num_classes).unsqueeze(1).to(device)
    writer = log_function(f'{args.log_dir}', purge_step=0)

    for epoch in range(args.trained_epochs,
                       args.trained_epochs + args.epochs):
        # Training Stage
        model.train()
        running_loss, correct = 0., 0
        pred_table_train = torch.zeros(
            args.num_classes, args.num_classes).long().to(device)
        pred_table_val = torch.zeros(
            args.num_classes, args.num_classes).long().to(device)

        train_iter_ss = iter(train_loader_ss)
        # val_iter_ss = iter(val_loader_ss)
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()

            imgs_ss, targets_ss = next(train_iter_ss)

            imgs_ss = imgs_ss.unsqueeze(0).to(device)
            targets_ss = targets_ss.to(device)
            logits_ss = model_ss(imgs_ss)
            loss_ss = criterion_ss(logits_ss, targets_ss)

            loss_total = loss + loss_ss
            loss_total.backward()

            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = predict_staged(logits)
            correct += torch.sum(targets == preds).item()
            pred_table_train += torch.sum(
                (preds == classes).unsqueeze(0) &
                (targets == classes).unsqueeze(1), dim=2)
            if (i+1) % args.log_step == 0:
                print('train\tEpoch: ' +
                      f'[{epoch+1}/{args.epochs + args.trained_epochs}]\t' +
                      f'Batch:[{i+1}/{len(train_loader)}]\t{loss}')
        running_loss /= len(train_loader.dataset)
        acc = correct / len(train_loader.dataset)
        score_train = calc_score(pred_table_train)
        print(('Training\tEpoch: [{}/{}]\tLoss: ' +
               '{:.4f}\tAcc: {:.4f}\tScore: {:.4f}').format(
                  epoch+1, args.epochs+args.trained_epochs,
                  running_loss, acc, score_train))
        writer.add_scalars('score', {'train': score_train}, epoch)
        writer.add_scalars('loss', {'train': loss}, epoch)
        writer.add_scalars('acc', {'train': acc}, epoch)

        # Validation Stage
        model.eval()
        running_loss_val, correct = 0., 0
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.to(device)
                tgts = targets.to(device)
                logits = model(imgs)
                loss = criterion(logits, tgts)
                preds = predict_staged(logits)
                correct += torch.sum(preds == tgts).item()
                pred_table_val += torch.sum(
                    (preds == classes).unsqueeze(0) &
                    (tgts == classes).unsqueeze(1), dim=2)
                running_loss_val += loss.item() * imgs.size(0)
                if (i+1) % args.log_step == 0:
                    print(f'validation\tEpoch: [{epoch+1}/{args.epochs}]\t' +
                          f'Batch:[{i+1}/{len(val_loader)}]\t{loss}')
        running_loss_val /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        score_val = calc_score(pred_table_val)
        writer.add_scalars('score', {'val': score_val}, epoch)
        writer.add_scalars('loss', {'val': running_loss_val}, epoch)
        writer.add_scalars('acc', {'val': acc}, epoch)
        print(('Validating\tEpoch: [{}/{}]\tLoss: ' +
              '{:.4f}\tAcc: {:.4f}\tScore: {:.4f}')
              .format(epoch+1, args.epochs+args.trained_epochs,
                      running_loss_val, acc, score_val))

        for t in range(args.num_classes):
            num_train = pred_table_train[t][t].item()
            num_val = pred_table_val[t][t].item()
            precision_train = num_train / (
                torch.sum(pred_table_train[:, t]).item() + EPS)
            precision_val = num_val / (torch.sum(
                pred_table_val[:, t]).item() + EPS)
            recall_train = num_train / (
                torch.sum(pred_table_train[t]).item() + EPS)
            recall_val = num_val / (torch.sum(pred_table_val[t]).item() + EPS)
            writer.add_scalars('recall_class_{}'.format(t), {
                'train': recall_train, 'val': recall_val
            }, epoch + args.trained_epochs)

            writer.add_scalars('precision_class_{}'.format(t), {
                'train': precision_train, 'val': precision_val
            }, epoch + args.trained_epochs)

        if (1 + epoch) % args.save_step == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.model_dir,
                             "model-epoch-{}.pth".format(epoch+1)))
        if acc > best_acc:
            print("Better acc found: {:.4f}. Saving checkpoint...".format(acc))
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": best_acc,
                        "score": best_score, "args": args}, os.path.join(
                            args.model_dir,
                            f"{args.checkpoint_path}_acc.pth"))
        if score_val > best_score:
            print(("Better score found: {:.4f}. " +
                  "Saving checkpoint...").format(score_val))
            best_score = score_val
            torch.save({"model": model.state_dict(), "acc": best_acc,
                        "score": best_score, "args": args},
                       os.path.join(args.model_dir,
                                    f"{args.checkpoint_path}_score.pth"))
    return best_acc, best_score


def evaluate_model(args, model, loader):
    r"""Evaluate model on wsi image tiles"""
    model.eval()
    classes = torch.arange(args.num_classes).unsqueeze(1).to(device)
    pred_table = torch.zeros(args.num_classes,
                             args.num_classes).long().to(device)
    correct = 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(loader):
            imgs = imgs.to(device)
            tgts = targets.to(device)
            logits = model(imgs)
            preds = predict_staged(logits)
            correct += torch.sum(preds == tgts).item()
            pred_table += torch.sum(
                (preds == classes).unsqueeze(0) &
                (tgts == classes).unsqueeze(1), dim=2)
    for t in range(args.num_classes):
        num = pred_table[t][t].item()
        precision = num / (torch.sum(pred_table[:, t]).item() + EPS)
        recall = num / (torch.sum(pred_table[t]).item() + EPS)
        print("Metrics for class {:d}: Recall {:.4f} Precision {:.4f}"
              .format(t, recall, precision))
    acc = correct / len(loader.dataset)
    score = calc_score(pred_table)
    return acc, score


def train_model_on_labeled_data(args, model, criterion, optimizer,
                                pre_transforms, transform_train,
                                transform_test):
    r"""Train model for self supervised learning"""
    with open(args.split_path, 'r') as f:
        result_split = json.load(f)

    train_loader = get_labeled_data_loader(
        result_split["train"], pre_transforms, transform_train,
        args.image_crop_size, args.image_size,
        args.batch_size_ss, args.workers, shuffle=True)
    val_loader = get_labeled_data_loader(
        result_split["val"], transform_test,
        args.image_crop_size, args.image_size,
        args.batch_size_ss, args.workers)
    writer = log_function(f'{args.log_dir_ss}', purge_step=0)
    best_acc = 0.
    total_epochs = args.trained_epochs_ss + args.epochs_ss
    for epoch in range(args.trained_epochs_ss, total_epochs):
        # Training Stage
        model.train()
        running_loss = 0.
        correct = 0
        for i, (imgs, targets) in enumerate(train_loader):
            imgs = imgs.unsqueeze(dim=1).to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = predict_staged(logits)
            correct += torch.sum(targets == preds).item()
            if (i+1) % args.log_step_ss == 0:
                print('train\tEpoch: ' +
                      f'[{epoch+1}/{total_epochs}]\t' +
                      f'Batch:[{i+1}/{len(train_loader)}]\t{loss}')
        running_loss /= len(train_loader.dataset)
        acc = correct / len(train_loader.dataset)
        print(('Training\tEpoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}').format(
                  epoch+1, args.epochs_ss + args.trained_epochs_ss,
                  running_loss, acc))
        writer.add_scalars('loss', {'train': loss}, epoch)
        writer.add_scalars('acc', {'train': acc}, epoch)

        # Validation Stage
        model.eval()
        running_loss_val = 0.
        correct = 0
        with torch.no_grad():
            for i, (imgs, targets) in enumerate(val_loader):
                imgs = imgs.unsqueeze(dim=1).to(device)
                tgts = targets.to(device)
                logits = model(imgs)
                loss = criterion(logits, tgts)
                preds = predict_staged(logits)
                correct += torch.sum(preds == tgts).item()
                running_loss_val += loss.item() * imgs.size(0)
                if (i+1) % args.log_step_ss == 0:
                    print(f'Validation\tEpoch: '
                          f'[{epoch+1}/{args.epochs_ss}]\t'
                          f'Batch:[{i+1}/{len(val_loader)}]\t{loss}')
        running_loss_val /= len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)

        if acc > best_acc:
            print("Better acc found: {:.4f}. Saving checkpoint...".format(acc))
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc}, os.path.join(
                args.model_dir, f"{args.checkpoint_path}.pth"))

        writer.add_scalars('loss', {'val': running_loss_val}, epoch)
        writer.add_scalars('acc', {'val': acc}, epoch)
        print('Validating\tEpoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
              .format(epoch+1, args.epochs_ss+args.trained_epochs_ss,
                      running_loss_val, acc))
