import os
import torch
from tensorboardX import SummaryWriter
from typing import List, Dict

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


def train_model(args, model, criterion, train_loader, val_loader,
                subtasks: List[Dict], optimizer, epoch):
    r"""Train model on wsi image tiles"""
    classes = torch.arange(args.num_classes).unsqueeze(1).to(device)
    writer = log_function(f'{args.log_dir}', purge_step=0)

    losses_sub = [0.] * len(subtasks)
    train_iters_sub = [iter(task['trainloader']) for task in subtasks]
    val_iters_sub = [iter(task['valloader']) for task in subtasks]

    # Training Stage
    model.train()
    for task in subtasks:
        task["model"].train()
    running_loss, correct = 0., 0
    pred_table_train = torch.zeros(
        args.num_classes, args.num_classes).long().to(device)
    pred_table_val = torch.zeros(
        args.num_classes, args.num_classes).long().to(device)

    for i, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        loss_total = loss

        for k, (train_iter_sub, task) in enumerate(zip(train_iters_sub,
                                                       subtasks)):
            try:
                imgs, tgts = next(train_iter_sub)
            except StopIteration:
                train_iters_sub[k] = iter(subtasks[k]['trainloader'])
                imgs, tgts = next(train_iters_sub[k])

            imgs = imgs.to(device)
            tgts = tgts.to(device)
            out = task["model"](imgs)
            cur_loss = task["criterion"](out, tgts)
            loss_total += cur_loss
            losses_sub[k] = cur_loss.item()

        optimizer.zero_grad()
        loss_total.backward()

        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = predict_staged(logits)
        correct += torch.sum(targets == preds).item()
        pred_table_train += torch.sum(
            (preds == classes).unsqueeze(0) &
            (targets == classes).unsqueeze(1), dim=2)
        if (i + 1) % args.log_step == 0:
            print('train\tEpoch: ' +
                  f'[{epoch+1}/{args.epochs + args.trained_epochs}]\t' +
                  f'Batch:[{i+1}/{len(train_loader)}]\t{loss}\t{losses_sub}')
    running_loss /= len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)
    score = calc_score(pred_table_train)
    print(('Training\tEpoch: [{}/{}]\tLoss: ' +
           '{:.4f}\tAcc: {:.4f}\tScore: {:.4f}').format(
                epoch+1, args.epochs+args.trained_epochs,
                running_loss, acc, score))
    writer.add_scalars('score', {'train': score}, epoch)
    writer.add_scalars('loss', {'train': loss}, epoch)
    writer.add_scalars('acc', {'train': acc}, epoch)

    # Validation Stage
    model.eval()
    for task in subtasks:
        task["model"].eval()
    running_loss_val, correct = 0., 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            images = images.to(device)
            tgts = targets.to(device)
            logits = model(images)
            loss = criterion(logits, tgts)
            preds = predict_staged(logits)
            correct += torch.sum(preds == tgts).item()
            pred_table_val += torch.sum(
                (preds == classes).unsqueeze(0) &
                (tgts == classes).unsqueeze(1), dim=2)
            running_loss_val += loss.item() * images.size(0)

            for k, (val_iter_sub, task) in enumerate(zip(val_iters_sub,
                                                         subtasks)):
                try:
                    imgs, tgts = next(val_iter_sub)
                except StopIteration:
                    train_iters_sub[k] = iter(subtasks[k]['valloader'])
                    imgs, tgts = next(val_iters_sub[k])

                imgs = imgs.to(device)
                tgts = tgts.to(device)
                out = task["model"](imgs)
                cur_loss = task["criterion"](out, tgts)
                loss_total += cur_loss
                losses_sub[k] = cur_loss.item()

            if (i+1) % args.log_step == 0:
                print(f'Validation\tEpoch: [{epoch+1}/{args.epochs}]\t' +
                      f'Batch:[{i+1}/{len(val_loader)}]\t{loss}\t{losses_sub}')
    running_loss_val /= len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)
    score = calc_score(pred_table_val)
    writer.add_scalars('score', {'val': score}, epoch)
    writer.add_scalars('loss', {'val': running_loss_val}, epoch)
    writer.add_scalars('acc', {'val': acc}, epoch)
    print(("Validation\tEpoch: [{}/{}]\tLoss: " +
          "{:.4f}\tAcc: {:.4f}\tScore: {:.4f}")
          .format(epoch+1, args.epochs + args.trained_epochs,
                  running_loss_val, acc, score))

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
        torch.save(model.state_dict(),
                   os.path.join(args.model_dir,
                                "model-epoch-{}.pth".format(epoch+1)))
    return acc, score


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
