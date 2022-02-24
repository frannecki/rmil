import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS = 1e-5  # Avoid division by zero
ERROR_TABLE = [[.0, .1, .7, 1.], [.1, .0, .3, .7],
               [.7, .3, .0, .3], [1., .7, .3, .0]]
ERROR_TABLE = torch.Tensor(ERROR_TABLE).to(device)


def predict_staged(logits):
    probs = torch.sigmoid(logits).sum(1)
    preds = probs.round().int()
    return preds


def calc_score(pred_table):
    score = 1 - torch.sum(ERROR_TABLE * pred_table.float()) \
        .item() / torch.sum(pred_table.float()).item()
    return score


def train_model(args, model, criterion, train_loader,
                val_loader, train_loader_anno, val_loader_anno,
                optimizer, epoch, writer):
    r"""Train model on wsi image tiles"""
    classes = torch.arange(args.num_classes).unsqueeze(1).to(device)

    pred_table_train = torch.zeros(args.num_classes,
                                   args.num_classes).long().to(device)
    pred_table_val = torch.zeros(args.num_classes,
                                 args.num_classes).long().to(device)

    # Training Stage
    train_loader_anno_iter = iter(train_loader_anno) \
        if args.pretrain else None

    model.train()
    running_loss, correct = 0., 0
    for i, (imgs, targets) in enumerate(train_loader):
        if args.pretrain:
            try:
                imgs_anno, targets_anno = next(train_loader_anno_iter)
            except StopIteration:
                train_loader_anno_iter = iter(train_loader_anno)
                imgs_anno, targets_anno = next(train_loader_anno_iter)
            imgs_anno = imgs_anno.unsqueeze(dim=1).to(device)
            targets_anno = targets_anno.to(device)
            logits_anno = model(imgs_anno)
            loss_anno = criterion(logits_anno, targets_anno)
        else:
            loss_anno = 0.

        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        optimizer.zero_grad()

        loss += loss_anno
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        if args.criterion == 'graded':
            preds = predict_staged(logits)
        else:
            preds = torch.argmax(logits, dim=1)
        correct += torch.sum(targets == preds).item()
        pred_table_train += torch.sum((preds == classes).unsqueeze(0) &
                                      (targets == classes).unsqueeze(1),
                                      dim=2)
        if (i+1) % args.log_step == 0:
            print(f'Training\tEpoch: [{epoch+1}/{args.epochs}]\t' +
                  f'Batch:[{i+1}/{len(train_loader)}]\t{loss}')
    running_loss /= len(train_loader.dataset)
    acc_train = correct / len(train_loader.dataset)
    score_train = calc_score(pred_table_train)
    print(('Training\tEpoch: [{}/{}]\tLoss: ' +
           '{:.4f}\tAcc: {:.4f}\tScore: {:.4f}')
          .format(epoch+1, args.epochs, running_loss,
                  acc_train, score_train))
    writer.add_scalars('score', {'train': score_train}, epoch)
    writer.add_scalars('loss', {'train': loss}, epoch)
    writer.add_scalars('acc', {'train': acc_train}, epoch)

    # Validation Stage
    model.eval()
    running_loss_val, correct = 0., 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            if args.criterion == 'graded':
                preds = predict_staged(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == targets).item()
            pred_table_val += torch.sum((preds == classes).unsqueeze(0) &
                                        (targets == classes).unsqueeze(1),
                                        dim=2)
            running_loss_val += loss.item() * imgs.size(0)
            if (i+1) % args.log_step == 0:
                print(f'Validation\tEpoch: [{epoch+1}/{args.epochs}]\t' +
                      f'Batch:[{i+1}/{len(val_loader)}]\t{loss}')
    running_loss_val /= len(val_loader.dataset)
    acc_val = correct / len(val_loader.dataset)
    score_val = calc_score(pred_table_val)
    writer.add_scalars('score', {'val': score_val}, epoch)
    writer.add_scalars('loss', {'val': running_loss_val}, epoch)
    writer.add_scalars('acc', {'val': acc_val}, epoch)
    print(('Validating\tEpoch: [{}/{}]\tLoss: ' +
          '{:.4f}\tAcc: {:.4f}\tScore: {:.4f}')
          .format(epoch+1, args.epochs,
                  running_loss_val, acc_val, score_val))

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
        writer.add_scalars('recall_class_{}'.format(t),
                           {'train': recall_train, 'val': recall_val},
                           epoch)

        writer.add_scalars('precision_class_{}'.format(t),
                           {'train': precision_train, 'val': precision_val},
                           epoch)

    if (1 + epoch) % args.save_step == 0:
        torch.save(model.state_dict(),
                   os.path.join(args.model_dir,
                                "model-epoch-{}.pth".format(epoch+1)))
    return acc_val, score_val


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
            if args.criterion == 'graded':
                preds = predict_staged(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == tgts).item()
            pred_table += torch.sum((preds == classes).unsqueeze(0) &
                                    (tgts == classes).unsqueeze(1),
                                    dim=2)
    for t in range(args.num_classes):
        num = pred_table[t][t].item()
        precision = num / (torch.sum(pred_table[:, t]).item() + EPS)
        recall = num / (torch.sum(pred_table[t]).item() + EPS)
        print("Metrics for class {:d}: Recall {:.4f} Precision {:.4f}"
              .format(t, recall, precision))
    acc = correct / len(loader.dataset)
    score = calc_score(pred_table)
    return acc, score


def train_model_on_annotated_regions(args, model, criterion, train_loader,
                                     val_loader, optimizer, transform,
                                     transform_test, epoch, writer):
    r"""Train model on annotated regions"""
    # Training Stage
    model.train()
    running_loss, correct = 0., 0
    for i, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.unsqueeze(dim=1).to(device)
        targets = targets.to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

        if args.criterion == 'graded':
            preds = predict_staged(logits)
        else:
            preds = torch.argmax(logits, dim=1)
        correct += torch.sum(targets == preds).item()
        if (i+1) % args.log_step_region == 0:
            print(f'train\tEpoch: [{epoch+1}/{args.epochs_region}]\t' +
                  f'Batch:[{i+1}/{len(train_loader)}]\t{loss}')
    running_loss /= len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)
    print('Training\tEpoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'.format(
              epoch+1, args.epochs_region, running_loss, acc))
    writer.add_scalars('loss', {'train': loss}, epoch)
    writer.add_scalars('acc', {'train': acc}, epoch)

    # Validation Stage
    model.eval()
    running_loss_val, correct = 0., 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.unsqueeze(dim=1).to(device)
            targets = targets.to(device)
            logits = model(imgs)
            loss = criterion(logits, targets)
            if args.criterion == 'graded':
                preds = predict_staged(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == targets).item()
            running_loss_val += loss.item() * imgs.size(0)
            if (i+1) % args.log_step_region == 0:
                print(f'Validation\tEpoch: [{epoch+1}/{args.epochs_region}]\t'
                      f'Batch:[{i+1}/{len(val_loader)}]\t{loss}')
    running_loss_val /= len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    writer.add_scalars('loss', {'val': running_loss_val}, epoch)
    writer.add_scalars('acc', {'val': acc}, epoch)
    print('Validating\tEpoch: [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
          .format(epoch+1, args.epochs_region, running_loss_val, acc))
    return acc


def evaluate_model_on_annotated_regions(args, model, data_loader):
    r"""Evaluate model on annotated regions"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(data_loader):
            imgs = imgs.unsqueeze(dim=1).to(device)
            targets = targets.to(device)
            logits = model(imgs)
            if args.criterion == 'graded':
                preds = predict_staged(logits)
            else:
                preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == targets).item()
    acc = correct / len(data_loader.dataset)
    return acc
