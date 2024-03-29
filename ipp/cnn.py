import os.path
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import argparse
import math
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from resnet1d import ResNet
from feature_classifier import evaluate
import random

random.seed(0)


class IppDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        x = self.x[item]
        y = self.y[item]
        x = torch.tensor(x.T)
        return x, int(y)

    def __len__(self):
        return len(self.x)


def train(model, device, train_loader, optimizer, epoch, summarywriter):
    model.train()
    correct = 0
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = correct / len(train_loader.dataset)
    loss = total_loss / len(train_loader.dataset)
    summarywriter.add_scalar('train_acc', acc, epoch)
    summarywriter.add_scalar('train_loss', loss, epoch)
    print(f"Train epoch: {epoch}, lr: {optimizer.param_groups[0]['lr']}, "
          f"loss: {loss:.4f}, acc: {acc:.4f}")
    return acc, loss


def val(model, device, test_loader, epoch, summarywriter):
    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='mean').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        acc = correct / len(test_loader.dataset)
        loss = test_loss / len(test_loader.dataset)
        summarywriter.add_scalar('val_acc', acc, epoch)
        summarywriter.add_scalar('val_loss', loss, epoch)
        print(f'Test epoch: {epoch}, loss: {loss:.4f}, acc: {acc:.4f}')
    return acc, loss


def _test(model, device, test_loader):
    model.eval()

    y_pred = []
    y_test = []
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data = data.to(device)
            y_test += target.tolist()

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            y_pred += pred.cpu().squeeze().tolist()
    evaluate(y_test, y_pred, 'val')


def run(args):
    BATCH_SIZE = 64
    EPOCHS = 100
    tol = 5
    lr_patient = 3
    val_period = 3
    DEVICE = torch.device('cuda')
    resume_path = args.resume_path
    mode = args.mode
    save_path = args.checkpoint
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    x_train, x_test, y_train, y_test = np.load(args.data_path, allow_pickle=True)
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    x_train = x_train.reshape([len(x_train), -1, 16])
    x_train = np.asarray([x.T for x in x_train])
    x_test = x_test.reshape([len(x_test), -1, 16])
    x_test = np.asarray([x.T for x in x_test])
    in_channels = x_train.shape[-1]
    classes = len(set(y_train))

    model = ResNet(in_channels=in_channels, classes=classes)
    model = model.to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = lr_scheduler.ReducelrOnPlateau(optimizer, 'max', factor=0.1, patience=lr_patient, verbose=True)

    best_metric = 0
    begin_epoch = 0
    if resume_path != "":
        print('resuming checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        try:
            best_metric = checkpoint['best_metric']
            begin_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            optimizer.param_groups[0]['lr'] = args.lr
            # scheduler.load_state_dict(checkpoint['scheduler_state'])
        except Exception as e:
            print(f"Exception: {e}!")
            pass

    train_loader = torch.utils.data.DataLoader(IppDataset(x_train, y_train), batch_size=BATCH_SIZE,
                                               shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(IppDataset(x_test, y_test), batch_size=BATCH_SIZE,
                                              shuffle=False)

    summarywriter = SummaryWriter('./tensorboard/')
    if mode == 'train':
        print("start training")
        last_improve = begin_epoch
        epoch = begin_epoch
        while (epoch < EPOCHS):
            train(model, DEVICE, train_loader, optimizer, epoch, summarywriter)
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'optimizer_state': optimizer.state_dict(),
                # 'scheduler_state': scheduler.state_dict(),
            }
            if (epoch + 1) % val_period == 0:
                val_acc, val_loss = val(model, DEVICE, test_loader, epoch, summarywriter)
                # scheduler.step(val_acc)
                is_best = val_acc > best_metric
                if is_best:
                    last_improve = epoch
                best_metric = max(val_acc, best_metric)
                state["best_metric"] = best_metric
                # state["scheduler_state"] = scheduler.state_dict()

                if is_best:
                    print("improved")
                    torch.save(state, f'{save_path}/best.pth')
                else:
                    print(f"didn't not improve from {best_metric}")
                    torch.save(state, f'{save_path}/last.pth')
            else:
                torch.save(state, f'{save_path}/last.pth')

            if epoch - last_improve > tol * val_period:
                print("No optimization for a long time, auto-stopping...")
                best_model = ResNet(in_channels=in_channels, classes=classes)
                best_model = best_model.to(DEVICE)
                checkpoint = torch.load(f'{save_path}/best.pth')
                best_model.load_state_dict(checkpoint['state_dict'], strict=True)
                _test(best_model, DEVICE, test_loader)
                break
            epoch += 1
        print("finished epoch")
    elif mode == 'test':
        _test(model, DEVICE, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/l123_center_zscore.npy')
    parser.add_argument('--checkpoint', default="checkpoint")
    parser.add_argument('--resume_path', default="checkpoint/best.pth")
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--mode', default='train')
    args = parser.parse_args()

    run(args)
