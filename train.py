from tqdm import tqdm
import torch


def train(epoch, model, criterion_class, criterion_metric, criterion_center, weight, num_mislabel, optimizer, optimizer_center,
          train_loader, device, logger):
    model.train()
    train_losses = 0
    class_losses = 0
    metric_losses = 0
    center_losses = 0
    num_valid = 1

    for index, (imgs, pids, _) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        pids = pids.to(device)
        imgs = imgs.to(device)
        ys, fs = model(imgs)

        # ignore the loss of those non-control drugs predicting as control
        pred = ys.data.max(1, keepdim=True)[1]
        mislabel_index = ((pred.view_as(pids) == 0) * (pids != 0))
        if torch.sum(mislabel_index) < num_mislabel:
            valid_index = ~mislabel_index
        else:
            valid_index = range(len(pred))
        valid_ys = ys[valid_index]
        valid_fs = fs[valid_index]
        valid_pids = pids[valid_index]
        valid_pred = pred[valid_index]

        if criterion_class is None:
            loss = criterion_metric(valid_fs, valid_pids)
        elif criterion_metric is None:
            loss = criterion_class(valid_ys, valid_pids)
        else:
            class_loss = criterion_class(valid_ys, valid_pids)
            class_losses += class_loss.data
            metric_loss = criterion_metric(valid_fs, valid_pids)
            metric_losses += metric_loss.data
            center_loss = weight * criterion_center(valid_fs, valid_pids)
            center_losses += center_loss.data
            loss = class_loss + metric_loss + center_loss
        train_losses += loss.data

        loss.backward()
        optimizer.step()
        if criterion_class is not None and criterion_metric is not None:
            for param in criterion_center.parameters():
                param.grad.data *= (1. / weight)
            optimizer_center.step()
            optimizer_center.zero_grad()

        num_valid += int(valid_pred.shape[0])

    train_losses /= num_valid
    if criterion_class is not None and criterion_metric is not None:
        class_losses /= num_valid
        metric_losses /= num_valid
        center_losses /= num_valid
        extra_info = f"\tclass_loss: {class_losses:.4f}\tmetric_loss: {metric_losses:.4f}\tcenter_loss: {center_losses:.4f}"
    else:
        extra_info = ""
    logger.info(
        f"train epoch: {epoch}\tlr: {optimizer.param_groups[0]['lr']}\ttrain_loss: {train_losses:.4f}{extra_info}")

    return train_losses, class_losses, metric_losses, center_losses
