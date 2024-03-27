import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder


def plot_roc(q_pids, y_pred, cmc_path):
    y_test = np.array(q_pids).reshape((-1, 1))
    enc = OneHotEncoder().fit(y_test)
    y_test = enc.transform(y_test).toarray()
    y_pred = np.array(y_pred)
    pd.DataFrame(y_test).to_csv('result/query_roc_gt.csv', index=None)
    pd.DataFrame(y_pred).to_csv(cmc_path.replace('_cmc.csv', '_roc.csv'), index=None)
    exit()


def evaluate(distmat, q_pids, g_pids, q_drugs, g_drugs, logger, max_rank=50):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        logger.error("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query lib and drug
        q_drug = q_drugs[q_idx]
        # remove gallery samples that have the same lib and drug with query
        order = indices[q_idx]
        remove = g_drugs[order] == q_drug
        keep = np.invert(remove)
        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches

        # orig_cmc = matches[q_idx]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def cosine_distance(input1, input2):
    """Computes cosine distance, between 0~2, lower distance, better performance

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    input1_normed = F.normalize(input1, p=2, dim=1)
    input2_normed = F.normalize(input2, p=2, dim=1)
    distmat = 1 - torch.mm(input1_normed, input2_normed.t())
    return distmat


def val(mode, model, queryloader, galleryloader, criterion_class, logger, device, cmc_path=None):
    model.eval()
    val_losses = 0
    correct = 0

    with torch.no_grad():
        qf, q_pids, q_drugs, y_pred = [], [], [], []
        num_val = 0
        for imgs, pids, drugs in tqdm(queryloader):
            imgs = imgs.to(device)

            y, f = model(imgs)

            qf.append(f)
            q_pids.extend(pids)
            q_drugs.extend(drugs)
            m = nn.Softmax(dim=1)
            y_pred.extend(m(y).cpu().tolist())
            num_val += int(y.shape[0])

            if criterion_class != None:
                pids = pids.to(device)
                loss = criterion_class(y, pids)
                val_losses += loss.data
                pred = y.data.max(1, keepdim=True)[1]
                pred_correct = pred.eq(pids.data.view_as(pred)).cpu()
                correct += pred_correct.sum()

        # plot_roc(q_pids, y_pred, cmc_path)

        if criterion_class != None:
            val_losses /= num_val
            val_acc = correct / num_val

        q_pids = np.asarray(q_pids)
        q_drugs = np.asarray(q_drugs)
        gf, g_pids, g_drugs = [], [], []
        for imgs, pids, drugs in tqdm(galleryloader):
            imgs = imgs.to(device)

            y, f = model(imgs)

            gf.append(f)
            g_pids.extend(pids)
            g_drugs.extend(drugs)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_drugs = np.asarray(g_drugs)

        qf = torch.cat(qf, 0)
        distmat = cosine_distance(qf, gf)
        distmat = distmat.cpu().numpy()

        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_drugs, g_drugs, logger, max_rank=50)
        if mode == 'test' and criterion_class != None:
            logger.info(
                f"Rank-1\tRank-5\tRank-10\tmAP\tAcc")
            logger.info(
                f"{cmc[0]*100:.2f}\t{cmc[4]*100:.2f}\t{cmc[9]*100:.2f}\t{mAP*100:.2f}\t{val_acc*100:.2f}")
            df = pd.DataFrame([cmc], columns=range(1, 51))
            df.to_csv(cmc_path, index=None)
        else:
            logger.info(f"mAP: {mAP:.2%}\tRank-1: {cmc[0]:.2%}")
    return cmc[0], mAP


def val_multi(epoch, model, criterion_class, query_loader, device, num_classes, logger):
    model.eval()
    val_losses = 0
    correct = 0
    num_val = 0
    for imgs, pids, _ in tqdm(query_loader):
        imgs, pids = imgs.to(device), pids.to(device)
        y, f = model(imgs)
        loss = criterion_class(y, pids)
        val_losses += loss.data
        pred = torch.sigmoid(y) > 0.5
        num_val += int(pred.shape[0])
        correct += (pred.eq(pids.data.view_as(pred)).cpu().sum(dim=1) == num_classes).sum()

    val_acc = correct / num_val * 100
    val_losses /= num_val
    logger.info(f"val epoch: {epoch}\tlr: val_acc: {val_acc:.3f}\tval_loss: {val_losses:.3f}")
    return val_acc

