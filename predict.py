import random

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from val import cosine_distance


def predict(target_path, model, save_path, queryloader, galleryloader, device, id_col, logger):
    model.eval()

    with torch.no_grad():
        qf, q_pids = [], []
        for imgs, pids, drugs in tqdm(queryloader):
            imgs = imgs.to(device)

            y, features = model(imgs)
            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)

        gf, g_pids = [], []
        for imgs, pids, drugs in tqdm(galleryloader):
            imgs = imgs.to(device)

            y, features = model(imgs)
            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)

    distmat = cosine_distance(qf, gf)
    distmat = distmat.cpu().numpy()

    indices = np.argsort(distmat, axis=1)
    distmat = np.sort(distmat, axis=1)
    rank_g_pids = g_pids[indices]
    df = pd.DataFrame(q_pids, columns=['pid'])
    df['rank1'] = rank_g_pids[:, 0]
    series = df.groupby(['pid', 'rank1'])['rank1'].count()
    stat_df = series / series.groupby(['pid']).sum()
    df["cosine_distance"] = distmat[:, 0]
    dist_df = df.groupby(['pid', 'rank1'])["cosine_distance"].mean()

    stat_df = stat_df.fillna(0)
    df = stat_df.copy()
    df = pd.concat([df, dist_df], axis=1)
    df.rename(columns={df.columns[0]: 'tmp'}, inplace=True)
    df = df.reset_index()
    df = df.dropna()
    df.rename(columns={df.columns[1]: 'pred', df.columns[2]: 'rank1'}, inplace=True)
    df = df.sort_values(by=['pid', 'rank1', 'cosine_distance'], ascending=[True, False, True])

    df["pred"] = df["pred"].astype(str)
    target_df = pd.read_excel(target_path, sheet_name="FDA-list")
    target_df["index"] = target_df["index"].astype(str)

    nc_target_df = pd.read_excel(target_path, sheet_name="Novel-list")
    # nc_target_df = target_df
    df["pid"] = df["pid"].astype(str)
    nc_target_df["index"] = nc_target_df["index"].astype(str)
    df = pd.merge(df, nc_target_df[["index", "name"]], left_on="pid", right_on="index")

    if id_col == "drug_id":
        target_df["index"] = target_df["index"].astype(str)
        df = pd.merge(df, target_df[["index", "name", "moa_id"]], left_on="pred", right_on="index")
        del df["index_x"]
        del df["index_y"]
    elif id_col == "moa_id":
        target_df["moa_id"] = target_df["moa_id"].astype(str)
        df = pd.merge(df, target_df[["moa_id", "common_moa"]].drop_duplicates(), left_on="pred", right_on="moa_id")
        del df["index"]
        del df["moa_id"]
    df.rename(columns={'name': 'pid_name', 'common_moa': 'pred_name'}, inplace=True)
    if len(df) > 0:
        df = df.sort_values(by=['pid', 'rank1', 'cosine_distance'], ascending=[True, False, True])
        df.to_excel(save_path, index=None)
        logger.info('save result to ' + save_path)
    else:
        logger.info('no result to save')