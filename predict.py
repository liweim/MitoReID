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

    groups = df.groupby('pid')# 需要删除
    count = 0
    for id, data in groups:
        top5 = list(data['pred'][:5])

        drug_id = str(id)
        if '-' in drug_id:
            drug_id = drug_id.split('-')[0]
        vals = target_df[target_df['index'] == drug_id]['moa_id'].values
        if len(vals) > 0:
            val = str(vals[0])
            if val in top5:
                count += 1
                logger.info(f'correct: {id}, rank: {top5.index(val) + 1}')
    logger.info(f'num of correct: {count}')

    nc_target_df = pd.read_excel(target_path, sheet_name="Novel-list")
    # nc_target_df = target_df
    df["pid"] = df["pid"].astype(str)
    nc_target_df["index"] = nc_target_df["index"].astype(str)
    df = pd.merge(df, nc_target_df[["index", "name", "cas"]], left_on="pid", right_on="index")

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


if __name__ == "__main__":
    # 需要删掉
    target_df = pd.read_excel('utils/annotation.xlsx', sheet_name="FDA-list")
    target_df["index"] = target_df["index"].astype(str)

    path = 'predict/mitosnet_resnet50_pretrain_38_kinetics-nc.xlsx'
    df = pd.read_excel(path)
    groups = df.groupby('pid')
    freq_count = {}
    for id, data in groups:
        top3 = list(data['pred'][:3])
        for top in top3:
            if top not in freq_count:
                freq_count[top] = 0
            freq_count[top] += 1
    freq_count = sorted(freq_count.items(), key=lambda kv: kv[1], reverse=True)
    freq_id = [id for id, count in freq_count[:10]]

    filtered_df = pd.DataFrame()
    select_num = 0
    for id, data in groups:
        if random.random() < 0.05:
            filtered_df = pd.concat([filtered_df, data])
            select_num += 1
        else:
            top5 = list(data['pred'][:5])
            count = 0
            for top in top5:
                if top in freq_id:
                    count += 1
            if count < 4 or len(top5) < 5:
                filtered_df = pd.concat([filtered_df, data])
                select_num += 1
    filtered_df = filtered_df.sort_values(by=['pid', 'rank1', 'cosine_distance'], ascending=[True, False, True])
    print(filtered_df)
    print('select', select_num)
    filtered_df.to_excel(path.replace('.xlsx', '_select.xlsx'), index=None)
