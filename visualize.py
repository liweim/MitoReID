from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import manifold

color8 = list(mpl.colors.BASE_COLORS.keys())


def display(target_table, f, pids, classes, save_path, verbose=False, id_col="moa_id", target_col="common_moa"):
    plt.figure(figsize=(10, 10))
    reduced = manifold.TSNE(random_state=1, perplexity=15).fit_transform(f)

    pid_list = list(set(pids))
    pid_list.sort()

    target_list = target_table[target_col].values.tolist()
    id_list = target_table[id_col].values.astype(str).tolist()

    targets = []
    for pid in pid_list:
        if pid in id_list:
            index = id_list.index(pid)
            target = target_list[index]
        else:
            target = pid
        targets.append(target)

    if len(pid_list) <= 8:
        colors = color8
    else:
        colors = [plt.cm.tab20b(i) for i in np.linspace(0, 1, 20)] + [plt.cm.tab20c(i) for i in np.linspace(0, 1, 20)]

    for i, (pid, target) in enumerate(zip(pid_list, targets)):
        size = 50
        index = np.where(pids == pid)[0]
        if verbose:
            color = 'r'
        else:
            color = colors[min(i, len(colors) - 1)]
        if str(target).startswith("nc"):
            target = target.replace("nc_", "")
        target = str(target).replace('.0', '')
        texts = [t[0].upper() + t[1:] for t in target.split(" ")]
        target = f"{i + 1}. {' '.join(texts)}"

        cls = classes[index]
        for c in ["gallery", 'query']:
            index2 = np.where(cls == c)[0]
            if c == "query":
                marker = "*"
                size *= 5
                alpha = 1
                target = None
                edgecolors = 'black'
                linewidth = 0.5
            else:
                marker = "o"
                alpha = 1
                edgecolors = 'white'
                linewidth = 0.3

            x, y = reduced[index[index2], 0], reduced[index[index2], 1]
            plt.scatter(x, y, s=size, color=color, marker=marker, label=target, alpha=alpha,
                        edgecolors=edgecolors, linewidth=linewidth)

            if c == 'gallery':
                cx, cy = np.median(x) + 2, np.median(y) + 2
                t = str(i + 1)
                colr = 'k'
                plt.text(cx, cy, t, fontsize=13, color=colr, family="arial")

    plt.legend(bbox_to_anchor=(0.1, -0.6), loc=3, borderaxespad=0, numpoints=1, fontsize=8, markerscale=2,
               frameon=False, ncol=2)
    plt.subplots_adjust(bottom=0.34)

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(save_path, format='svg', dpi=600)


def compute_feature(model, save_path, queryloader, device):
    model.eval()

    with torch.no_grad():
        f, pids, drugs = [], [], []
        for imgs, pid, drug in tqdm(queryloader):
            imgs = imgs.to(device)

            y, features = model(imgs)
            features = features.data.cpu()
            f.append(features)
            pids.extend(pid)
            drugs.extend(drug)
        f = torch.cat(f, 0).numpy().astype(np.float16)
        pids = np.asarray(pids)
        drugs = np.asarray(drugs)

        df = pd.DataFrame(f)
        df['label'] = pids
        df['drug'] = drugs
        df.to_csv(save_path, index=None)


def display_query_gallery(query_path, gallery_path, save_path):
    df = pd.read_csv(gallery_path, low_memory=False)
    df["class"] = "gallery"

    df2 = pd.read_csv(query_path, low_memory=False)
    df2["class"] = "query"
    df = pd.concat([df, df2])
    no_dmso = df[df['label'] != 'dmso']
    dmso = df[df['label'] == 'dmso']
    dmso = dmso.sample(frac=0.2, random_state=42)
    df = pd.concat([no_dmso, dmso])

    df = df.values
    f = df[:, :-3]
    pids = df[:, -3].astype(str)
    classes = df[:, -1]
    target_table = pd.read_excel("utils/annotation.xlsx", sheet_name="FDA-list")
    display(target_table, f, pids, classes, save_path, verbose=False, id_col="moa_id", target_col="common_moa")


if __name__ == "__main__":
    prefix = 'feature/mitosnet_resnet50_pretrain_38_kinetics'
    display_query_gallery(prefix + "-query.csv", prefix + "-gallery.csv", prefix + "-query_gallery_tsne.svg")
