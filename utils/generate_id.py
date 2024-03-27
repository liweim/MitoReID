import pandas as pd
import networkx as nx


def format_str(key):
    if type(key) == str:
        if '；' in key:
            print()
        ks = key.lower().replace('  ', ' ').replace('；', ';').split(";")
        keys = []
        for k in ks:
            k = k.strip()
            if len(k) > 0:
                keys.append(k)
        keys.sort()
        key = ";".join(keys)
    else:
        key = ''
    return key


def get_target_id(path, target):
    df = pd.read_excel(path, sheet_name="FDA-list")
    count = {}
    keys = []
    for i in range(len(df)):
        key = df[target][i]
        key = format_str(key)
        keys.append(key)
        if key == "unknown" or key == "" or key == "nan":
            continue
        count[key] = count.get(key, 0) + 1
    df[target] = keys
    distinct_num = len(count)
    print(distinct_num)
    distinct_list = list(count.keys())
    distinct_list.sort()

    for i in range(len(df)):
        key = df[target][i]
        if key == "unknown" or key == "" or key == "nan":
            df["moa_id"][i] = key
            df["moa_count"][i] = -1
        else:
            df["moa_id"][i] = distinct_list.index(key)
            df["moa_count"][i] = count[key]

    save_path = path.replace(".xlsx", "_tmp.xlsx")
    df.to_excel(save_path, index=None)


def find_2th_neighbors(G, node):
    neighbors = []
    for n1 in list(nx.neighbors(G, node)):  # find 1_th neighbors
        neighbors.append(n1)
        for n2 in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            neighbors.append(n2)
    neighbors = set(neighbors)
    return neighbors


def find_3th_neighbors(G, node):
    neighbors = []
    for n1 in list(nx.neighbors(G, node)):  # find 1_th neighbors
        neighbors.append(n1)
        for n2 in list(nx.neighbors(G, n1)):  # find 2_th neighbors
            neighbors.append(n2)
            for n3 in list(nx.neighbors(G, n2)):  # find 3_th neighbors
                neighbors.append(n3)
    neighbors = set(neighbors)
    return neighbors


def get_target_id_from_graph(path):
    df = pd.read_excel(path, sheet_name="FDA-list")
    drugs = df["index"].values.astype(str).tolist()
    drug_set = set(drugs)
    target_df = pd.read_csv("../../data/drug_target_target.csv", low_memory=False)[["index", "target"]]
    target_df.columns = ["source", "target"]
    G = nx.from_pandas_edgelist(target_df)
    nodes = list(G.nodes())

    count = {}
    for i in range(len(df)):
        drug = drugs[i]
        if drug not in nodes:
            continue
        neighbors = find_2th_neighbors(G, drug) - drug_set
        neighbors = list(neighbors)
        neighbors.sort()
        key = ";".join(neighbors)
        df["target"][i] = key
        count[key] = count.get(key, 0) + 1
    distinct_num = len(count)
    print(distinct_num)
    distinct_list = list(count.keys())
    distinct_list.sort()

    for i in range(len(df)):
        key = df["target"][i]
        if key not in distinct_list:
            df["id"][i] = key
            df["count"][i] = -1
        else:
            df["id"][i] = distinct_list.index(key)
            df["count"][i] = count[key]

    save_path = path.replace(".xlsx", "_tmp.xlsx")
    df.to_excel(save_path, index=None)


def combine_target_action(target_path):
    df = pd.read_excel(target_path, sheet_name="FDA-list")
    target_action = []
    for i in range(len(df)):
        targets = df.loc[i, "target"].split(";")
        actions = df.loc[i, "action"].split(";")
        new_targets = []
        for target, action in zip(targets, actions):
            print(target, action)
            if action == "" or "unknown" in action or "other" in action:
                new_target = target
            else:
                new_target = f"{target} {action}"
            new_targets.append(new_target)
        new_targets = ";".join(new_targets)
        target_action.append(new_targets)
    df["target_action"] = target_action
    print(df)
    df.to_excel("target2.xlsx", index=None)


if __name__ == "__main__":
    target_path = "annotation.xlsx"
    get_target_id(target_path, "common_moa")


