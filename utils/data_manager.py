import os
import os.path as osp
import random
import numpy as np
import glob
import pandas as pd

random.seed(0)


class Cell(object):
    def __init__(self, target_path, train_path, query_path, gallery_path, id_col, logger, choose_col, model_path,
                 relabel=False, is_predict=False, sample_rate=1):
        target_table = pd.read_excel(target_path, sheet_name="FDA-list")
        self.model_path = model_path
        self.drug_list = target_table['index'].values.astype(str).tolist()

        if id_col == "drug_id":
            self.group_list = self.drug_list
        else:
            self.group_list = target_table[id_col].values.astype(str).tolist()
        if choose_col == "all":
            self.filtered_drugs = self.drug_list
        else:
            index = target_table[choose_col].values == 1
            filtered_drugs = np.asarray(self.drug_list)[index]
            self.filtered_drugs = list(filtered_drugs)
        if is_predict:
            self.filtered_drugs.remove('dmso')

        self.pid2label = None
        train, num_train_pid = self._process_path(train_path, relabel=True, sample_rate=sample_rate)

        # self.pid2label = None  # don't use train's pid2label
        gallery, num_gallery_pid = self._process_path(gallery_path,
                                                      relabel=relabel,
                                                      sample_rate=sample_rate)
        query, num_query_pid = self._process_path(query_path,
                                                  relabel=relabel,
                                                  sample_rate=sample_rate, is_predict=is_predict)

        logger.info(f"num of train pid: {num_train_pid}, num of train tracklet: {len(train)}\n"
                    f"num of query pid: {num_query_pid}, num of query tracklet: {len(query)}\n"
                    f"num of gallery pid: {num_gallery_pid}, num of gallery tracklet: {len(gallery)}")

        self.train = train
        self.query = query
        self.gallery = gallery

    def _process_path(self, folder_path, relabel, sample_rate, is_predict=False):
        if folder_path == '':
            return [], 0

        dataset = []
        pid_set = []
        for path in folder_path.split(","):
            drugs = os.listdir(path)
            for drug in drugs:
                if '-' in drug:
                    drug = drug.split('-')[0]
                if is_predict:
                    pid = drug
                else:
                    if drug not in self.filtered_drugs:
                        continue
                    pid = self.group_list[self.drug_list.index(drug)]
                if pid == "unknown" or pid == "nan":
                    continue
                pid_set.append(pid)
        pid_set = list(set(pid_set))
        pid_set.sort(reverse=True)
        if self.pid2label is None:
            self.pid2label = {p: label for label, p in enumerate(pid_set)}

        for path in folder_path.split(","):
            lib = osp.split(osp.split(path)[0])[1]
            tracklets = glob.glob(path + "/*/*")
            for tracklet in tracklets:
                if tracklet.endswith(".avi"):
                    continue
                drug, cell = osp.split(osp.split(tracklet)[0])[1], osp.split(tracklet)[1]
                if '-' in drug:
                    drug = drug.split('-')[0]

                if is_predict:
                    pid = drug
                else:
                    if drug not in self.drug_list:
                        continue
                    pid = self.group_list[self.drug_list.index(drug)]
                if pid not in pid_set:
                    continue
                lib_drug = f"{lib}_{drug}"
                if relabel:
                    if pid not in self.pid2label:
                        continue
                    data = [tracklet, self.pid2label[pid], lib_drug]
                else:
                    data = [tracklet, pid, lib_drug]
                if random.random() <= sample_rate:
                    dataset.append(data)
        num_pid = len(pid_set)
        return dataset, num_pid
