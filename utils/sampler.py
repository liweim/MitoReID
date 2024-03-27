from collections import defaultdict
import numpy as np

import torch
from torch.utils.data.sampler import Sampler


class RandomPidSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py.

    Args:
        data_source (Dataset): dataset to sample from.
        num_instances (int): number of instances per identity.
    """

    def __init__(self, data_source, num_instances=4):
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_identities = len(self.pids)
        self.duplicate = max(1, int(len(data_source) / (self.num_identities * num_instances)))

    def __iter__(self):
        ret = []
        last_end = None
        for k in range(self.duplicate):
            indices = torch.randperm(self.num_identities)
            while last_end is not None and indices[0].equal(last_end):
                indices = torch.randperm(self.num_identities)
            for i in indices:
                pid = self.pids[i]
                t = self.index_dic[pid]
                replace = False if len(t) >= self.num_instances else True
                t = np.random.choice(t, size=self.num_instances, replace=replace)
                ret.extend(t)
            last_end = indices[-1]
        return iter(ret)

    def __len__(self):
        return self.num_identities * self.num_instances * self.duplicate
