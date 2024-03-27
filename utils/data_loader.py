import torch
from torch.utils.data import Dataset
from PIL import Image


class CellDataset(Dataset):
    def __init__(self, dataset, num_frame, type, spatial_transform=None):
        self.dataset = dataset
        self.num_frame = num_frame
        self.type = type
        self.spatial_transform = spatial_transform

    def __getitem__(self, item):
        tracklet, pid, drug = self.dataset[item]
        imgs = []
        for i in range(1, self.num_frame + 1):
            path = f"{tracklet}/{i}.jpg"
            img = Image.open(path)
            imgs.append(img)

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            imgs = [self.spatial_transform(img) for img in imgs]
            if self.type == "flow":
                imgs = [img[1:, :, :] for img in imgs]
        clip = torch.stack(imgs, 0).permute(1, 0, 2, 3)
        return clip, pid, drug

    def __len__(self):
        return len(self.dataset)