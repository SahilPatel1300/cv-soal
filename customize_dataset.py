import numpy as np
import os
import torch
from torch.utils.data import Dataset

class DexNetNPZDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.files = sorted([f for f in os.listdir(tensor_dir) if f.endswith('.npz')])
        self.samples = []
        self.labels = []
        for f in self.files:
            path = os.path.join(tensor_dir, f)
            if 'depth' in path:
                data = np.load(path)
                # Assume each npz file contains multiple samples
                for i in range(len(data['arr_0'])):
                    self.samples.append((path, i))  # store index in each file
            if 'label' in path:
                data = np.load(path)
                # Assume each npz file contains multiple samples
                for i in range(len(data['arr_0'])):
                    self.labels.append((path, i))  # store index in each file
        #sort the samples and labels
        self.samples.sort()
        self.labels.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file_path, i = self.samples[idx]
        label_file_path,i = self.labels[idx]
        data = np.load(img_file_path)
        label_data = np.load(label_file_path)
        
        depth = data['arr_0'][i].squeeze(-1)
        label = label_data['arr_0'][i]  # or 'grasp_qualities' depending on field
        depth = np.expand_dims(depth, 0)  # (1, H, W)
        return torch.tensor(depth, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


## Any extra processing