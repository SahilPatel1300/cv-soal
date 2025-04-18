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
        #make this depth 3 color channel
        depth = np.repeat(depth, 3, axis=0)
        return torch.tensor(depth, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


from glob import glob

class DexNetNPZDatasetAll(Dataset):
    def __init__(self, tensor_dir, use_regression=False, pose_dims=[2]):
        """
        tensor_dir: path to the folder with .npz files
        use_regression: if True, use grasp metrics as labels; else use binary labels
        pose_dims: indices of pose elements to include (default: [2], only depth)
        """
        self.tensor_dir = tensor_dir
        self.use_regression = use_regression
        self.pose_dims = pose_dims

        # Get all files grouped by type and sorted by index
        self.depth_files = sorted(glob(os.path.join(tensor_dir, 'depth_ims_tf_table_*.npz')))
        self.pose_files = sorted(glob(os.path.join(tensor_dir, 'hand_poses_*.npz')))
        self.label_files = sorted(glob(os.path.join(tensor_dir, 'labels_*.npz')))
        self.metric_files = sorted(glob(os.path.join(tensor_dir, 'grasp_metrics_*.npz')))

        self.samples = []

        # Assume all files aligned by index
        for file_idx, depth_file in enumerate(self.depth_files):
            depth_data = np.load(depth_file)['arr_0']
            num_samples = depth_data.shape[0]
            for i in range(num_samples):
                self.samples.append((file_idx, i))  # reference to index in that file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.samples[idx]

        # Load all relevant files
        depth = np.load(self.depth_files[file_idx])['arr_0'][sample_idx].squeeze(-1)
        pose = np.load(self.pose_files[file_idx])['arr_0'][sample_idx]
        label = (
            np.load(self.metric_files[file_idx])['arr_0'][sample_idx]
            if self.use_regression
            else np.load(self.label_files[file_idx])['arr_0'][sample_idx]
        )

        # Slice only desired pose dimensions
        #pose = pose[self.pose_dims]

        # Prepare depth image: (1, 32, 32) → (3, 32, 32)
        depth = np.expand_dims(depth, axis=0)
        # depth = np.repeat(depth, 3, axis=0)

        return (
            torch.tensor(depth, dtype=torch.float32),
            torch.tensor(pose, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

## Any extra processing
