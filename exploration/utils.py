import re
import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from customize_dataset import DexNetNPZDataset
from customize_dataset import DexNetNPZDatasetAll

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

# Define all categories and their shape descriptions
categories = {
    'camera_poses_': (1000, 7),
    'hand_poses_': (1000, 6),
    'depth_ims_tf_table_': (466, 32, 32, 1),
    'labels_': (1000,),
    'traj_ids_': (1000,),
    'grasp_metrics_': (1000,),
    'camera_intrs_': (1000, 4),
    'grasped_obj_keys_': (1000,),
    'grasp_collision_metrics_': (1000,),
    'pile_ids_': (1000,)
}

# file loading functions 
def find_common_file_numbers(path):
    files = os.listdir(path)
    category_to_nums = {cat: set() for cat in categories}
    for fname in files:
        for cat in categories:
            if fname.startswith(cat) and fname.endswith('.npz'):
                try:
                    num = int(fname[len(cat):-4])
                    category_to_nums[cat].add(num)
                except:
                    continue
    # Find intersection of all sets
    common_nums = set.intersection(*category_to_nums.values())
    return sorted(list(common_nums))

def load_file(path, category, file_num):
    fname = f"{category}{file_num:05d}.npz"
    fpath = os.path.join(path, fname)
    return np.load(fpath)['arr_0']


def visualize_random_example(path):
    common_files = find_common_file_numbers(path)
    if not common_files:
        print("No common file numbers found across all categories.")
        return

    chosen_file_num = random.choice(common_files)

    # Load a sample file to determine valid index range
    depth_map = load_file(path, 'depth_ims_tf_table_', chosen_file_num)
    max_index = depth_map.shape[0]  # Likely 466
    chosen_index = random.randint(0, max_index - 1)

    print(f"Selected file number: {chosen_file_num:05d}, sample index: {chosen_index}\n")

    # Store and print/plot each category
    for category in categories:
        data = load_file(path, category, chosen_file_num)

        if category == 'depth_ims_tf_table_':
            image = data[chosen_index].squeeze()
            plt.figure()
            plt.title("Depth Map")
            plt.imshow(image, cmap='gray')
            plt.colorbar()
            plt.show()

        elif category == 'grasp_metrics_':
            plt.figure()
            plt.title("Grasp Metric (value)")
            plt.bar([0], [data[chosen_index]])
            plt.xticks([0], ['Grasp Metric'])
            plt.ylabel('Score')
            plt.show()

        else:
            print(f"{category}{chosen_file_num:05d} -> Example[{chosen_index}]: {data[chosen_index]}\n")

def analyze_directory(path):
    file_pattern = re.compile(r"^(.*?)(\d{5})\..+$")  # captures category and 5-digit number
    category_files = defaultdict(list)

    for filename in os.listdir(path):
        match = file_pattern.match(filename)
        if match:
            category, number_str = match.groups()
            category_files[category].append((filename, int(number_str)))

    # Print summary of categories
    for category, files in category_files.items():
        numbers = [num for _, num in files]
        print(f"Category: {category}")
        print(f"  Number of files: {len(files)}")
        print(f"  Number range: {min(numbers)} to {max(numbers)}")

    print("\nInspecting one file per category:")
    for category, files in category_files.items():
        sample_file = next(f for f in files if f[0].endswith('.npz'))[0]
        filepath = os.path.join(path, sample_file)
        print(f"\nSample file for category '{category}': {sample_file}")
        try:
            data = np.load(filepath)
            for key in data:
                print(f"  Key: {key}, Shape: {data[key].shape}")
        except Exception as e:
            print(f"  Could not load file '{sample_file}': {e}")

# model focused functions
def evaluate_accuracy(model, dataloader, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, poses, labels in dataloader:
            images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            poses = poses.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            outputs = model(images, poses)
            outputs = torch.sigmoid(outputs)
            predictions = (outputs >= threshold).float()  # binary threshold
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"✅ Accuracy: {accuracy * 100:.2f}%")
    return accuracy


def DexNetDataloader(tensor_dir="../dexnet_2.1/dexnet_2.1_eps_10/tensors/", use_regression=True, pose_dims=[2], val_split=0.2, batch_size=32):

    dataset = DexNetNPZDatasetAll(tensor_dir=tensor_dir, use_regression=use_regression, pose_dims=pose_dims)
    val_split = 0.2  # 20% for validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    return train_loader, val_loader

