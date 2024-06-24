import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Directory containing the data.
root = 'data/'
train_root = "/new_data/yhang/GNSS/TEXBAT/train/"
test_root = "/new_data/yhang/GNSS/TEXBAT/test/"

def normalize_features(data):
    # 计算每个特征的平均值和标准差
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)

    # 避免除以0
    std[std == 0] = 1

    # 归一化数据
    normalized_data = (data - mean) / std

    return normalized_data


def get_data(dataset, batch_size, mode):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.MNIST(root+'mnist/', train='train', 
                                download=True, transform=transform)

    # Get TEXBAT dataset.
    elif dataset == 'TEXBAT' and mode == 'train':
        data_train = np.load("/new_data/yhang/GNSS/TEXBAT/train/ds34/data.npy")
        labels_train = np.load("/new_data/yhang/GNSS/TEXBAT/train/ds34/label.npy") 
        # 将numpy数组转换为torch张量
        data_torch = torch.from_numpy(data_train).float()
        data_torch = normalize_features(data_torch)
        labels_torch = torch.from_numpy(labels_train).float()

        # 创建一个TensorDataset
        dataset = TensorDataset(data_torch, labels_torch)

    elif dataset == 'TEXBAT' and mode == 'test':
        data_test = np.load("/new_data/yhang/GNSS/TEXBAT/test/ds34/data.npy")
        labels_test = np.load("/new_data/yhang/GNSS/TEXBAT/test/ds34/label.npy")   
        # 将numpy数组转换为torch张量
        data_torch = torch.from_numpy(data_test).float()
        data_torch = normalize_features(data_torch)
        labels_torch = torch.from_numpy(labels_test).float()

        # 创建一个TensorDataset
        dataset = TensorDataset(data_torch, labels_torch)
    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = dsets.FashionMNIST(root+'fashionmnist/', train='train', 
                                download=True, transform=transform)

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = dsets.ImageFolder(root=root+'celeba/', transform=transform)

    # Create dataloader.
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True, drop_last = True)

    return dataloader