import torch
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np


class MultimodalDataset(Dataset):
    """
    简化的多模态数据集（MNIST图像 + 模拟文本）
    """

    def __init__(self, mnist_data, text_dim=100):
        self.mnist_data = mnist_data
        self.text_dim = text_dim

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        img, label = self.mnist_data[idx]

        # 展平图像
        img_flat = img.view(-1)  # [784]

        # 模拟文本特征（实际应用中可以是BERT embedding）
        # 这里简单用 label 的 one-hot + 噪声
        txt_feat = torch.randn(self.text_dim) * 0.1
        txt_feat[label] += 1.0  # one-hot with noise

        return img_flat, txt_feat, label


class SingleModalityWrapper(Dataset):
    """
    包装器：用来创建单模态数据集（隐藏某个模态）
    """

    def __init__(self, multimodal_dataset, modalities=['img', 'txt']):
        """
        Args:
            multimodal_dataset: MultimodalDataset
            modalities: List of modalities to keep ['img', 'txt']
        """
        self.dataset = multimodal_dataset
        self.modalities = modalities

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_flat, txt_feat, label = self.dataset[idx]

        if 'img' not in self.modalities:
            img_flat = torch.zeros_like(img_flat)
        if 'txt' not in self.modalities:
            txt_feat = torch.zeros_like(txt_feat)

        return img_flat, txt_feat, label


def create_federated_data(num_clients=10,
                          client_types=None,
                          samples_per_client=500):
    """
    创建联邦数据分割
    Args:
        num_clients: 客户端数量
        client_types: List of str, ['full', 'img_only', 'txt_only']
        samples_per_client: 每个客户端的样本数
    Returns:
        client_datasets: List of Dataset
        client_modality_flags: List of (has_img, has_txt)
    """
    print("Loading MNIST dataset...")

    # 加载 MNIST (不需要 transform，直接用 PIL Image)
    mnist_train = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    print(f"MNIST loaded with {len(mnist_train)} samples")

    # 创建多模态包装
    multimodal_data = MultimodalDataset(mnist_train, text_dim=100)

    # 如果未指定客户端类型，默认设置
    if client_types is None:
        # 1个全模态，剩余一半图像、一半文本
        client_types = ['full'] + \
                       ['img_only'] * (num_clients // 2) + \
                       ['txt_only'] * (num_clients - 1 - num_clients // 2)

    # 计算要使用的总样本数
    total_samples = min(samples_per_client * num_clients, len(multimodal_data))
    print(f"Using {total_samples} samples total for {num_clients} clients")

    # 随机分割数据
    subset_data, _ = random_split(
        multimodal_data,
        [total_samples, len(multimodal_data) - total_samples]
    )

    # 为每个客户端分配数据
    client_datasets = []
    client_modality_flags = []

    split_sizes = [samples_per_client] * num_clients
    split_sizes[-1] = total_samples - sum(split_sizes[:-1])  # 调整最后一个

    client_data_splits = random_split(subset_data, split_sizes)

    for i, (data_split, ctype) in enumerate(zip(client_data_splits, client_types)):
        if ctype == 'full':
            client_datasets.append(data_split)
            client_modality_flags.append((True, True))
            print(f"  Client {i}: Full modality (Img+Txt)")

        elif ctype == 'img_only':
            # 创建图像单模态数据集
            wrapped_data = SingleModalityWrapper(data_split, modalities=['img'])
            client_datasets.append(wrapped_data)
            client_modality_flags.append((True, False))
            print(f"  Client {i}: Image only")

        elif ctype == 'txt_only':
            # 创建文本单模态数据集
            wrapped_data = SingleModalityWrapper(data_split, modalities=['txt'])
            client_datasets.append(wrapped_data)
            client_modality_flags.append((False, True))
            print(f"  Client {i}: Text only")

        else:
            raise ValueError(f"Unknown client type: {ctype}")

    return client_datasets, client_modality_flags


def get_test_data():
    """获取测试集"""
    print("Loading MNIST test set...")
    mnist_test = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    return MultimodalDataset(mnist_test, text_dim=100)


def collate_fn(batch):
    """
    自定义 collate 函数，处理单模态情况下的零张量
    """
    imgs = []
    txts = []
    labels = []

    for img, txt, label in batch:
        imgs.append(img)
        txts.append(txt)
        labels.append(label)

    return torch.stack(imgs), torch.stack(txts), torch.tensor(labels)