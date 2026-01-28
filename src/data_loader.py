import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from torchvision import transforms
from sklearn.model_selection import train_test_split  # 新增导入

class CustomDataset(Dataset):
    def __init__(self, path_label, transform=None):
        self.path_label = path_label
        self.transform = transform

    def __len__(self):
        return len(self.path_label)

    def __getitem__(self, idx):
        path, label = self.path_label[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, path_label, batch_size=32):
        super().__init__()
        self.path_label = path_label
        self.batch_size = batch_size
        self.transform = self._get_transforms()

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # 分离路径和标签用于分层抽样
        paths = [item[0] for item in self.path_label]
        labels = [item[1] for item in self.path_label]

        # 使用分层抽样确保每个类别在训练集和验证集中都有代表
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            paths,
            labels,
            test_size=0.2,  # 8:2 划分
            random_state=42,  # 固定随机种子
            stratify=labels  # 关键：分层抽样
        )

        # 重新组合为 (path, label) 格式
        train_data = list(zip(train_paths, train_labels))
        val_data = list(zip(val_paths, val_labels))

        self.train_dataset = CustomDataset(train_data, self.transform)
        self.val_dataset = CustomDataset(val_data, self.transform)

        # 打印划分信息用于调试
        print("=== 数据划分统计 ===")
        print(f"总样本数: {len(self.path_label)}")
        print(f"训练集样本数: {len(train_data)}")
        print(f"验证集样本数: {len(val_data)}")

        # 统计每个类别的分布
        from collections import Counter
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)

        print("训练集类别分布:")
        for label in sorted(set(labels)):
            count = train_counter.get(label, 0)
            percentage = count / len(train_data) * 100
            print(f"  类别 {label}: {count} 个样本 ({percentage:.1f}%)")

        print("验证集类别分布:")
        for label in sorted(set(labels)):
            count = val_counter.get(label, 0)
            percentage = count / len(val_data) * 100
            print(f"  类别 {label}: {count} 个样本 ({percentage:.1f}%)")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


def create_path_label_list(df):
    """创建路径-标签列表"""
    path_label_list = []
    for _, row in df.iterrows():
        path_label_list.append((row['path'], row['label']))
    return path_label_list


def prepare_dataframe(data_dir):
    """准备数据DataFrame"""
    classes = []
    paths = []

    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            # 只处理图像文件，忽略文本文件
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                class_name = os.path.basename(dirname)
                classes.append(class_name)
                paths.append(os.path.join(dirname, filename))

    # 创建类别映射
    class_names = sorted(set(classes))
    normal_mapping = {name: i for i, name in enumerate(class_names)}

    # 创建数据框
    data = pd.DataFrame({
        'path': paths,
        'class': classes,
        'label': [normal_mapping[cls] for cls in classes]
    })

    return data, class_names, normal_mapping
