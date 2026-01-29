import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .modules import ImageFeatureEnhancement, MultiScaleFeatureExtraction, SelfAttentionWeightAllocation


class LSFMModel(pl.LightningModule):

    def __init__(self, num_classes, in_channels=3, base_dim=16):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = max(num_classes, 4)

        # 初始特征提取
        self.conv1 = nn.Conv2d(in_channels, base_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_dim, base_dim * 2, 3, 2, 1)  # 下采样
        self.conv3 = nn.Conv2d(base_dim * 2, base_dim * 4, 3, 2, 1)  # 下采样

        # 三个核心模块 
        self.feature_enhancement = ImageFeatureEnhancement(dim=base_dim * 4)
        self.multi_scale_extraction = MultiScaleFeatureExtraction(dim=base_dim * 4)
        self.attention_allocation = SelfAttentionWeightAllocation(channels=base_dim * 4)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)  # 使用修正后的num_classes
        )

    def forward(self, x):
        # 初始特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 三个核心模块处理
        x = self.feature_enhancement(x)  # 图像特征增强
        x = self.multi_scale_extraction(x)  # 多尺度特征提取
        x = self.attention_allocation(x)  # 自注意力权重分配

        # 分类输出
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=1e-4)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {'loss': loss, 'preds': preds, 'targets': y}
