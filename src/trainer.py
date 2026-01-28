import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .model import LSFMModel  # 修改导入
import torch


class ModelTrainer:
    def __init__(self, data_module, num_classes, max_epochs=50):
        self.data_module = data_module
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.model = self._initialize_model()

    def _initialize_model(self):
        return LSFMModel(self.num_classes)  # 使用新模型

    def train(self):
        """训练模型"""
        # 回调函数
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath='checkpoints/',
            filename='lsfm-best-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max'
        )

        early_stop_callback = EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max'
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            callbacks=[checkpoint_callback, early_stop_callback],
            accelerator='gpu',
            devices=1,
            precision = 16, # 使用混合精度训练加速
        )

        trainer.fit(self.model, self.data_module)
        return self.model

    def evaluate(self):
        """评估模型"""
        trainer = pl.Trainer(accelerator='gpu', devices=1)
        results = trainer.test(self.model, dataloaders=self.data_module.val_dataloader())
        return results