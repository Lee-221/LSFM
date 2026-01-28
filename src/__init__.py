from .data_loader import CustomDataset, ImageDataModule, prepare_dataframe, create_path_label_list
from .model import LSFMModel  # 更新为新的模型类
from .trainer import ModelTrainer
from .utils import visualize_images, generate_classification_report
from .modules import ImageFeatureEnhancement, MultiScaleFeatureExtraction, SelfAttentionWeightAllocation  # 更新导出

__all__ = [
    'CustomDataset', 'ImageDataModule', 'prepare_dataframe', 'create_path_label_list',
    'LSFMModel', 'ModelTrainer', 'visualize_images', 'generate_classification_report',
    'ImageFeatureEnhancement', 'MultiScaleFeatureExtraction', 'SelfAttentionWeightAllocation'
]

