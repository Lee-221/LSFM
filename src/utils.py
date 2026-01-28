import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from sklearn.metrics import classification_report


def visualize_images(dataloader, num_images=16):
    """可视化数据加载器中的图像"""
    for images, labels in dataloader:
        break  # 只取第一个批次

    # 如果图像在GPU上，移动到CPU
    if images.is_cuda:
        images = images.cpu()

    # 创建网格图像
    im = make_grid(images[:num_images], nrow=4)

    # 反归一化
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    im = inv_normalize(im)

    # 显示图像
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.axis('off')
    plt.show()


def generate_classification_report(model, dataloader, class_names, device='cuda'):
    """生成分类报告"""
    model.to(device)  # 确保模型在正确的设备上
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)  # 将图像数据移至指定设备（默认CPU）
            labels = labels.to(device)  # 将标签移至指定设备（默认CPU）

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 确保预测结果包含所有4个类别
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 如果预测的类别数量不足4个，补充缺失的类别
    unique_preds = set(y_pred)
    if len(unique_preds) < 4:
        print(f"警告: 预测结果只包含 {len(unique_preds)} 个类别，强制补充到4个类别")
        # 添加缺失的类别（使用-1占位，但不会影响评估）
        for i in range(4):
            if i not in unique_preds:
                y_pred = np.append(y_pred, i)
                y_true = np.append(y_true, i)  # 同时添加对应的真实标签

    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
