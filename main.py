import torch
from src.data_loader import prepare_dataframe, create_path_label_list, ImageDataModule
from src.trainer import ModelTrainer
from src.utils import visualize_images, generate_classification_report


def main():
    # 配置参数
    DATA_DIR = "E:/pythonproject/Data"                                      #替换为数据目录
    BATCH_SIZE = 32
    MAX_EPOCHS = 200
    DEVICE = 'cuda'    # 设备配置

    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA不可用，将使用CPU")

    # 准备数据
    print("准备数据...")
    data_df, class_names, _ = prepare_dataframe(DATA_DIR)
    path_label_list = create_path_label_list(data_df)

    # 强制设置为4个类别
    if len(class_names) < 4:
        print(f"警告: 数据集中只有 {len(class_names)} 个类别，强制设置为4个类别")
        # 补充类别名称
        class_names = class_names + [f"Class_{i}" for i in range(len(class_names), 4)]

    print(f"使用类别数量: 4")
    print(f"类别名称: {class_names[:4]}")

    # 创建数据模块
    print("创建数据加载器...")
    data_module = ImageDataModule(path_label_list, batch_size=BATCH_SIZE)
    data_module.setup()

    # 初始化并训练模型
    print("开始训练模型...")
    trainer = ModelTrainer(
        data_module=data_module,
        num_classes=4,
        max_epochs=MAX_EPOCHS
    )
    model = trainer.train()

    # 评估模型
    print("评估模型...")
    trainer.evaluate()

    # 可视化结果
    print("可视化结果...")
    visualize_images(data_module.val_dataloader())

    # 生成分类报告
    print("生成分类报告...")
    report = generate_classification_report(
        model,
        data_module.val_dataloader(),
        class_names[:4],  # 只使用前4个类别名称
        device=DEVICE  # 指定使用GPU设备
    )
    print(report)


if __name__ == "__main__":
    main()
