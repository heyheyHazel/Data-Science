import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from ResNet import resnet34, resnet50, resnet101

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
def data_transforms():
    # 写成字典形式 对训练和测试集分别进行不同的预处理
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    return data_transform

# 加载数据
def create_data_loaders(data_transform, batch_size=64):
    # 设置数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "./"))
    image_path = os.path.join(data_root, "data_custom")
    # 断言路径存在
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    # 使用datasets.ImageFolder加载数据 每个子目录当作一个类别并自动生成index
    train_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "train"),
        transform=data_transform["train"]
    )
    validate_dataset = datasets.ImageFolder(
        root=os.path.join(image_path, "val"),
        transform=data_transform["val"]
    )

    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    # 类别名映射到整数标签，如 {"normal": 0, "pothole": 1}
    class_to_idx = train_dataset.class_to_idx
    # 整数标签映射到类别名，如 {0: "normal", 1: "pothole"}
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 将类别索引映射保存为json文件 indent是缩进4个字符
    json_str = json.dumps(idx_to_class, indent=4)
    with open('class_indices.json', 'w') as json_file:  # 写入json文件
        json_file.write(json_str)

    print(f"数据集信息: {len(class_to_idx)}个类别 - {idx_to_class}")

    # 设置设备和工作进程数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    else:
        nw = 0

    print(f"Using {device} device.")
    print(f'Using {nw} dataloader workers every process')

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw
    )

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    return train_loader, validate_loader, train_num, val_num, device, len(class_to_idx)

# 设置模型
def setup_model(num_classes, device):
    # 二元分类输出层只有1个节点
    net = resnet50(num_classes = 1)
    net.to(device)

    # 使用二分类的损失函数
    loss_function = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    return net, loss_function, optimizer

# 训练一个epoch
def train_epoch(net, train_loader, loss_function, optimizer, device, epoch, epochs, train_num):
    net.train() # 模型的训练模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # 进度条
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch_idx, (images, labels) in enumerate(train_bar):
        images = images.to(device)
        # 标签形状从 [batch_size] 变为 [batch_size, 1] 和模型输出对齐
        labels = labels.to(device).float().unsqueeze(1)

        # 在每个batch开始之前清零梯度
        optimizer.zero_grad()

        # 前向传播 输入images转换为logits分数
        outputs = net(images)

        # 计算当前batch损失
        loss = loss_function(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # running_loss 是在 epoch 中累积每个 batch 的平均
        running_loss += loss.item()

        # 计算准确率 - 使用sigmoid将输出转换为概率，然后四舍五入得到预测标签
        with torch.no_grad():
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # 更新进度条
        current_acc = correct_predictions / total_samples
        train_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_acc:.4f}'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / train_num

    return epoch_loss, epoch_acc

# 验证一个epoch
def validate_epoch(net, validate_loader, loss_function, device, val_num):
    net.eval()  # 模型切换到evaluation模式 影响dropout batchnorm 但不关闭梯度计算
    running_loss = 0.0
    correct_predictions = 0

    # 验证阶段不计算梯度
    with torch.no_grad():
        val_bar = tqdm(validate_loader, desc="Validating")
        for images, labels in val_bar:
            # 转移到device
            images = images.to(device)
            # 标签形状从 [batch_size] 变为 [batch_size, 1] 和模型输出对齐
            labels = labels.to(device).float().unsqueeze(1)

            # 输出形状 [batch_size, 1]，未经过sigmoid的logits分数
            outputs = net(images)
            loss = loss_function(outputs, labels)
            running_loss += loss.item()

            # logits转化为概率值
            probabilities = torch.sigmoid(outputs)
            
            # 概率值转化为二分类预测标签 阈值0.5
            predicted = (probabilities > 0.5).float()

            # 计算准确率
            correct_predictions += (predicted == labels).sum().item()

            val_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(validate_loader)
    epoch_acc = correct_predictions / val_num

    return epoch_loss, epoch_acc

# 绘制曲线
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):

    # 绘制损失曲线
    plt.figure(figsize=(12, 4),dpi = 300)

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./plt/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主训练函数
def train():

    data_transform = data_transforms()

    train_loader, validate_loader, train_num, val_num, device, num_classes = create_data_loaders(data_transform)

    net, loss_function, optimizer = setup_model(num_classes, device)

    epochs = 100
    save_path = './model/best_ResNet.pth' # 保存最佳模型
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # 开始训练
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss, train_acc = train_epoch(
            net, train_loader, loss_function, optimizer, device, epoch, epochs, train_num
        )

        val_loss, val_acc = validate_epoch(
            net, validate_loader, loss_function, device, val_num
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
            print(f"Saved best model with val_acc: {val_acc:.4f}")


    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    train()