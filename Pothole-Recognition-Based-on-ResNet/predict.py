import os, json, random
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from ResNet import resnet34, resnet50, resnet101
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 预测
def predict():

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 读取类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    # 读取类别字典
    with open(json_path, "r") as json_file:
        class_indict = json.load(json_file)

    print(f"类别映射: {class_indict}")

    model = resnet50(num_classes=1).to(device)

    # 加载最优模型权重
    weights_path = "./model/best_ResNet.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."

    # 尝试加载权重，如果遇到不匹配则尝试修复
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()    # 开启模型evaluation模式

    # 定义文件夹路径
    base_path = "data_custom/val"
    pothole_path = os.path.join(base_path, "pothole")
    normal_path = os.path.join(base_path, "normal")

    # 随机选择五张图片
    def get_random_images(folder_path, num=5):
        if not os.path.exists(folder_path):
            return []
        images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return random.sample(images, min(num, len(images)))

    pothole_images = get_random_images(pothole_path)
    normal_images = get_random_images(normal_path)

    # 准备所有要预测的图片
    images_to_predict = []

    # 添加pothole图片
    for img_name in pothole_images:
        images_to_predict.append({
            'path': os.path.join(pothole_path, img_name),
            'true_label': 'pothole'
        })

    # 添加normal图片
    for img_name in normal_images:
        images_to_predict.append({
            'path': os.path.join(normal_path, img_name),
            'true_label': 'normal'
        })

    print(f"Selected {len(images_to_predict)} images for prediction")

    # 创建2行5列的子图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi = 300)
    axes = axes.ravel()

    # 对每张图片进行预测
    with torch.no_grad():
        for i, img_info in enumerate(images_to_predict):
            img_path = img_info['path']
            true_label = img_info['true_label']

            # 加载和预处理图片
            img = Image.open(img_path).convert('RGB')
            img_tensor = data_transform(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)

            # 预测
            output = model(img_tensor.to(device))

            # 根据模型输出类型处理预测结果
            if output.shape[1] == 1:
                # 二分类情况 - 使用sigmoid
                probability = torch.sigmoid(output).cpu().item()
                # 假设类别0是normal，类别1是pothole
                predicted_class = 1 if probability > 0.5 else 0
                predicted_label = class_indict[str(predicted_class)]
                prob_text = f"P({predicted_label}): {probability:.3f}"


            # 判断预测是否正确
            is_correct = (predicted_label == true_label)
            color = 'green' if is_correct else 'red'

            # 显示图片
            axes[i].imshow(img)
            axes[i].set_title(f"True: {true_label}\n{prob_text}\n{'✓' if is_correct else '✗'}",
                             color=color, fontsize=10)
            axes[i].axis('off')

            # 打印详细信息
            print(f"Image {i+1}: {os.path.basename(img_path)}")
            print(f"  True: {true_label}, Predicted: {predicted_label}, Prob: {probability:.3f}")
            print(f"  {'CORRECT' if is_correct else 'WRONG'}")

            # 保存预测结果用于后续统计
            img_info['predicted_label'] = predicted_label
            img_info['probability'] = probability
            img_info['is_correct'] = is_correct

    # 隐藏多余的子图
    for i in range(len(images_to_predict), 10):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('plt/prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 统计准确率
    correct_predictions = sum(1 for img_info in images_to_predict if img_info['is_correct'])
    accuracy = correct_predictions / len(images_to_predict)
    print(f"\nOverall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(images_to_predict)})")

if __name__ == "__main__":
    predict()