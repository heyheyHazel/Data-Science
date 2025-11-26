import os
from shutil import copy, rmtree
import random

# os.chdir('/Users/lotus/Desktop/Pothole-Recognition-Based-on-ResNet')

def mk_file(file_path: str):
    ''' 
    创建文件夹
    file_path: 文件夹路径
    '''
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def split_data():
    random.seed(42)

    split_rate = 0.2    # 训练集测试集划分比例

    cwd = os.getcwd()
    data_root = os.path.join(cwd, "data_custom")
    # 指向原始存放数据的文件夹
    origin_pothole_path = os.path.join(data_root, "data_pothole")
    assert os.path.exists(origin_pothole_path), "path '{}' does not exist.".format(origin_pothole_path)

    # 遍历所有文件夹 每个文件夹中的数据为一个类别
    pothole_class = [cla for cla in os.listdir(origin_pothole_path)
                    if os.path.isdir(os.path.join(origin_pothole_path, cla))]

    # 准备目录结构
    # 建立保存训练集的文件夹
    train_root = os.path.join(data_root, "train")
    mk_file(train_root)
    for cla in pothole_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(train_root, cla))

    # 建立保存验证集的文件夹
    val_root = os.path.join(data_root, "val")
    mk_file(val_root)
    for cla in pothole_class:
        # 建立每个类别对应的文件夹
        mk_file(os.path.join(val_root, cla))

    # 随机划分数据
    for cla in pothole_class:
        cla_path = os.path.join(origin_pothole_path, cla)
        images = os.listdir(cla_path)
        num = len(images)
        # 随机采样验证集的索引
        eval_index = random.sample(images, k=int(num*split_rate))
        for index, image in enumerate(images):
            if image in eval_index:
                # 将分配至验证集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)  # 原图像路径
                new_path = os.path.join(val_root, cla)  # 目标路径
                copy(image_path, new_path)  # 从image_path复制到new_path
            else:
                # 将分配至训练集中的文件复制到相应目录
                image_path = os.path.join(cla_path, image)
                new_path = os.path.join(train_root, cla)
                copy(image_path, new_path)
            print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
        print()

    print("processing done!")


if __name__ == '__main__':
    split_data()