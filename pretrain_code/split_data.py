import os
import shutil
import random

# 原始数据集目录
dataset_dir = 'E:/second-year graduate student/new_change_demo_data'

# 创建输出目录
output_dir = 'E:/ocr code/Handwriting-Recognition-pytorch-main/data'
train_dir = os.path.join(output_dir, 'train_data')
val_dir = os.path.join(output_dir, 'val_data')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 指定验证集的比例
validation_split = 0.2  # 20% 的数据作为验证集

# 获取所有类别文件夹
class_folders = os.listdir(dataset_dir)

# 遍历每个类别文件夹
for class_folder in class_folders:
    class_path = os.path.join(dataset_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    # 获取该类别下所有图片文件
    image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]

    # 随机打乱图片文件列表
    random.shuffle(image_files)

    # 计算分割点，用于划分验证集和训练集
    split_point = int(len(image_files) * validation_split)

    # 分割数据集
    val_files = image_files[:split_point]
    train_files = image_files[split_point:]

    # 创建类别文件夹
    class_train_dir = os.path.join(train_dir, class_folder)
    class_val_dir = os.path.join(val_dir, class_folder)
    os.makedirs(class_train_dir, exist_ok=True)
    os.makedirs(class_val_dir, exist_ok=True)

    # 复制训练集图片到相应目录
    for train_file in train_files:
        src = os.path.join(class_path, train_file)
        dst = os.path.join(class_train_dir, train_file)
        shutil.copy(src, dst)

    # 复制验证集图片到相应目录
    for val_file in val_files:
        src = os.path.join(class_path, val_file)
        dst = os.path.join(class_val_dir, val_file)
        shutil.copy(src, dst)
