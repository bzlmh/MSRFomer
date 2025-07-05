import os
import cv2
import numpy as np
from PIL import Image

def resize_and_convert_images(src_dir, dst_dir, size=(224, 224)):
    """
    将指定文件夹中的所有子文件夹中的图像调整为指定大小，并确保它们是3通道的RGB图像。

    :param src_dir: 源文件夹路径
    :param dst_dir: 目标文件夹路径
    :param size: 目标图像大小 (width, height)
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # 遍历源文件夹中的所有子文件夹
    for dir_name in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, dir_name)
        if os.path.isdir(subfolder_path):
            # 为每个子文件夹创建一个对应的目标子文件夹
            subfolder_dst = os.path.join(dst_dir, dir_name)
            if not os.path.exists(subfolder_dst):
                os.makedirs(subfolder_dst)

            # 遍历子文件夹中的所有图像文件
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    src_path = os.path.join(subfolder_path, filename)
                    # 修改文件扩展名为 .jpg
                    dst_filename = os.path.splitext(filename)[0] + '.jpg'
                    dst_path = os.path.join(subfolder_dst, dst_filename)

                    # 使用 OpenCV 读取图像，直接以彩色模式读取图像
                    image = cv2.imread(src_path, cv2.IMREAD_COLOR)  # 以彩色模式读取图像

                    # 如果图像为空，跳过
                    if image is None:
                        print(f"Warning: Failed to load image {src_path}. Skipping...")
                        continue

                    # 调整图像大小
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

                    # 打印图像形状以验证通道数
                    print(f"Image {filename} shape: {image.shape}")  # 应该是 (224, 224, 3)

                    # 保存调整后的图像为 .jpg 格式
                    cv2.imwrite(dst_path, image)
                    print(f"Processed and saved {dst_path}")

if __name__ == '__main__':
    src_dir = "HWDB100/test/original"  # 源文件夹路径，包含子文件夹
    dst_dir = "HWDB100/test/original2"  # 目标文件夹路径，用于保存调整后的图像
    resize_and_convert_images(src_dir, dst_dir)
