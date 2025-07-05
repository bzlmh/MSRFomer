from PIL import Image
import os

# 原始数据集目录
dataset_dir = 'E:/second-year graduate student/smaller_than10'

# 遍历所有类别文件夹
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    # 遍历每个类别文件夹中的文件
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)

        # 检查文件是否为图像文件（根据文件扩展名）
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            try:
                # 打开图像文件
                image = Image.open(file_path)

                # 将图像保存为 JPEG 格式
                jpg_path = os.path.splitext(file_path)[0] + '.jpg'
                image.save(jpg_path, 'JPEG')
                print(f"已将文件 '{file_name}' 转换为 JPEG 格式")

                # 可选：如果需要，可以删除原始文件
                # os.remove(file_path)

            except Exception as e:
                print(f"无法处理文件 '{file_name}': {str(e)}")

print("转换完成")
