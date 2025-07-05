import os
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class HWDB(Dataset):
    def __init__(self, path, transform=None, mode='train'):
        """
        初始化数据集类
        path: 数据集根目录路径
        transform: 数据预处理（如图像变换）
        mode: 'train' 或 'test'，选择数据集模式
        """
        self.path = path
        self.transform = transform
        self.mode = mode

        # 获取训练集和测试集的路径
        self.train_dir = os.path.join(path, 'train')
        self.test_dir = os.path.join(path, 'test')

        # 选择对应模式的数据集路径
        if self.mode == 'train':
            self.dataset_dir_original = os.path.join(self.train_dir)
        elif self.mode == 'test':
            self.dataset_dir_original = os.path.join(self.test_dir)
        else:
            raise ValueError("Mode should be 'train' or 'test'")

        # 获取所有类别文件夹
        self.folders = os.listdir(self.dataset_dir_original)
        self.num_classes = len(self.folders)

        # 收集所有图像的路径和对应的类别标签
        self.samples = []
        for label, folder_name in enumerate(self.folders):
            original_folder = os.path.join(self.dataset_dir_original, folder_name)
            image_files = os.listdir(original_folder)

            for image_file in image_files:
                original_path = os.path.join(original_folder, image_file)
                self.samples.append((original_path, label))

        # 计算数据集大小
        self.dataset_size = len(self.samples)

    def __len__(self):
        """返回数据集的大小"""
        return self.dataset_size

    def __getitem__(self, index):
        """
        获取指定索引的样本
        index: 需要获取的样本索引
        返回原图及其对应的类别标签
        """
        original_path, label = self.samples[index]

        # 读取原图
        original_img = Image.open(original_path).convert('RGB')

        # 应用预处理
        if self.transform:
            original_img = self.transform(original_img)

        return original_img, label

    def get_loader(self, batch_size=100):
        """
        返回数据加载器
        batch_size: 批大小
        """
        return DataLoader(self, batch_size=batch_size, shuffle=(self.mode == 'train'))


if __name__ == '__main__':
    # 定义数据预处理操作
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像调整为 224x224
        transforms.ToTensor(),  # 转换为 Tensor 格式
    ])

    # 创建训练集数据集对象
    data_path = './dataset/HWDB100/'  # 假设数据路径为 HWDB100 文件夹
    train_dataset = HWDB(path=data_path, transform=transform, mode='train')
    print("训练集数据大小:", len(train_dataset))
    print("训练集类别数量：", train_dataset.num_classes)

    # 创建测试集数据集对象
    test_dataset = HWDB(path=data_path, transform=transform, mode='test')
    print("测试集数据大小:", len(test_dataset))
    print("测试集类别数量：", test_dataset.num_classes)

    # 获取数据加载器
    train_loader = train_dataset.get_loader(batch_size=16)
    test_loader = test_dataset.get_loader(batch_size=16)

    # 查看测试集的一批数据
    for batch_idx, (original_imgs, labels) in enumerate(test_loader):
        # 显示原图
        for i in range(len(original_imgs)):
            plt.imshow(original_imgs[i].permute(1, 2, 0).cpu().numpy())  # 转换为numpy格式并调整维度
            plt.title(f'Test Label: {labels[i]}')
            plt.show()
        break  # 只展示一批数据
