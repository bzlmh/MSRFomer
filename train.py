import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from torchvision import transforms
from torchsummary import summary
from torchvision import models, transforms
from dataloader import HWDB
from model import MultiScaleNetwork
from msa import MSA
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载已训练的 ResNet，并冻结参数
def load_trained_resnet(model_path):
    model = models.resnet50(num_classes=1000)
    model.load_state_dict(torch.load(model_path))

    for param in model.parameters():
        param.requires_grad = False

    model = nn.Sequential(*list(model.children())[:-1])

    return ModifiedResNet(model)


# 提取中间层特征
class ModifiedResNet(nn.Module):
    def __init__(self, trained_resnet):
        super(ModifiedResNet, self).__init__()

        self.layer1 = nn.Sequential(*list(trained_resnet.children())[:5])  # 64通道 (N, 64, 112, 112)
        self.layer2 = nn.Sequential(*list(trained_resnet.children())[5:6])  # 256通道 (N, 256, 56, 56)
        self.layer3 = nn.Sequential(*list(trained_resnet.children())[6:7])  # 512通道 (N, 512, 28, 28)

    def forward(self, x):
        x1 = self.layer1(x)  # 第一层特征
        # print(x1.shape)
        x2 = self.layer2(x1)  # 第二层特征
        # print(x2.shape)
        x3 = self.layer3(x2)  # 第三层特征
        # print(x3.shape)

        return x1, x2, x3


# 训练函数
def train(epoch, net, msa_layer1, msa_layer2, msa_layer3, model, criterion, optimizer, train_loader, writer, save_iter=100):
    print(f"Epoch {epoch} 开始训练...")
    model.train()
    msa_layer1.train()
    msa_layer2.train()
    msa_layer3.train()

    sum_loss = 0.0
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 只使用原始字符图像提取特征
        ori256, ori512, ori1024 = net(images)

        # 空间注意力融合
        SCA_layer1 = msa_layer1(ori256)
        SCA_layer2 = msa_layer2(ori512)
        SCA_layer3 = msa_layer3(ori1024)

        # 前向传播
        outputs = model(SCA_layer1, SCA_layer2, SCA_layer3)
        loss = criterion(outputs, labels)

        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if (i + 1) % save_iter == 0:
            batch_loss = sum_loss / save_iter
            acc = 100 * correct / total
            print(f"Epoch {epoch}, Batch {i + 1}, Loss: {batch_loss:.4f}, Accuracy: {acc:.4f}")

            writer.add_scalar("train_loss", batch_loss, global_step=i + len(train_loader) * epoch)
            writer.add_scalar("train_acc", acc, global_step=i + len(train_loader) * epoch)

            total = 0
            correct = 0
            sum_loss = 0.0


# 验证函数
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def valid(epoch, net, msa_layer1, msa_layer2, msa_layer3, model, test_loader, writer):
    print(f"Epoch {epoch} 开始验证...")
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_labels = []
    all_predictions = []

    model.eval()
    msa_layer1.eval()
    msa_layer2.eval()
    msa_layer3.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            ori256, ori512, ori1024 = net(images)

            # 空间注意力融合
            SCA_layer1 = msa_layer1(ori256)
            SCA_layer2 = msa_layer2(ori512)
            SCA_layer3 = msa_layer3(ori1024)

            outputs = model(SCA_layer1, SCA_layer2, SCA_layer3)

            # 计算 Top-1 和 Top-5 准确率
            _, predicted_top1 = torch.max(outputs, 1)
            _, predicted_top5 = torch.topk(outputs, 5, dim=1)
            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()
            correct_top5 += (predicted_top5 == labels.unsqueeze(1)).sum().item()

            # 收集标签和预测结果，用于计算 F1-score
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_top1.cpu().numpy())

        # 计算准确率
        acc_top1 = 100 * correct_top1 / total
        acc_top5 = 100 * correct_top5 / total

        # 计算 F1-score
        f1 = f1_score(all_labels, all_predictions, average='macro')

        print(f"Epoch {epoch}, Top-1 验证准确率: {acc_top1:.2f}%")
        print(f"Epoch {epoch}, Top-5 验证准确率: {acc_top5:.2f}%")
        print(f"Epoch {epoch}, F1-score: {f1:.4f}")

        # 将结果写入 TensorBoard
        writer.add_scalar('valid_acc_top1', acc_top1, global_step=epoch)
        writer.add_scalar('valid_acc_top5', acc_top5, global_step=epoch)
        writer.add_scalar('valid_f1_score', f1, global_step=epoch)

if __name__ == "__main__":
    epochs = 600
    batch_size = 8
    lr = 0.00001

    data_path = r'./dataset/HWDB100'
    log_path = r'logs/batch_{}_lr_{}'.format(batch_size, lr)
    save_path = r'checkpoints/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 将图像大小调整为 224x224
        transforms.RandomAffine(
            degrees=10,  # 随机旋转角度范围
            translate=(0.1, 0.1),  # 随机平移范围
            scale=(0.9, 1.1),  # 随机缩放范围
            shear=5  # 随机剪切角度范围
        ),
        transforms.ToTensor()  # 将图像转换为 [0, 1] 范围的张量
    ])
    train_dataset = HWDB(path=data_path, transform=transform, mode='train')
    test_dataset = HWDB(path=data_path, transform=transform, mode='test')

    print("训练集数据大小:", len(train_dataset))
    print("测试集数据大小:", len(test_dataset))
    print("类别数量：", train_dataset.num_classes)

    train_loader = train_dataset.get_loader(batch_size=batch_size)
    test_loader = test_dataset.get_loader(batch_size=batch_size)

    net_path = 'Pretrained_model/resnet50_026.pth'  # 只保留原始图像模型
    net = load_trained_resnet(net_path).to(device)

    model = MultiScaleNetwork(
        num_classes=train_dataset.num_classes,
        embed_dims=[128, 256, 512],
        num_heads=[4, 8, 16],
        mlp_dims=[256, 512, 1024],
        num_layers=[2, 2, 2]
    ).to(device)

    msa_layer1 = MSA(256).to(device)
    msa_layer2 = MSA(512).to(device)
    msa_layer3 = MSA(1024).to(device)

    optimizer = optim.Adam(
        list(model.parameters()) +
        list(msa_layer1.parameters()) +
        list(msa_layer2.parameters()) +
        list(msa_layer3.parameters()),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_path)

    for epoch in range(epochs):
        train(epoch, net, msa_layer1, msa_layer2, msa_layer3, model, criterion, optimizer, train_loader, writer)
        valid(epoch, net, msa_layer1, msa_layer2, msa_layer3, model, test_loader, writer)

        torch.save(model.state_dict(), os.path.join(save_path, f'handwriting_iter_{epoch:03d}.pth'))
