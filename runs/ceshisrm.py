import torch
import torch.nn as nn

# 定义 SRMLayer 类
class Stylemodule(nn.Module):
    def __init__(self, channel, reduction=None):
        super(Stylemodule, self).__init__()

        # CFC: channel-wise fully connected layer
        # 将输入通道数传入 groups 参数
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False, groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, h, w = x.size()  # 获取 batch_size, channels, height, width

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)  # 平均池化
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)    # 标准差池化
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # 将样本特征进行处理
        z = self.cfc(u)  # 使用Conv1d进行通道融合 (b, c, 1)
        z = self.bn(z)   # 批量归一化
        g = torch.sigmoid(z)  # 激活函数
        g = g.view(b, c, 1, 1)  # 变形回4D

        return x * g.expand_as(x)  # 扩展g并与输入x相乘


# 测试代码
def test_SRMLayer():
    # 定义输入张量：假设输入的尺寸为 (batch_size, channels, height, width)
    batch_size = 8
    channels = 64
    height = 56
    width = 56
    x = torch.randn(batch_size, channels, height, width)

    # 创建 SRMLayer 实例
    srm_layer = Stylemodule(channel=channels)

    # 前向传播
    output = srm_layer(x)

    # 输出尺寸
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

# 运行测试
test_SRMLayer()
