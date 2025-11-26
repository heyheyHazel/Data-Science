import torch.nn as nn
import torch

# BasicBlock：两个3x3卷积，通道数不变（或略有增加）
class BasicBlock(nn.Module):
    # 通道的扩展系数，应满足in_channels = expansion * out_channels
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__() # 调用父类的构造函数

        # 残差块的左侧部分 正常连接
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        # 短路管理：使用下采样调整维度确保两者可以相加 一个1x1卷积
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # 残差连接
        out = self.relu(out)

        return out


# Bottleneck：1x1降维，3x3卷积，1x1升维，通道数大幅增加
# 在深层网络中显著降低计算量
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        # 计算瓶颈层的中间通道数
        width = int(out_channel * (width_per_group / 64.)) * groups

        # 第一个1x1卷积 降维 stride=1 不改变特征图大小
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 第二个3x3卷积 负责空间特征提取 可能改变特征图大小
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        # 第三个1x1卷积 升维 stride=1 不改变特征图大小
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


# 完整的ResNet网络结构
class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top  # 是否包含顶层的全连接层
        self.in_channel = 64

        self.groups = groups    # 组卷积的组数
        self.width_per_group = width_per_group  # 每组的通道数

        # 使用7x7卷积核和最大池化层进行初步特征提取
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层 随着网络加深 逐步增加通道数和减少特征图尺寸
        # 降采样发生在layer2, layer3, layer4中 感受野逐步增大 学习更多全局特征
        self.layer1 = self._make_layer(block, 64, blocks_num[0])    # 不下采样 专注提取局部特征
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2) # 第一次降采样
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2) # 第二次降采样
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2) # 第三次降采样

        # include_top为True时包含完整的分类头部（训练分类任务）
        # 否则省略分类头部（特征提取或迁移学习）
        if self.include_top:
            # 使用全局平均池化 例如[B,2048,7,7] → [B,2048,1,1]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            # 全连接层 将特征映射转换为类别分数
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重 遍历所有网络层
        for m in self.modules():
            # 检查是否有卷积层
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming正态初始化 卷积层适用于ReLU激活函数
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # 负责每个阶段的残差块堆叠
    def _make_layer(self, block, channel, block_num, stride=1):
        ''' 
        args:
            block: 残差块的类型(BasicBlock或Bottleneck)
            channel: 当前层的输出通道数
            block_num: 当前层残差块的数量
            stride: 当前层第一个残差块的步距
        '''
        downsample = None

        # 两种情况下需要下采样：stride不为1，或者输入输出通道数不匹配
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 定义下采样操作 使用1x1卷积调整通道数和尺寸
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        # 添加层中的第一个残差块，可能包含下采样
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        # 更新输入通道数
        self.in_channel = channel * block.expansion

        # 添加剩余的残差块 stride固定为1 不用下采样 保持维度和通道数不变 深度提取特征
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        # 返回包含所有残差块的Sequential模块
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)
