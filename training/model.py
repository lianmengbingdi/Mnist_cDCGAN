import torch
import torch.nn as nn
import torch.nn.functional as fun

# 正态分布随机初始化函数
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, 256, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(256)
        self.deconv1_2 = nn.ConvTranspose2d(10, 256, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 1, 4, 2, 1)

    # 参数初始化
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = fun.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = fun.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = fun.relu(self.deconv2_bn(self.deconv2(x)))
        x = fun.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))
        return x


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = nn.Conv2d(512, 1, 4, 1, 0)

    # 参数初始化
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        x = fun.leaky_relu(self.conv1_1(input), 0.2)
        y = fun.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = fun.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = fun.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = torch.sigmoid(self.conv4(x))
        return x


