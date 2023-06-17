import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as fun
from torchvision.transforms import Resize
import random
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



class AiGcMn:
    def __init__(self, gen_path):
        # 初始化生成器
        self.gen = Generator()
        self.gen.load_state_dict(torch.load(gen_path, map_location=torch.device('cpu')))
        self.resize = Resize(28)

        # 初始化独热编码表
        self.onehot = fun.one_hot(torch.arange(10), num_classes=10).float().view(10, 10, 1, 1)



    def generate(self, labels):
        # 获取批次大小

        batch_size = labels.size(0)
        # 进行独热编码
        hot_label = self.onehot[labels.type(torch.LongTensor)]
        # 生成随机噪声
        rd_noise = torch.randn(batch_size, 100, 1, 1)
        # 生成图像
        gen_output = self.gen(rd_noise, hot_label)
        # 处理为28*28大小
        output = torch.stack([self.resize(img) for img in gen_output])
        return output

        



