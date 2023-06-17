import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils import data
from torchvision.utils import save_image
from model import Generator, Discriminator
from PIL import Image, ImageDraw, ImageFont

# 运行设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
batch_size = 128
lr = 0.0002
epoch_num = 12

# 训练结果保存路径
g_net_path = '../interface/generator.pth'
d_net_path = '../interface/discriminator.pth'

# 加载数据集
img_size = 32
transform = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
dataloader = data.DataLoader(datasets.MNIST('../data', train=True, transform=transform, download=True), 
                            batch_size=batch_size, 
                            shuffle=True,
                            drop_last=True)
dataloader_len = len(dataloader)

def encode_1hot(self, labels):
    # 对标签进行独热编码
    hot_label = self.onehot[labels.type(torch.LongTensor)]
    return hot_label
# 损失函数
BCE_loss = nn.BCELoss()

# 独热编码
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1).view(10, 10, 1, 1).to(device)

fill = torch.zeros([10, 10, img_size, img_size])
for i in range(10):
    fill[i, i, :, :] = 1
fill = fill.to(device)

# 固定噪声，用于检验生成器训练结果
noises = torch.randn(10, 100)
noise0 = noises
for i in range(9):
    noise0 = torch.cat([noise0, noises], 0)
noise0 = noise0.view(-1, 100, 1, 1)
noise0 = noise0.to(device)

# 计算噪声对应的独热编码
labels = torch.zeros(10, 1)
for i in range(9):
    temp = torch.ones(10, 1) + i
    labels = torch.cat([labels, temp], 0)
labels = onehot[labels.type(torch.LongTensor)].squeeze()
labels = labels.view(-1, 10, 1, 1)
labels = labels.to(device)

# 生成器、判别器实例化
generator = Generator()
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

# 生成器、判别器参数初始化
generator.weight_init(mean=0.0, std=0.02)
discriminator.weight_init(mean=0.0, std=0.02)

# 定义优化器
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

def resize_image(img):
    # 调整图像大小为28x28
    new_size = (28, 28)
    resized_image = img.resize(new_size)
    return resized_image
# 训练过程
for epoch in range(epoch_num):

    labels_real_ = torch.ones(batch_size).to(device)
    labels_fake_ = torch.zeros(batch_size).to(device)

    G_loss = 0
    D_loss = 0

    for step, (image, labels_) in enumerate(dataloader):
        # 训练判别器
        discriminator.zero_grad()

        mini_batch = image.size()[0]

        labels_fill_ = fill[labels_]
        image, labels_fill_ = image.to(device), labels_fill_.to(device)

        D_result = discriminator(image, labels_fill_).squeeze()
        D_real_loss = BCE_loss(D_result, labels_real_)

        noise = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        labels_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        labels_label_ = onehot[labels_]
        labels_fill_ = fill[labels_]
        noise, labels_label_, labels_fill_ = noise.to(device), labels_label_.to(device), labels_fill_.to(device)

        G_result = generator(noise, labels_label_)
        D_result = discriminator(G_result, labels_fill_).squeeze()
        D_fake_loss = BCE_loss(D_result, labels_fake_)

        D_train_loss = D_real_loss + D_fake_loss

        D_train_loss.backward()
        D_optimizer.step()

        # 更新数据
        with torch.no_grad():
            D_loss += D_train_loss.item()

        # 训练生成器
        generator.zero_grad()

        noise = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)
        labels_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
        labels_label_ = onehot[labels_]
        labels_fill_ = fill[labels_]
        noise, labels_label_, labels_fill_ = noise.to(device), labels_label_.to(device), labels_fill_.to(device)

        G_result = generator(noise, labels_label_)
        D_result = discriminator(G_result, labels_fill_).squeeze()

        G_train_loss = BCE_loss(D_result, labels_real_)

        G_train_loss.backward()
        G_optimizer.step()

        # 更新数据
        with torch.no_grad():
            G_loss += G_train_loss.item()

        if (step + 1) % 10 == 0 or (step + 1) == dataloader_len:
            print(f"Epoch[{epoch + 1}/{epoch_num}] , Batch[{step + 1}/{dataloader_len}] , 生成器损失: {G_loss/(step + 1)} , 判别器损失: {D_loss/(step + 1)}")


    # 生成器输出检验
    with torch.no_grad():
        generator_output = generator(noise0, labels)
        gen_resize = transforms.Resize(28)
        processed_output = torch.zeros(labels.shape[0], 1, 28, 28)
        for i in range(labels.shape[0]):
            processed_output[i] = gen_resize(generator_output[i])
        save_image(processed_output, 'output/epoch_{}.png'.format(epoch + 1))
        img = Image.open('output/epoch_{}.png'.format(epoch + 1))
        # 创建绘图对象
        draw = ImageDraw.Draw(img)
        # 设置文本和字体样式
        text = "epoch_{}.png".format(epoch+1)
        font = ImageFont.truetype("arial.ttf", 10)  # 字体样式和大小可根据需要进行调整

        # 获取文本的宽度和高度
        text_width, text_height = draw.textsize(text, font=font)

        # 计算绘制文本的起始坐标
        x = img.width - text_width - 10  # 图像宽度减去文本宽度，并留出一定的边距
        y = img.height - text_height - 10  # 图像高度减去文本高度，并留出一定的边距

        # 在图像的右下角绘制文本
        draw.text((x, y), text, font=font)

        # 保存输出
        img.save('output/epoch_{}.png'.format(epoch + 1))

        
      

torch.save(obj=Generator.state_dict(generator), f=g_net_path)
torch.save(obj=Discriminator.state_dict(discriminator), f=d_net_path)
