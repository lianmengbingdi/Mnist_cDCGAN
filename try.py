from interface.aigcmn import AiGcMn
import torch
from torchvision.utils import save_image

# 创建AiGcMn实例并加载模型权重
aigcmn = AiGcMn('interface/generator.pth')

# 定义标签

# 接收用户输入的数字字符串
input_str = input("请输入数字，使用空格分隔：")

# 将输入的字符串按空格分割并转换为整数
input_list = [int(num) for num in input_str.split()]

# 将列表中的元素转换为PyTorch张量
output_tensor = torch.tensor(input_list)
# 生成图像
gen_output = aigcmn.generate(output_tensor)

# 保存生成的图像
save_image(gen_output, 'result.png')
print("成功保存生成的图像！")
