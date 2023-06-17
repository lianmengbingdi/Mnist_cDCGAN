# 人工智能大作业——Mnist条件生成器

## 目录

`data`：包含Mnist数据集

`training`：`model.py`是神经网络结构代码，`train.py`是神经网络的训练代码；`output`是训练过程中生成的图片以观察训练效果。

`interface`：接口类文件`aignmc.py`文件，其中定义了产生手写体数字的接口；`generator.pth`与`discriminator.pth`是训练得到的数据。

`report.pdf`：作业报告。

`try.py`：接口调用示例代码。

`result.png`：示例代码运行结果。

## aigcmn接口调用

引入模块`aigcmn.py`，并且利用接口初始化模型：

```python
from interface.aigcmn import AiGcMn
aigcmn = AiGcMn('interface/generator.pth')
```

用户输入0-9的数字，用空格分开，然后将数字分割转换后形成列表，再转换成tensor，作为`aigcmn.generate`的参数调用

```python
# 接收用户输入的数字字符串
input_str = input("请输入数字，使用空格分隔：")

# 将输入的字符串按空格分割并转换为整数
input_list = [int(num) for num in input_str.split()]

# 将列表中的元素转换为PyTorch张量
output_tensor = torch.tensor(input_list)
# 生成图像
gen_output = aigcmn.generate(output_tensor)
```

其返回一个`batch_size*1*28*28`的Tensor，即batch_size个单通道的28*28的灰度图，也是用户输入数字的个数。输出的图像保存在`/output`中：

```python
from torchvision.utils import save_image

# 保存生成的图像
save_image(gen_output, 'result.png')
print("成功保存生成的图像！")
```

完整示例代码可见：[try.py](./try.py)。



