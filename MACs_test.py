import torch
from torchprofile import profile_macs
from torchvision.models import resnet50
from torchvision.models import resnet18
import pretty_errors
from googlenet_FashionMnist import Googlenet

# model = resnet50()
# inputs = torch.randn(1, 3, 224, 224)

# # 计算模型的MACs
# macs = profile_macs(model, inputs)
# print(f'MACs: {macs}')


# 实例化模型
model = Googlenet()

# 准备输入数据的示例，确保其尺寸匹配模型的输入要求
inputs = torch.randn(1, 3, 224, 224)

# 遍历模型的每一层
for name, module in model.named_modules():
    try:
        # 只对具有可计算MACs的层进行分析
        if len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            macs = profile_macs(module, inputs)
            print(f'{name}: {macs} MACs')
            # 更新inputs为当前层的输出，以便于传递给下一层
            inputs = module(inputs)
    except Exception as e:
        # 忽略无法计算MACs的层，如激活层、池化层等
        print(f"Skipping {name}, due to: {e}")
