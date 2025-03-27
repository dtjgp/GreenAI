import torch
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# 设置设备：优先使用 mps（适用于 macOS），否则回退到 cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 加载预训练的 AlexNet 模型，并设置为评估模式
model = models.alexnet(pretrained=True)
model.eval()
model.to(device)

# 定义数据预处理：
# 1. 将图像 resize 到 256（保证短边不小于 256）
# 2. CenterCrop 到 224×224，符合 AlexNet 输入要求
# 3. 转换为 tensor，并归一化（使用 ImageNet 的均值和标准差）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载 CIFAR100 测试集（train=False），数据会自动下载到 './data' 目录
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 预热操作：多次前向传播有助于稳定性能（这里预热 10 个样本）
with torch.no_grad():
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        _ = model(inputs)
        if i >= 10:
            break

# 测量推断时间：对前 num_samples 个样本进行时间测量，并计算平均推断时间
num_samples = 100
total_time = 0.0
with torch.no_grad():
    for i, (inputs, _) in enumerate(data_loader):
        if i >= num_samples:
            break
        inputs = inputs.to(device)
        # 对于 mps 设备，目前无需手动同步，直接计时即可
        start_time = time.time()
        _ = model(inputs)
        end_time = time.time()
        total_time += (end_time - start_time)

average_time = total_time / num_samples
print("Average inference time over {} samples: {:.6f} seconds".format(num_samples, average_time))