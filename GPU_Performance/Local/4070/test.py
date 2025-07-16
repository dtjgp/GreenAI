import pynvml
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from time import time

# 初始化 NVML
pynvml.nvmlInit()

# 获取 GPU 句柄（假设使用第 0 个 GPU）
gpu_index = 0
handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

# 获取 GPU 的默认功率范围
min_power = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[0]  # 最小功率限制
max_power = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1]  # 最大功率限制
print(f"GPU 支持的功率范围: {min_power // 1000}W - {max_power // 1000}W")

# # 检查功率范围是否适配
# if min_power > 45000 or max_power < 285000:
#     raise ValueError("指定功率范围超出设备支持范围")

# 准备数据集和模型
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

model = torchvision.models.resnet18(pretrained=False, num_classes=10)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 功率测试范围
power_limits = range(145000, 285001, 5000)  # 从 45W 到 285W，每 5W 一档

# 功率测试循环
results = []

for power_limit in power_limits:
    # 设置功率上限
    print(f"设置 GPU 功率上限为 {power_limit // 1000} W")
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit)

    # 开始训练
    print(f"开始训练 ResNet18 模型...")
    start_time = time()

    # 简化训练过程，只跑一个 epoch
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 获取 GPU 的实时频率
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)

            # # 打印频率信息
            # print(f"Iter {i + 1}: Graphics Clock: {graphics_clock} MHz, "
            #       f"Memory Clock: {memory_clock} MHz, SM Clock: {sm_clock} MHz")
            
            # # 打印功率信息
            # power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
            # print(f"Iter {i + 1}: Power Usage: {power_usage / 1000:.2f} W")

            # # 打印进度
            # if i % 100 == 99:
            #     print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
            #     running_loss = 0.0

    end_time = time()
    duration = end_time - start_time

    # 记录结果
    results.append({"power_limit": power_limit // 1000, "time": duration})
    print(f"功率上限 {power_limit // 1000} W 训练完成，耗时 {duration:.2f} 秒")
    # 打印功率信息和频率信息
    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
    print(f"Power Usage: {power_usage / 1000:.2f} W")
    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    print(f"Graphics Clock: {graphics_clock} MHz, Memory Clock: {memory_clock} MHz, SM Clock: {sm_clock} MHz")

# 恢复默认功率限制
default_power_limit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_power_limit)
print(f"恢复默认功率限制为 {default_power_limit // 1000} W")

# 释放 NVML
pynvml.nvmlShutdown()

# 打印结果
print("\n功率限制测试结果:")
for result in results:
    print(f"功率上限: {result['power_limit']} W, 耗时: {result['time']:.2f} 秒")