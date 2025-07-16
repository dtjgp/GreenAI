import time
import pynvml

# 初始化 NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 测量 100 次调用的平均时间
num_samples = 100
start_time = time.time()
for _ in range(num_samples):
    power = pynvml.nvmlDeviceGetPowerUsage(handle)
end_time = time.time()

# 计算平均调用时间
average_time = (end_time - start_time) / num_samples
print(f"Average nvmlDeviceGetPowerUsage call time: {average_time * 1000:.3f} ms")

# 关闭 NVML
pynvml.nvmlShutdown()