from pynvml import *

nvmlInit()
# device_count = nvmlDeviceGetCount()
# print(f"Device count: {device_count}")
device_count = 10000
for i in range(device_count):
    handle = nvmlDeviceGetHandleByIndex(0)
    power_usage = nvmlDeviceGetPowerUsage(handle)  # 返回值为毫瓦（mW）
    power_limit = nvmlDeviceGetEnforcedPowerLimit(handle) # 获取功耗上限
    print(f"GPU {i}: Current Power Usage = {power_usage/1000.0} W, Power Limit = {power_limit/1000.0} W")

nvmlShutdown()