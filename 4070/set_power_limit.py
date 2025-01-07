import pynvml
import argparse

def set_power_limit(power_limit_watts):
    pynvml.nvmlInit()
    gpu_index = 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    
    # 将瓦特转换为毫瓦
    power_limit = int(power_limit_watts * 1000)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, power_limit)
    print(f"功率上限已设置为 {power_limit_watts} W")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='设置GPU功率限制')
    parser.add_argument('power_limit', type=int, help='功率限制（单位：瓦特）')
    args = parser.parse_args()
    set_power_limit(args.power_limit)