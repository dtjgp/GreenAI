'''
This code is used to plot the analysis of the data.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# load the interpolate function
def integrate_power_over_interval(samples, start_time, end_time):
    # 假定 samples是按时间升序排序的 (t, p)
    # 若未排序，请先排序:
    # samples = sorted(samples, key=lambda x: x[0])
    
    def interpolate(samples, target_time):
        # 在 samples 中找到 target_time 左右最近的两个点，并进行线性插值
        # 若 target_time 恰好等于某个样本点时间，直接返回该点功率
        # 若无法找到两侧点（如 target_time在样本时间轴外），根据情况返回None或边界点
        n = len(samples)
        if n == 0:
            return None
        # 若 target_time 小于第一个样本点时间，无法向左插值，这里直接返回第一个点的功率值(或None)
        if target_time <= samples[0][0]:
            # 简化处理：返回最早样本点的功率（或None）
            return samples[0][1]
        # 若 target_time 大于最后一个样本点时间，无法向右插值，返回最后一个点的功率（或None）
        if target_time >= samples[-1][0]:
            return samples[-1][1]

        # 否则，在中间插值
        # 使用二分查找快速定位
        import bisect
        times = [t for t, _ in samples]
        pos = bisect.bisect_left(times, target_time)
        # pos是使times保持有序插入target_time的位置
        # 因为target_time不在已有样本点中，pos不会越界且pos>0且pos<n
        t1, p1 = samples[pos-1]
        t2, p2 = samples[pos]
        # 线性插值： p = p1 + (p2 - p1)*((target_time - t1)/(t2 - t1))
        ratio = (target_time - t1) / (t2 - t1)
        p = p1 + (p2 - p1)*ratio
        return p

    # 从原始 samples 中筛选出位于[start_time, end_time]内的点
    filtered = [(t, p) for t, p in samples if start_time <= t <= end_time]

    # 如果不足2个点，则尝试使用插值
    if len(filtered) < 2:
        # 无论如何都需要在边界处插值出两个点(起码start和end)
        start_power = interpolate(samples, start_time)
        end_power = interpolate(samples, end_time)

        # 如果从样本中无法插值出任何有意义的点（比如samples为空或无法插值），返回0.0
        if start_power is None or end_power is None:
            return 0.0

        # 将插值的边界点加入到 filtered
        # 注意：如果filtered中有一个点在区间内，我们也需要确保边界有两点以上
        # 例如filtered只有一个点在中间，则需要在start和end插值点全部加入。
        # 若filtered为空，则只用start/end两点插值点求积分
        new_filtered = [(start_time, start_power)] + filtered + [(end_time, end_power)]
        # 确保按时间排序
        new_filtered.sort(key=lambda x: x[0])
        filtered = new_filtered

    # 正常积分计算
    if len(filtered) < 2:
        # 经过插值仍不够，返回0
        return 0.0

    total_energy = 0.0
    for i in range(len(filtered)-1):
        t1, p1 = filtered[i]
        t2, p2 = filtered[i+1]
        dt = t2 - t1
        avg_p = (p1 + p2)/2.0
        total_energy += avg_p * dt

    return total_energy


# Read the data
def load_data(model_path):
    # load the csv files 
    energy_data = pd.read_csv(os.path.join(model_path, 'energy_consumption_file.csv'))
    labeled_energy_data = pd.read_csv(os.path.join(model_path, 'labeled_energy_data.csv'))

    # load the npy files
    to_device = np.load(os.path.join(model_path, 'to_device.npy'), allow_pickle=True)
    forward = np.load(os.path.join(model_path, 'forward.npy'), allow_pickle=True)
    loss = np.load(os.path.join(model_path, 'loss.npy'), allow_pickle=True)
    backward = np.load(os.path.join(model_path, 'backward.npy'), allow_pickle=True)
    optimize = np.load(os.path.join(model_path, 'optimize.npy'), allow_pickle=True)

    # load the energy data
    to_device_energy = np.load(os.path.join(model_path, 'to_device_energy.npy'), allow_pickle=True)
    forward_energy = np.load(os.path.join(model_path, 'forward_energy.npy'), allow_pickle=True)
    loss_energy = np.load(os.path.join(model_path, 'loss_energy.npy'), allow_pickle=True)
    backward_energy = np.load(os.path.join(model_path, 'backward_energy.npy'), allow_pickle=True)
    optimize_energy = np.load(os.path.join(model_path, 'optimize_energy.npy'), allow_pickle=True)
    
    return energy_data, labeled_energy_data, to_device, forward, loss, backward, optimize, \
            to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy

# Plot the energy data of each sample with scatter plot and line plot
def plot_energy_data(labeled_energy_data, step_colors, step_markers):
    # Plot the data with a larger figure size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each step with a different marker
    for step in step_colors.keys():
        step_data = labeled_energy_data[labeled_energy_data['step'] == step]
        ax.scatter(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step, s=5, marker=step_markers[step])

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Power in Watts')
    ax.legend()
    plt.show()

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot each step with a different color
    for step in step_colors.keys():
        step_data = labeled_energy_data[labeled_energy_data['step'] == step]
        if step != 'idle':
            ax.plot(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Power in Watts')
    ax.legend()
    plt.show()


# Plot the energy data of a time period with scatter plot
def plot_period_energy_data(labeled_energy_data, step_colors):
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12,8))

    labeled_energy_data_rows = labeled_energy_data[2000:2200]

    # Plot each step with a different color, except 'optimize' and 'loss'
    for step in step_colors.keys():
        if step not in ['optimize', 'loss']:
            step_data = labeled_energy_data_rows[labeled_energy_data_rows['step'] == step]
            ax.scatter(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step)

    # Plot 'optimize' and 'loss' steps last to ensure they are on top
    for step in ['optimize', 'loss']:
        step_data = labeled_energy_data_rows[labeled_energy_data_rows['step'] == step]
        ax.scatter(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step)

    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Power in Watts')
    ax.legend()
    plt.show()

# plot each step energy data in each batch
def plot_batch_step_energy(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy, step_colors):
    batch_info = ['to_device', 'forward', 'loss', 'backward', 'optimize']
    energy_data_dict = {
        'to_device': to_device_energy,
        'forward': forward_energy,
        'loss': loss_energy,
        'backward': backward_energy,
        'optimize': optimize_energy
    }

    # plot the energy consumption for each step
    fig, ax = plt.subplots(figsize=(10,6))
    for step in batch_info:
        energy_data = energy_data_dict[step]
        energy_consumption = [energy_data[epoch][batch][2] for epoch in range(to_device_energy.shape[0]) for batch in range(to_device_energy.shape[1])]
        ax.plot(range(len(energy_consumption)), energy_consumption, color=step_colors[step], label=step)

    ax.set_xlabel('Batch Index(Total 5 Epochs)')
    ax.set_ylabel('Energy Consumption per batch in Joules')
    ax.legend()
    plt.show()


