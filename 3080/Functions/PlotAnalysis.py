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
def plot_energy_data(labeled_energy_data, step_colors, step_markers, model_name, plot_folder):
    # Plot the data with a larger figure size
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each step with a different marker
    for step in step_colors.keys():
        step_data = labeled_energy_data[labeled_energy_data['step'] == step]
        ax.scatter(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step, s=5, marker=step_markers[step])

    ax.set_xlabel(f'Timestamp Across All Samples in {model_name}')
    ax.set_ylabel('Power in Watts')
    ax.set_title(f'Energy Data of Each Sample in {model_name}')
    ax.legend()
    plt.show()
    # save the figure, to a specific directory
    plt.savefig(os.path.join(plot_folder, f'energy_data of {model_name}.png'))

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12,6))

    # Plot each step with a different color
    for step in step_colors.keys():
        step_data = labeled_energy_data[labeled_energy_data['step'] == step]
        if step != 'idle':
            ax.plot(step_data['timestamp'], step_data['power_in_watts'], color=step_colors[step], label=step)

    ax.set_xlabel(f'Timestamp Across All Samples in {model_name}')
    ax.set_ylabel('Power in Watts')
    ax.set_title(f'Energy Data of Each Sample in {model_name} in Line Plot')
    ax.legend()
    plt.show()
    # save the figure, to a specific directory
    plt.savefig(os.path.join(plot_folder, f'energy_data of {model_name} in line plot.png'))


# Plot the energy data of a time period with scatter plot
def plot_period_energy_data(labeled_energy_data, step_colors, model_name, plot_folder):
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

    ax.set_xlabel(f'Timestamp Across Selected Samples in {model_name}')
    ax.set_ylabel('Power in Watts')
    ax.set_title(f'Energy Data of a Time Period in {model_name}')
    ax.legend()
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'Selected period energy data of {model_name}.png'))


# plot each step energy data in each batch
def plot_batch_step_energy(to_device_energy, forward_energy, 
                           loss_energy, backward_energy, 
                           optimize_energy, step_colors, 
                           modelname, plot_folder):
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

    ax.set_xlabel(f'Batch Index(Total 5 Epochs) in all Epcohs of {modelname}')
    ax.set_ylabel('Energy Consumption in Joules')
    ax.set_title(f'Energy Consumption for Each Step in all Batches of {modelname}')
    ax.legend()
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'each step energy consumption in each batch of {modelname}.png'))


# plot each step energy data with all batch with the x axis set to each step
def plot_step_energy_distribution(to_device_energy, forward_energy, 
                                  loss_energy, backward_energy, 
                                  optimize_energy, step_colors,
                                  modelname, plot_folder):
    batch_info = ['to_device', 'forward', 'loss', 'backward', 'optimize']
    energy_data_dict = {
        'to_device': to_device_energy,
        'forward': forward_energy,
        'loss': loss_energy,
        'backward': backward_energy,
        'optimize': optimize_energy
    }

    # Define markers for each step
    markers = {
        'to_device': 'o',  # circle
        'forward': '^',    # triangle up
        'loss': 's',      # square
        'backward': 'D',   # diamond
        'optimize': 'v'    # triangle down
    }

    # Create figure with grid
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Plot each step's energy consumption with different markers and lighter colors
    for i, step in enumerate(batch_info):
        energy_data = energy_data_dict[step]
        energy_consumption = [energy_data[epoch][batch][2] 
                            for epoch in range(to_device_energy.shape[0]) 
                            for batch in range(to_device_energy.shape[1])]
        
        # Make colors lighter by adding alpha
        ax.scatter([i] * len(energy_consumption), 
                  energy_consumption,
                  color=step_colors[step], 
                  label=step,
                  marker=markers[step],
                  alpha=0.6,
                  s=50)  # Increase marker size

    # Customize plot appearance
    ax.set_xticks(range(len(batch_info)))
    ax.set_xticklabels(batch_info, rotation=45)
    ax.set_xlabel(f'Batch Steps in {modelname}', fontsize=12)
    ax.set_ylabel('Energy Consumption in Joules', fontsize=12)
    ax.set_title(f'Energy Consumption Distribution for Each Step\nin all Batches of {modelname}', 
                 fontsize=14, pad=20)
    
    # Place legend outside of plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    
    # Save the figure
    plt.savefig(os.path.join(plot_folder, 
                f'each_step_energy_consumption_distribution_{modelname}.png'),
                bbox_inches='tight', dpi=300)


# plot each step mean and std energy data with all batch with the x axis set to each step
def plot_step_energy_distribution_bar(to_device_energy, forward_energy, 
                                      loss_energy, backward_energy, 
                                      optimize_energy, modelname, plot_folder):
    # Use a more subtle color palette
    step_colors = {
        'idle': '#999999',  # light gray
        'to_device': '#1f77b4',  # muted blue
        'forward': '#2ca02c',  # muted green
        'loss': '#d62728',  # muted red
        'backward': '#9467bd',  # muted purple
        'optimize': '#ff7f0e'  # muted orange
    }

    energy_data_dict = {
        'to_device': to_device_energy,
        'forward': forward_energy,
        'loss': loss_energy,
        'backward': backward_energy,
        'optimize': optimize_energy
    }

    # Calculate the mean and standard deviation for each step
    mean_energy_consumption = {step: np.mean([energy_data[epoch][batch][2] for epoch in range(to_device_energy.shape[0]) for batch in range(to_device_energy.shape[1])]) for step, energy_data in energy_data_dict.items()}
    std_energy_consumption = {step: np.std([energy_data[epoch][batch][2] for epoch in range(to_device_energy.shape[0]) for batch in range(to_device_energy.shape[1])]) for step, energy_data in energy_data_dict.items()}

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the mean energy consumption with error bars representing the standard deviation
    steps = list(mean_energy_consumption.keys())
    means = list(mean_energy_consumption.values())
    stds = list(std_energy_consumption.values())

    ax.bar(steps, means, yerr=stds, capsize=5, color=[step_colors[step] for step in steps])

    # Set the x-axis and y-axis labels and title
    ax.set_xlabel(f'Batch Steps in {modelname}', fontsize=12)
    ax.set_ylabel('Energy Consumption in Joules', fontsize=12)
    ax.set_title(f'Mean and Standard Deviation of Energy Consumption for Each Step in 1 Batch in {modelname}', fontsize=14)

    # Improve the layout and aesthetics
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'mean and std energy consumption of each step in 1 batch of {modelname}.png'))


# plot each step time consumption with all batch with the x axis set to each batch
def plot_step_time_distribution(to_device_energy, forward_energy, 
                                loss_energy, backward_energy, 
                                optimize_energy, step_colors,
                                modelname, plot_folder):
    batch_info = ['to_device', 'forward', 'loss', 'backward', 'optimize']
    energy_data_dict = {
        'to_device': to_device_energy,
        'forward': forward_energy,
        'loss': loss_energy,
        'backward': backward_energy,
        'optimize': optimize_energy
    }

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot time duration for each step
    for step in batch_info:
        energy_data = energy_data_dict[step]
        # Calculate time duration (end_time - start_time) for each batch
        time_durations = [energy_data[epoch][batch][1] - energy_data[epoch][batch][0] 
                         for epoch in range(energy_data.shape[0]) 
                         for batch in range(energy_data.shape[1])]
        
        # Plot with corresponding color from step_colors
        ax.plot(range(len(time_durations)), time_durations, 
                color=step_colors[step], label=step)

    ax.set_xlabel(f'Batch Steps (Total 5 Epochs) in {modelname}')
    ax.set_ylabel('Time Duration (seconds)')
    ax.set_title(f'Time Consumption per Step across Batches in {modelname}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'time consumption per step across batches in {modelname}.png'))


# plot each step time consumption with all batch with the x axis set to each step
def plot_step_time_distribution_box(to_device_energy, forward_energy, loss_energy, 
                                    backward_energy, optimize_energy, 
                                    step_colors, step_markers,
                                    modelname, plot_folder):
    # Create a dictionary to store time durations for each step
    step_times = {
        'to_device': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'optimize': []
    }
    
    # Calculate time durations for each step
    energy_data_dict = {
        'to_device': to_device_energy,
        'forward': forward_energy,
        'loss': loss_energy,
        'backward': backward_energy,
        'optimize': optimize_energy
    }
    
    for step, energy_data in energy_data_dict.items():
        time_durations = [energy_data[epoch][batch][1] - energy_data[epoch][batch][0] 
                         for epoch in range(energy_data.shape[0]) 
                         for batch in range(energy_data.shape[1])]
        step_times[step] = time_durations

    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot for each step
    for i, (step, times) in enumerate(step_times.items(), 1):
        plt.scatter([i] * len(times), times, color=step_colors[step], 
                   alpha=0.6, label=step, marker=step_markers[step])

    # Customize plot
    plt.xticks(range(1, len(step_times) + 1), list(step_times.keys()), rotation=45)
    plt.xlabel(f'Steps in {modelname}')
    plt.ylabel('Time Duration (seconds)')
    plt.title(f'Distribution of Time Consumption per Step in {modelname}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'time consumption distribution of each step across batches in {modelname}.png'))


# plot each step total energy data with all batch with the x axis set to each step in each epoch
# calculate the energy consumption for each step in each epoch
def cal_energy_epoch(to_device_energy, forward_energy, loss_energy, 
                     backward_energy, optimize_energy):
    energy_step_epoch = np.zeros((to_device_energy.shape[0], 5))
    for epoch in range(to_device_energy.shape[0]):
        for batch in range(to_device_energy.shape[1]):
            energy_step_epoch[epoch][0] += to_device_energy[epoch][batch][2]
            energy_step_epoch[epoch][1] += forward_energy[epoch][batch][2]
            energy_step_epoch[epoch][2] += loss_energy[epoch][batch][2]
            energy_step_epoch[epoch][3] += backward_energy[epoch][batch][2]
            energy_step_epoch[epoch][4] += optimize_energy[epoch][batch][2]
    return energy_step_epoch


def plot_epoch_step_energy(to_device_energy, forward_energy, 
                           loss_energy, backward_energy, 
                           optimize_energy, step_colors, modelname, plot_folder):
    energy_step_epoch = cal_energy_epoch(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
    # print(energy_step_epoch)
    step_info = ['to_device', 'forward', 'loss', 'backward', 'optimize']
    num_epochs = energy_step_epoch.shape[0]
    bar_width = 0.15  # Width of each bar

    # Use a color palette for epochs
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the energy consumption for each step in each epoch using bar chart
    for epoch in range(num_epochs):
        # Calculate the position of each bar
        bar_positions = np.arange(len(step_info)) + epoch * bar_width
        ax.bar(bar_positions, energy_step_epoch[epoch], width=bar_width, label=f'Epoch {epoch+1}', color=epoch_colors[epoch])

    # Set the x-axis labels to the step_info
    ax.set_xticks(np.arange(len(step_info)) + (num_epochs - 1) * bar_width / 2)
    ax.set_xticklabels(step_info)

    ax.set_xlabel(f'Steps in {modelname}')
    ax.set_ylabel('Total Energy Consumption in Joules')
    ax.legend()
    ax.set_title(f'Energy Consumption for Each Step in Each Epoch of {modelname}')
    # set a grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'epoch step energy consumption of {modelname}.png'))

    # Calculate mean and std across epochs for each step
    means = np.mean(energy_step_epoch, axis=0)
    stds = np.std(energy_step_epoch, axis=0)

    # Create new figure for mean/std plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with error bars
    ax.bar(step_info, means, yerr=stds, capsize=5, color=[step_colors[step] for step in step_info])

    ax.set_xlabel(f'Steps in {modelname}', fontsize=12)
    ax.set_ylabel('Energy Consumption (Joules)', fontsize=12)
    ax.set_title(f'Mean and Standard Deviation of Energy Consumption Across Epochs of {modelname}', fontsize=14)

    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'mean and std energy consumption of each step in each epoch of {modelname}.png'))

    # return the means and stds
    return means, stds


# plot each step total time data with all batch with the x axis set to each step in each epoch
def cal_time_epoch(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy):
    time_step_epoch = np.zeros((to_device_energy.shape[0], 5))
    for epoch in range(to_device_energy.shape[0]):
        for batch in range(to_device_energy.shape[1]):
            # Calculate duration for each step (end_time - start_time)
            time_step_epoch[epoch][0] += to_device_energy[epoch][batch][1] - to_device_energy[epoch][batch][0]
            time_step_epoch[epoch][1] += forward_energy[epoch][batch][1] - forward_energy[epoch][batch][0]
            time_step_epoch[epoch][2] += loss_energy[epoch][batch][1] - loss_energy[epoch][batch][0]
            time_step_epoch[epoch][3] += backward_energy[epoch][batch][1] - backward_energy[epoch][batch][0]
            time_step_epoch[epoch][4] += optimize_energy[epoch][batch][1] - optimize_energy[epoch][batch][0]
    return time_step_epoch

def plot_epoch_step_time(to_device_energy, forward_energy, 
                         loss_energy, backward_energy, 
                         optimize_energy, step_colors, modelname, plot_folder):
    time_step_epoch = cal_time_epoch(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
    # print(time_step_epoch)
    step_info = ['to_device', 'forward', 'loss', 'backward', 'optimize']
    num_epochs = time_step_epoch.shape[0]
    bar_width = 0.15

    # Use a color palette for epochs
    epoch_colors = plt.cm.viridis(np.linspace(0, 1, num_epochs))

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the time consumption for each step in each epoch using bar chart
    for epoch in range(num_epochs):
        bar_positions = np.arange(len(step_info)) + epoch * bar_width
        ax.bar(bar_positions, time_step_epoch[epoch], width=bar_width, label=f'Epoch {epoch+1}', color=epoch_colors[epoch])

    ax.set_xticks(np.arange(len(step_info)) + (num_epochs - 1) * bar_width / 2)
    ax.set_xticklabels(step_info)

    ax.set_xlabel(f'Steps in {modelname}')
    ax.set_ylabel('Time Consumption in Seconds')
    ax.set_title(f'Time Consumption for Each Step in Each Epoch of {modelname}')
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'epoch step time consumption of {modelname}.png'))

    # Calculate mean and std across epochs for each step
    means = np.mean(time_step_epoch, axis=0)
    stds = np.std(time_step_epoch, axis=0)

    # Create new figure for mean/std plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars with error bars
    ax.bar(step_info, means, yerr=stds, capsize=5, color=[step_colors[step] for step in step_info])

    ax.set_xlabel(f'Steps in {modelname}', fontsize=12)
    ax.set_ylabel('Time Consumption (Seconds)', fontsize=12)
    ax.set_title(f'Mean and Standard Deviation of Time Consumption Across Epochs of {modelname}', fontsize=14)

    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    # save the figure
    plt.savefig(os.path.join(plot_folder, f'mean and std time consumption of each step in each epoch of {modelname}.png'))


# # plot with different models
# def plot_model(modelname, model_data_folder_list, plot_folder):
#     model_data_path = [model_data_folder_list[i] for i in range(len(model_data_folder_list)) if f'{modelname}' in model_data_folder_list[i]][0]
#     energy_data, labeled_energy_data, to_device, forward, loss, backward, optimize, \
#     to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy = load_data(model_data_path)

#     plot_energy_data(labeled_energy_data,)
#     plot_period_energy_data(labeled_energy_data)
#     plot_batch_step_energy(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_step_energy_distribution(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_step_energy_distribution_bar(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_epoch_step_energy(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_step_time_distribution(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_step_time_distribution_box(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)
#     plot_epoch_step_time(to_device_energy, forward_energy, loss_energy, backward_energy, optimize_energy)