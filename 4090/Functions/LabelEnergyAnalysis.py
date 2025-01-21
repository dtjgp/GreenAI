'''
This code is to analysis the raw data captured during the training process of the models.
'''

import numba as nb
import numpy as np
import pandas as pd

@nb.jit(nopython=True)
def interpolate_point(times, powers, target_time):
    n = len(times)
    if n == 0:
        return 0.0
    if target_time <= times[0]:
        return powers[0]
    if target_time >= times[-1]:
        return powers[-1]
    
    # Binary search
    left, right = 0, n-1
    while left <= right:
        mid = (left + right) // 2
        if times[mid] == target_time:
            return powers[mid]
        elif times[mid] < target_time:
            left = mid + 1
        else:
            right = mid - 1
            
    # Linear interpolation
    pos = left
    t1, p1 = times[pos-1], powers[pos-1]
    t2, p2 = times[pos], powers[pos]
    ratio = (target_time - t1) / (t2 - t1)
    return p1 + (p2 - p1) * ratio

@nb.jit(nopython=True)
def integrate_power_over_interval(samples, start_time, end_time):
    times = samples[:, 0]
    powers = samples[:, 1]
    
    # Get start and end powers through interpolation
    start_power = interpolate_point(times, powers, start_time)
    end_power = interpolate_point(times, powers, end_time)
    
    # Filter points within interval
    mask = (times >= start_time) & (times <= end_time)
    interval_times = times[mask]
    interval_powers = powers[mask]
    
    # Create array including boundary points
    n_points = len(interval_times)
    full_times = np.zeros(n_points + 2)
    full_powers = np.zeros(n_points + 2)
    
    # Add boundary points
    full_times[0] = start_time
    full_powers[0] = start_power
    full_times[-1] = end_time
    full_powers[-1] = end_power
    
    # Add interior points
    if n_points > 0:
        full_times[1:-1] = interval_times
        full_powers[1:-1] = interval_powers
    
    # Integration using trapezoidal rule
    total_energy = 0.0
    for i in range(len(full_times)-1):
        dt = full_times[i+1] - full_times[i]
        avg_p = (full_powers[i] + full_powers[i+1]) / 2.0
        total_energy += avg_p * dt
        
    return total_energy

def label_energy_consumption(energy_data, to_device, forward, loss, backward, optimize):
    # Create a copy of the energy_data dataframe to avoid modifying the original
    labeled_energy_data = energy_data.copy()
    
    # Initialize a new column for the step labels
    labeled_energy_data['step'] = 'idle'
    
    # Define a helper function to label the steps
    def label_steps(energy_data, step_energy, step_name):
        for epoch in range(step_energy.shape[0]):
            for batch in range(step_energy.shape[1]):
                start_time = step_energy[epoch][batch][0]
                end_time = step_energy[epoch][batch][1]
                mask = (energy_data['timestamp'] >= start_time) & (energy_data['timestamp'] <= end_time)
                labeled_energy_data.loc[mask, 'step'] = step_name
    
    # Label each step
    label_steps(labeled_energy_data, to_device, 'to_device')
    label_steps(labeled_energy_data, forward, 'forward')
    label_steps(labeled_energy_data, loss, 'loss')
    label_steps(labeled_energy_data, backward, 'backward')
    label_steps(labeled_energy_data, optimize, 'optimize')
    
    return labeled_energy_data

# transfer the numpy files to pandas dataframe
# the shape of the ndarray is (5,469,2), 5 is the number of epochs, 469 is the number of batches, 2 is the start and end time
def transfer_type(npfiles):
    # Initialize an empty DataFrame with epochs as columns and batch indices as rows
    result_df = pd.DataFrame(columns=[f'epoch_{i}' for i in range(npfiles.shape[0])], 
                            index=range(npfiles.shape[1]))
    # print(f'the shape of the result_df is {result_df.shape}')

    # Fill the DataFrame 
    for epoch in range(npfiles.shape[0]):
        for batch in range(npfiles.shape[1]):
            result_df.iloc[batch, epoch] = [npfiles[epoch, batch, 0], npfiles[epoch, batch, 1]]

    return result_df