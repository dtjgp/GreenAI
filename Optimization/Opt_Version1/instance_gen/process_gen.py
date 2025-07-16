import random
import numpy as np

def generate_instance():
    operations_info = {}
    # Each tuple in the list corresponds to (num_operations, power_per_operation)
    processes_details = [(32, 100), (28, 125), (26, 150), (25, 175), (24, 200), (23, 225), (22, 250)]
    # processes_details = [(16, 100), (15, 125), (14, 150), (13, 175), (12, 200), (11, 225), (10, 250)]
    for i, (num_operations, power) in enumerate(processes_details):
        # Set every operation's duration to 1 and power to the specified power
        operations = [{'power': power, 'duration': 1} for _ in range(num_operations)]
        # The maximum time for completion is set to the number of operations
        max_time = num_operations
        operations_info[i] = {'operations': operations, 'max_time': max_time}
    
    return operations_info