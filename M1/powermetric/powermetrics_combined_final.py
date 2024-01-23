'''this code is devided into 2 parts:
1. calculate the idle power consumption of the system, 
    to get this goal, first we need to run the powermetrics for 30 seconds in idle situation,
    and then calculate the average power consumption of the system per second.
2. calculate the power consumption of the system when the system is running the model,
    and the output contains the total energy consumption and the running time of the model.
'''
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor(),)

# Download test data from open datasets.
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor(),)

# set the batch size
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
print(model)
    

# optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def run_powermetrics(file_path):
    """
    Run powermetrics and retrieve the output.

    :param interval: Sampling interval in milliseconds.
    :param count: Number of samples to retrieve.s
    :return: The output from powermetrics.
    """

    # Define the command as a list of arguments
    cmd = ["sudo", "powermetrics",  "-i", "1000", "--samplers", "cpu_power,gpu_power", "-a", "1", "-o", file_path]
    
    process = subprocess.Popen(cmd)
    return process

def txt_data_process(file_path):
    """
    Read the output file of powermetric and extract the power value

    :param file_path: The path of the output file of powermetric.
    :return: The list of power values.
    """

    list_power = []
    with open(file_path, 'r') as f:
        for line in f:
            if 'Combined Power' in line:
                power_value = line.split(':')[1].strip()
                print(power_value)

                # Remove the unit
                power_value = power_value.replace('mW', '')

                # Convert to integer
                power_value = int(power_value)
                list_power.append(power_value)

    print(list_power)
    print(len(list_power))

    # do the data process
    '''
    The data from list_power is the Conbined Power of each second.
    The data is the Power of each second.
    we need to calculate the energy consumption of the whole process.
    and need to change the J to kWh.
    '''
    # calculate the energy consumption
    energy_consumption = 0
    for i in range(len(list_power)):
       energy_consumption += list_power[i]
    print(energy_consumption)

    # change the mW to W
    energy_consumption = energy_consumption / 1000

    # calculate the energy consumption, the interval is 1 second, and the energy unit is J
    energy_consumption = energy_consumption * 1
    
    # change the J to kWh
    energy_consumption = energy_consumption / 3600000
    print(energy_consumption)
    
    return energy_consumption, list_power

def calculate_idle(file_path_idle):
    # run powermetrics
    idle_process = run_powermetrics(file_path_idle)

    # wait for 30 seconds
    time.sleep(30)

    # kill the powermetrics process
    idle_process.terminate()
    idle_process.wait()

    # calculate the average power consumptions
    idle_energy_consumption, idle_list = txt_data_process(file_path_idle)
    avg_idel_energy_consumption = idle_energy_consumption / len(idle_list)
    print('The average idle energy consuption per second is ', avg_idel_energy_consumption, 'kWh')

    return avg_idel_energy_consumption, idle_list

def calculate_model(file_path_run):
    # Create the file name by appending the time string to the file path
    # print(file_path_run)

    # import the time lib to calculate the running time
    time_start = time.time()

    # run powermetrics
    powermetrics_process = run_powermetrics(file_path_run)

    epochs = 50
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    powermetrics_process.terminate()
    powermetrics_process.wait()

    time_end = time.time()

    # calculate the total running time
    time_total = time_end - time_start
    print("The total running time is: ", time_total)

    # process the data
    energy_consumption, power_list_model = txt_data_process(file_path_run)
    total_training_time = len(power_list_model)

    return energy_consumption, total_training_time, power_list_model

def main():
    '''
    1. set a 30s timer to calculate the idle power consumption of the system
    '''
    # create the file name: powermetric/pm_idle.txt
    file_path_idle = "powermetric/pm_idle_final.txt"

    # calculate the idle power consumption of the system
    avg_idel_energy_consumption, idle_list = calculate_idle(file_path_idle)

    '''
    2. calculate the power consumption of the system when the system is running the model
    '''
    # Create the file name by appending the time string to the file path
    file_path_run = "powermetric/pm_calculate_final.txt"
    # print(file_path_run)
    
    energy_consumption, total_training_time, power_list_model = calculate_model(file_path_run)

    '''
    3. Approximate estimate of the energy consumed in computing the model
    '''
    # calculate the energy consumption of the model
    energy_consumption_model = energy_consumption - avg_idel_energy_consumption * total_training_time
    system_energy_consumption = avg_idel_energy_consumption * total_training_time

    # calculate the average power during running the model
    model_power = 0
    for i in range(len(power_list_model)):
        model_power += power_list_model[i]
    avg_total_power = model_power / len(power_list_model)/ 1000

    # calculate the average power during running the system
    system_power = 0
    for i in range(len(idle_list)):
        system_power += idle_list[i]
    avg_system_power = system_power / len(idle_list) / 1000

    avg_model_power = avg_total_power - avg_system_power

    print('The average total power during running the model is: ', avg_total_power, 'W')
    print('The average power during running the system is: ', avg_system_power, 'W')
    print('The average power during running the model is: ', avg_model_power, 'W')
    print('*' * 50)
    print('The total training time is: ', total_training_time, 's')
    print('*' * 50)
    print('The total energy consumption is: ', energy_consumption, 'kWh')
    # print('The average idle energy consuption per second is ', avg_idel_energy_consumption, 'kWh') 
    print('The total system energy consumption during running the model is: ', system_energy_consumption, 'kWh')
    print('The total energy consumption of the model is: ', energy_consumption_model, 'kWh')

if __name__ == "__main__":
    main()