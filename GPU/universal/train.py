# Function: train the model
import torch
from torch import nn
from d2l import torch as d2l
import numpy as np
import time
import subprocess
import pandas as pd

'''
对train_ch6()函数进行修改，使得能够在每一层的前向传播的时候，记录下时间, 
能耗部分不好进行具体统计，因为每个层的能耗有区别，并且时间过短，所以采用的方式为计算总能耗和总时长，最后通过平均值来计算估算的能耗
'''
def train_func(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m): # 初始化权重
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # set a list of layer name
    list_layer_name = ['Conv2d','ReLU','MaxPool2d','Linear','Dropout','Flatten'] # 该模型中包括的所有的层的名字
    # save each epoch's data, including time and energy
    time_energy_data_forward = np.zeros((num_epochs, len(list_layer_name), 2)) # 用于存储每一层的前向传播的时间和能耗
    time_energy_data_round = np.zeros((num_epochs, 6, 2))  # 用于存储每一轮的时间和能耗
    time_train = []  # 用于存储每一轮的训练时间
    acc_data = []   # 用于存储每一轮的test 
    train_l = []    # 用于存储每一轮的loss
    train_acc = []  # 用于存储每一轮的train准确率
    time_data_epoch = []   # 用于存储每一轮的时间
    energy_data_epoch = [] # 用于存储每一轮的能耗
    
    # print the training device
    print('training on', device)
    net.to(device) # 将模型放到对应的设备上
    # 初始化optimizer和loss
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # 初始化计时器
    timer, num_batches = d2l.Timer(), len(train_iter)   
    # 开始训练
    for epoch in range(num_epochs):
        print('epoch %d' % (epoch + 1))
        # each epoch, set a timer to record the time
        timer.start()
        ##############################################################################################################
        # start the nvidia-smi command
        with open('gpu_power_usage.csv', 'w') as file:
            # Start the nvidia-smi command
            nvidia_smi_process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv", "--loop-ms=1000"],
                stdout=file,  # Redirect the output directly to the file
                stderr=subprocess.PIPE,
                text=True
            )
        ##############################################################################################################
        net.train() # 设置为训练模式
        # 初始化每个epoch的统计时间的变量
        ttd_epoch, time_forward, time_loss = 0,0,0 # time to device, time forward, time cost loss of each epoch
        time_backward, time_optimizer, Ttrain_epoch = 0,0,0 # time cost backward, time cost optimizer, time cost test acc of each epoch
        train_l_epoch = []
        train_acc_epoch = []
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            print('round %d' % (i))
            time_round_start = time.time() # 计算每一轮的时间s
            optimizer.zero_grad() # 将optimizer的梯度清零s
            time_forward_epoch = np.zeros((len(list_layer_name), 1))
        ##################################################################################
            # 计算将数据放到对应的设备上的时间
            ttdi = 0 # time to device i
            time_to_device = time.time()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            time_to_device_end = time.time()
            ttdi = time_to_device_end - time_to_device
            print('time to device %f sec' % (ttdi))
            ttd_epoch += ttdi
        ##################################################################################
            # 将原本的模型进行修改，使得能够逐层进行运行，并且在这个过程中，记录下时间和能量
            y_hat = X
            for layer in net:
                time_cost_layer = 0
                layer_name = layer.__class__.__name__ # 获取层的名字
                # find out the layer name is in where of the list
                layer_index = list_layer_name.index(layer_name)
                # calculate the time
                time_start_layer = time.time()
                y_hat = layer(y_hat)
                time_end_layer = time.time()
                time_cost_layer = time_end_layer - time_start_layer
                time_forward_epoch[layer_index,0] += time_cost_layer
                if torch.isinf(y_hat).any() or torch.isnan(y_hat).any():
                    print("Inf or NaN detected in y_hat")
            time_energy_data_forward[epoch,:,0] += time_forward_epoch
            # 计算前向的时间
            time_forward = np.sum(time_energy_data_forward[epoch,:,0])
            print('time forward of the %d epoch is: %f sec' % (epoch, time_forward))
            # 计算loss
            tli = 0 # 初始化loss的时间
            time_start_loss = time.time()
            loss = loss_fn(y_hat, y)
            time_end_loss = time.time()
            tli = time_end_loss - time_start_loss
            print('loss time of the %d epoch is: %f sec' % (epoch, tli))
            time_loss += tli
            # 计算backward
            tbi = 0 # 初始化backward的时间
            time_start_backward = time.time()
            loss.backward()
            time_end_backward = time.time()
            tbi = time_end_backward - time_start_backward
            print('backward time of the %d epoch is: %f sec' % (epoch, tbi))
            time_backward += tbi
            # 计算optimizer
            toi = 0 # 初始化optimizer的时间
            time_start_optimizer = time.time()
            optimizer.step()
            time_end_optimizer = time.time()
            toi = time_end_optimizer - time_start_optimizer
            print('optimizer time of the %d epoch is: %f sec' % (epoch, toi))
            time_optimizer += toi
        ##################################################################################
            time_round_end = time.time()
            Ttrain_i = time_round_end - time_round_start
            print(f'training time in round {i} cost {Ttrain_i} sec')
            Ttrain_epoch += Ttrain_i
        ##################################################################################
            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_l_i = metric[0] / metric[2]
            train_acc_i = metric[1] / metric[2]
            train_l_epoch.append(train_l_i)
            train_acc_epoch.append(train_acc_i)
            print('loss %f, train acc %f' % (train_l_i, train_acc_i))
        ##################################################################################
        time_train.append(Ttrain_epoch) # 将每一轮的训练时间加入到time_train中
        train_l.append(train_l_epoch) # 将每一轮的loss加入到train_l中
        train_acc.append(train_acc_epoch) # 将每一轮的train acc加入到train_acc中
        ##################################################################################
        # 进行模型的test部分运行
        Ttest_epoch = 0
        time_test_acc_start = time.time()
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter, device)
        time_test_acc_end = time.time()
        Ttest_epoch = time_test_acc_end - time_test_acc_start
        print('test acc is %f' % (test_acc))
        acc_data.append(test_acc)
        ##################################################################################
        # 将每一轮的每个部分的时间加入到time_data_round中
        time_energy_data_round[epoch,:,0] = ttd_epoch, time_forward, time_loss, time_backward, time_optimizer, Ttest_epoch
        ##################################################################################
        # stop the nvidia-smi command
        nvidia_smi_process.terminate()
        timer.stop() # 停止计时 
        time_data_epoch[epoch] = timer.sum()
        ##############################################################################################################
        # calculate the energy consumption of each epoch
        energy_datal = pd.read_csv('gpu_power_usage.csv')
        for l in range(len(energy_datal)):
            energy_datal.iloc[l,0] = energy_datal.iloc[l,0].replace(' W','')
        energy_datal = energy_datal.astype(float)  
        energy_datal = energy_datal.sum()
        energy_data_epoch[epoch] = energy_datal
        # print('epoch %d, time %f sec' % (epoch, timer.sum()))
    return time_energy_data_forward, time_energy_data_round, acc_data, train_l, train_acc, time_data_epoch, energy_data_epoch