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
def train_layers(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m): # 初始化权重
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # set a list of layer name
    LayerList = ['Conv2d','ReLU','MaxPool2d','Linear','Dropout','Flatten'] 
    # save each epoch's time and energy
    Time_Layers = np.zeros((num_epochs, len(LayerList), 1)) # save the forward time of each layer in the epoch
    Time_AllEpochs = np.zeros((num_epochs, 6, 1))  # save each part of the model's time and energy in the epoch
    TrainTime, TestAcc, TrainAcc, TrainLoss, TimeEpoch = [], [], [], [], [] # 用于存储每一轮的时间和能耗
    Energy_AllEpochs = np.zeros((num_epochs, 1), dtype='object') # save each part of the model's energy in the epoch
    Timport = [] # import data to ndarray
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
        # start the nvidia-smi command
        with open('gpu_power_usage.csv', 'w') as file:
            # Start the nvidia-smi command
            nvidia_smi_process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv", "--loop-ms=1000"],
                stdout=file,  # Redirect the output directly to the file
                stderr=subprocess.PIPE,
                text=True)
        net.train() # 设置为训练模式
        # store the time and energy of the current epoch
        TtD_epoch, Tforward_epoch, Tloss_epoch, Tback_epoch, Topt_epoch, Ttrain_epoch = 0,0,0,0,0,0 # to device time, forward time, loss cal time, backward time, optimizer time, epoch time
        TrainLoss_epoch, TrainAcc_epoch = [], [] # 用于存储每一轮的loss和acc
        TforwardLayer_epoch = np.zeros((len(LayerList), 1)) # time forward each layer in each epoch
        Timport_epoch = 0 # import data to ndarray in each epoch
        # set the metric using d2l.Accumulator model
        metric = d2l.Accumulator(3)
        for i, (X, y) in enumerate(train_iter):  # for each batch
            print('round %d' % (i))
            Tbatch_start = time.time() # time batch start
            optimizer.zero_grad() # 将optimizer的梯度清零
            TforwardLayer_batch = np.zeros((len(LayerList), 1)) # time forward each layer in each batch
            Tforward_batch = 0 # time forward in each batch
            # time to device in each batch
            TtD_batch = 0 # time to device i
            TtD = time.time()
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            TtD_end = time.time()
            TtD_batch = TtD_end - TtD
            print('time to device %f sec' % (TtD_batch))
            TtD_epoch += TtD_batch
            # 将原本的模型进行修改，使得能够逐层进行运行，并且在这个过程中，记录下时间和能量
            y_hat = X
            for layer in net:
                Tlayer_i = 0
                layer_name = layer.__class__.__name__ # 获取层的名字
                # find out the layer name is in where of the list
                layer_index = LayerList.index(layer_name)
                # calculate the time of each layer
                Tlayeri = time.time()
                y_hat = layer(y_hat)
                Tlayeri_end = time.time()
                Tlayer_i = Tlayeri_end - Tlayeri
                TforwardLayer_batch[layer_index,0] += Tlayer_i
                Timporti = time.time()
                Timporti_end = time.time()
                Tlayer_importi = Timporti_end - Timporti
                Timport_epoch += Tlayer_importi
                # if torch.isinf(y_hat).any() or torch.isnan(y_hat).any():
                #     print("Inf or NaN detected in y_hat")
                #     break
            # 计算前向的时间
            Tforward_batch = np.sum(TforwardLayer_batch)
            print('time forward %f sec' % (Tforward_batch))
            TforwardLayer_epoch += TforwardLayer_batch # save the forward time of each layer to the epoch
            Tforward_epoch += Tforward_batch # save the total forward time of each epoch
        ##################################################################################
            # 计算loss
            Tloss_batch = 0 # 初始化loss的时间
            Tli = time.time()
            loss = loss_fn(y_hat, y)
            Tli_end = time.time()
            Tloss_batch = Tli_end - Tli
            print('loss time %f sec' % (Tloss_batch))
            Tloss_epoch += Tloss_batch
        ##################################################################################
            # 计算backward
            Tback_batch = 0 # 初始化backward的时间
            Tbi = time.time()
            loss.backward()
            Tbi_end = time.time()
            Tback_batch = Tbi_end - Tbi
            print('backward time %f sec' % (Tback_batch))
            Tback_epoch += Tback_batch
        ##################################################################################
            # 计算optimizer
            Topt_batch = 0 # 初始化optimizer的时间
            Toi = time.time()
            optimizer.step()
            Toi_end = time.time()
            Topt_batch = Toi_end - Toi
            print('optimizer time %f sec' % (Topt_batch))
            Topt_epoch += Topt_batch
        ##################################################################################
            Tbatch_stop = time.time()
            Ttrain_batch = Tbatch_stop - Tbatch_start
            print(f'training time in batch {i} cost {Ttrain_batch} sec')
            Ttrain_epoch += Ttrain_batch
        ##################################################################################
            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            TrainLoss_batch = metric[0] / metric[2]
            TrainAcc_batch = metric[1] / metric[2]
            TrainLoss_epoch.append(TrainLoss_batch)
            TrainAcc_epoch.append(TrainAcc_batch)
            print('loss %f, train acc %f' % (TrainLoss_batch, TrainAcc_batch))
        ##################################################################################
        Time_Layers[epoch,:] = TforwardLayer_epoch # save the forward time of each layer to the epoch
        Timport.append(Timport_epoch) # 将每一轮的导入数据的时间加入到Timport中
        TrainTime.append(Ttrain_epoch) # 将每一轮的训练时间加入到TrainTime中
        TrainLoss.append(TrainLoss_epoch) # 将每一轮的loss加入到TrainLoss中
        TrainAcc.append(TrainAcc_epoch) # 将每一轮的train acc加入到TrainAcc中
        ##################################################################################
        # 进行模型的test部分运行
        Ttest_epoch = 0
        Tti = time.time()
        TestAcc_epoch = d2l.evaluate_accuracy_gpu(net, test_iter, device)
        Tti_end = time.time()
        Ttest_epoch = Tti_end - Tti
        print('test acc is %f' % (TestAcc_epoch))
        TestAcc.append(TestAcc_epoch)
        ##################################################################################
        # 将每一轮的每个部分的时间加入到time_data_round中
        Time_AllEpochs[epoch,:,0] = TtD_epoch, Tforward_epoch, Tloss_epoch, Tback_epoch, Topt_epoch, Ttest_epoch
        ##################################################################################
        # stop the nvidia-smi command
        nvidia_smi_process.terminate()
        timer.stop() # 停止计时 
        TimeEpoch.append(timer.sum())
        ##############################################################################################################
        # calculate the energy consumption of each epoch
        GPU_df = pd.read_csv('gpu_power_usage.csv')
        for row in range(len(GPU_df)):
            GPU_df.iloc[row,0] = GPU_df.iloc[row,0].replace(' W','')
        Consumption_df = GPU_df.astype(float)  
        EnergyDatai = Consumption_df.iloc[:,0].values # 将数据转换为numpy数组
        Energy_AllEpochs[epoch,0] = EnergyDatai
        # print('epoch %d, time %f sec' % (epoch, timer.sum()))
    return Time_Layers, Time_AllEpochs, TestAcc, TrainLoss, TrainAcc, TimeEpoch, Energy_AllEpochs, TrainTime, Timport