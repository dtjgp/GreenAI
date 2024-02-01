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
    LayerList = ['Conv2d','ReLU','MaxPool2d','Linear','Dropout','Flatten'] 
    # save each epoch's time and energy
    Time_AllEpochs = np.zeros((num_epochs, 6, 1))  # save each part of the model's time and energy in the epoch
    TrainTime, TestAcc, TrainAcc, TrainLoss, TimeEpoch = [], [], [], [], [] # 用于存储每一轮的时间和能耗
    TTrainAccLoss = [] # 用于存储计算acc和loss的时间 for each epoch
    Energy_AllEpochs = np.zeros((num_epochs, 1), dtype='object') # save each part of the model's energy in the epoch
    # print the training device
    print('training on', device)
    net.to(device) # 将模型放到对应的设备上
    # 初始化optimizer和loss
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    # 初始化计时器
    timer, num_batches = d2l.Timer(), len(train_iter) 
    # print('num of batches is %d' % (num_batches)) # 打印batch的数量which is 469
    # 开始训练
    for epoch in range(num_epochs):
        print('epoch %d' % (epoch + 1))
        correct = 0 
        total = 0
        # each epoch, set a timer to record the time
        timer.start()
        net.train() # 设置为训练模式
        # store the time and energy of the current epoch
        TtD_epoch, Tforward_epoch, Tloss_epoch, Tback_epoch, Topt_epoch, Ttrain_epoch = 0,0,0,0,0,0 # to device time, forward time, loss cal time, backward time, optimizer time, epoch time
        TrainLoss_epoch, TrainAcc_epoch = [], [] # 用于存储每一轮的loss和acc
        TTrainAccLoss_epoch = 0 # time to calculate the acc and loss in each epoch
        running_loss = 0.0  # 记录每个epoch的loss之和
        # set the metric using d2l.Accumulator model
        metric = d2l.Accumulator(3)
        # start the nvidia-smi command
        with open('gpu_power_usage.csv', 'w') as file:
            # Start the nvidia-smi command
            nvidia_smi_process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv", "--loop-ms=1000"],
                stdout=file,  # Redirect the output directly to the file
                stderr=subprocess.PIPE,
                text=True)
        for i, (X, y) in enumerate(train_iter):  # for each batch
            # print('round %d' % (i))
            total += y.size(0)
            optimizer.zero_grad() # 将optimizer的梯度清零
            Tforward_batch = 0 # time forward in each batch
            TtD_batch = 0 # time to device i
            # print('batch size is %d' % (X.shape[0]))
            torch.cuda.synchronize()  # 等待数据传输完成
            Tbatch_start = time.time() # time batch start
            # Ttesti = time.time()
            X,y = X.to(device), y.to(device)
            torch.cuda.synchronize()  # 等待数据传输完成
            TtD_end = time.time()
            TtD_batch = TtD_end - Tbatch_start
            # print('time to device %f sec' % (TtD_batch))
            TtD_epoch += TtD_batch
        ##################################################################################
            Tforward_batch = 0 # 初始化forward的时间
            y_hat = net(X)
            # predicted = y_hat.argmax(dim=1)
            # correct += (predicted == y).sum().item()
            torch.cuda.synchronize()  # 等待数据传输完成
            Tfi_end = time.time()
            Tforward_batch = Tfi_end - TtD_end
            # print('time forward %f sec' % (Tforward_batch))
            Tforward_epoch += Tforward_batch # save the total forward time of each epoch
        ##################################################################################
            loss = loss_fn(y_hat, y)
            torch.cuda.synchronize()  # 等待数据传输完成
            # running_loss += loss.item()
            Tli_end = time.time()
            Tloss_batch = Tli_end - Tfi_end
            # print('loss time %f sec' % (Tloss_batch))
            Tloss_epoch += Tloss_batch
        ##################################################################################
            loss.backward()
            torch.cuda.synchronize()  # 等待数据传输完成
            Tbi_end = time.time()
            Tback_batch = Tbi_end - Tli_end
            # print('backward time %f sec' % (Tback_batch))
            Tback_epoch += Tback_batch
        ##################################################################################
            optimizer.step()
            torch.cuda.synchronize()  # 等待数据传输完成
            Toi_end = time.time()
            Topt_batch = Toi_end - Tbi_end
            # print('optimizer time %f sec' % (Topt_batch))
            Topt_epoch += Topt_batch
            # Ttesti_end = time.time()
            # Ttesti_batch = Ttesti_end - Ttesti
            # TTrainAccLoss_epoch += Ttesti_batch
        ##################################################################################
            Tbatch_stop = time.time()
            Ttrain_batch = Tbatch_stop - Tbatch_start
            # print(f'training time in batch {i} cost {Ttrain_batch} sec')
            Ttrain_epoch += Ttrain_batch
        ##################################################################################
            Tci = time.time()
            # 计算acc和loss
            # running_loss += loss.item()
            # predicted = y_hat.argmax(dim=1)
            # total += y.size(0)
            # correct += (predicted == y).sum().item()
            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            TrainLoss_batch = metric[0] / metric[2]
            TrainAcc_batch = metric[1] / metric[2]
            torch.cuda.synchronize()  # 等待数据传输完成
            Tci_end = time.time()
            TCalAccLoss_batch = Tci_end - Tci
            # print(f'calculating time in batch {i} cost {TCalAccLoss_batch} sec')
            TTrainAccLoss_epoch += TCalAccLoss_batch
        ##################################################################################
            # TCalAccLoss_batch = 0 # 初始化计算acc和loss的时间
            # Tci = time.time()
            # with torch.no_grad():
            #     metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            # TrainLoss_batch = metric[0] / metric[2]
            # TrainAcc_batch = metric[1] / metric[2]
            # TrainLoss_epoch.append(TrainLoss_batch)
            # TrainAcc_epoch.append(TrainAcc_batch)
            # Tci_end = time.time()
            # TCalAccLoss_batch = Tci_end - Tci
            # TCalAccLoss_epoch += TCalAccLoss_batch
            # print('loss %f, train acc %f' % (TrainLoss_batch, TrainAcc_batch))
        ##################################################################################
        # epoch_loss = running_loss / len(train_iter)
        # epoch_acc = correct / total
        # print(f'Epoch {epoch+1} completed: Avg Loss: {epoch_loss}, Avg Accuracy: {epoch_acc}')
        # TrainLoss_epoch.append(epoch_loss)
        # TrainAcc_epoch.append(epoch_acc)
        print(f'Epoch {epoch+1} completed: Avg Loss: {TrainLoss_batch}, Avg Accuracy: {TrainAcc_batch}')
        TrainLoss_epoch.append(TrainLoss_batch)
        TrainAcc_epoch.append(TrainAcc_batch)
        TrainTime.append(Ttrain_epoch) # 将每一轮的训练时间加入到TrainTime中
        TrainLoss.append(TrainLoss_epoch) # 将每一轮的loss加入到TrainLoss中
        TrainAcc.append(TrainAcc_epoch) # 将每一轮的train acc加入到TrainAcc中
        TTrainAccLoss.append(TTrainAccLoss_epoch) # 将每一轮计算acc和loss的时间加入到TTrainAccLoss中
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
        print('epoch %d, time %f sec' % (epoch+1, timer.sum()))
    return Time_AllEpochs, TestAcc, TrainLoss, TrainAcc, TimeEpoch, Energy_AllEpochs, TrainTime, TTrainAccLoss