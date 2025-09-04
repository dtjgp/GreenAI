'''
This code is to use different functions to provide analysis of the model.
'''

from ptflops import get_model_complexity_info
import torch
import torchvision.transforms as transforms
import torchvision
import time
import pynvml
import threading
import subprocess
import queue
from d2l import torch as d2l
import numpy as np
import os 
from torch import nn
import pandas as pd

'''Get the model MACs number'''
def get_model_info(model, img_channel, num_labels):
    net = model(img_channel, num_labels)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (img_channel, 224, 224), as_strings=True,
                                                print_per_layer_stat=True, verbose=True)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    return net, macs, params

'''Interpolate method'''
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


'''Load the data'''
# load the data
# fashion mnist
def get_dataloader_workers():
    """Use 4 processes to read the data.

    Defined in :numref:`sec_utils`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集, 然后将其加载到内存中

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def load_data_cifar100(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # import the cifar100 dataset
    cifar_train = torchvision.datasets.CIFAR100(
        root="../data", train=True, transform=trans, download=True)
    cifar_test = torchvision.datasets.CIFAR100(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(cifar_train, batch_size, shuffle=True,
                                        num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(cifar_test, batch_size, shuffle=False,
                                        num_workers=get_dataloader_workers()))
    
def load_data_cifar10(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    # import the cifar100 dataset
    cifar_train = torchvision.datasets.CIFAR10(
        root="../data", train=True, transform=trans, download=True)
    cifar_test = torchvision.datasets.CIFAR10(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(cifar_train, batch_size, shuffle=True,
                                        num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(cifar_test, batch_size, shuffle=False,
                                        num_workers=get_dataloader_workers()))

'''pynvml sampling function'''
def nvml_sampling_thread(handle, filename, stop_event, sampling_interval):
    """
    在单独的线程中定期调用 NVML, 获取功耗数据并存储到 data_queue 中。
    参数：
    - handle: nvmlDeviceGetHandleByIndex(0) 得到的 GPU 句柄
    - data_queue: 用于存放 (timestamp, power_in_watts) 数据的队列
    - stop_event: 当此事件被设置时，线程应结束循环
    - sampling_interval: 采样间隔（秒）
    """
    with open(filename/'energy_consumption_file.csv', 'a') as f:  # 追加模式
        # 写入列名
        f.write("timestamp,power_in_watts,sm_clock\n")
        while not stop_event.is_set():
            try:
                # 采集功率和时间戳
                current_time = time.time()
                current_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # 转换 mW -> W
                sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)             # SM 频率
                # 写入文件
                f.write(f"{current_time}, {current_power}, {sm_clock}\n")
                # 等待下一次采样
                time.sleep(sampling_interval)
            except pynvml.NVMLError as e:
                print(f"NVML Error: {e}")
                break

'''train function without capturing layer consumption'''
def train_func(net, train_iter, test_iter, num_epochs, lr, device, filename, sampling_interval):
    torch.cuda.empty_cache()
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # save all epochs time data using list
    to_device_intervals_total = []
    forward_intervals_total = []
    loss_intervals_total = []
    backward_intervals_total = []
    optimize_intervals_total = []
    test_intervals_total = []

    # create a list to store the epoch time data
    epoch_intervals_total = []
    
    # Initialize NVML and sampling thread
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    stop_event = threading.Event()
    sampler_thread = threading.Thread(target=nvml_sampling_thread, args=(handle, filename, stop_event, sampling_interval))
    sampler_thread.start()

    for epoch in range(num_epochs):
        print('The epoch is:', epoch+1)
        metric = d2l.Accumulator(3)  # train_loss, train_acc, num_examples
        to_device_intervals_epoch = []  # 用来记录本epoch每个batch的to_device时间段
        forward_intervals_epoch = []  # 用来记录本epoch每个batch的forward时间段
        loss_intervals_epoch = []  # 用来记录本epoch每个batch的loss时间段
        backward_intervals_epoch = [] 
        optimize_intervals_epoch = []
        test_intervals_epoch = []   
        epoch_intervals_epoch = []  # 用来记录本epoch的时间段

        epoch_start_time = time.time()

        net.train()
        for i, (X, y) in enumerate(train_iter):
            torch.cuda.empty_cache()
            print('The batch is:', i+1)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()

            # 记录to_device前后的时间戳
            start_ttd_time = time.time()
            X, y = X.to(device), y.to(device)
            torch.cuda.synchronize()
            end_ttd_time = time.time()
            to_device_intervals_epoch.append((start_ttd_time, end_ttd_time))

            # forward with autocast
            start_forward_time = time.time()
            y_hat = net(X)
            torch.cuda.synchronize()
            end_forward_time = time.time()
            forward_intervals_epoch.append((start_forward_time, end_forward_time))

            # loss
            start_loss_time = time.time()
            l = loss_fn(y_hat, y)
            torch.cuda.synchronize()
            end_loss_time = time.time()
            loss_intervals_epoch.append((start_loss_time, end_loss_time))

            # backward with scaler
            start_backward_time = time.time()
            l.backward()
            torch.cuda.synchronize()
            end_backward_time = time.time()
            backward_intervals_epoch.append((start_backward_time, end_backward_time))

            # optimize with scaler
            start_optimize_time = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            end_optimize_time = time.time()
            optimize_intervals_epoch.append((start_optimize_time, end_optimize_time))

            with torch.no_grad():
                metric.add(l.item() * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            train_acc = metric[1] / metric[2]

            # Free memory for the batch
            del X, y, y_hat, l
            torch.cuda.empty_cache()

        # Evaluation (test)
        start_test_time = time.time()
        with torch.no_grad():
            test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        end_test_time = time.time()
        print(f'train acc {train_acc:.3f}, test acc {test_acc:.3f}')
        test_intervals_epoch.append((start_test_time, end_test_time))

        epoch_end_time = time.time()
        epoch_intervals_epoch.append((epoch_start_time, epoch_end_time))

        # data need to be saved
        # add the intervals_epoch to intervals_total
        to_device_intervals_total.append(to_device_intervals_epoch)
        forward_intervals_total.append(forward_intervals_epoch)
        loss_intervals_total.append(loss_intervals_epoch)
        backward_intervals_total.append(backward_intervals_epoch)
        optimize_intervals_total.append(optimize_intervals_epoch)
        test_intervals_total.append(test_intervals_epoch)
        epoch_intervals_total.append(epoch_intervals_epoch)
        torch.cuda.empty_cache()


    # End training and close thread
    stop_event.set()
    sampler_thread.join()
    pynvml.nvmlShutdown()

    return to_device_intervals_total, forward_intervals_total, loss_intervals_total, backward_intervals_total, optimize_intervals_total, test_intervals_total, epoch_intervals_total

'''train function with capturing layer consumption'''



'''train process with save files'''
def train_model(main_folder, batch_size, num_epochs, round, lr, device, sample_interval, net, dataset):
    print(f'The epoch is set: {num_epochs}, batch is set: {batch_size}, is in {round+1}th running')
    # create the folder to store the data
    sr_number = int(sample_interval*1000)
    epoch_batch_folder = f'E{num_epochs}_B{batch_size}_R{round}_SR{sr_number}'
    dataset_dir = dataset

    # the folder path is main_folder/epoch_batch_folder
    folder_path = main_folder/epoch_batch_folder
    print(f'The folder path is: {folder_path}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        pass
    
    if dataset == 'fashion_mnist':
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    elif dataset == 'cifar100':
        train_iter, test_iter = load_data_cifar100(batch_size, resize=224)
    elif dataset == 'cifar10':
        train_iter, test_iter = load_data_cifar10(batch_size, resize=224)
        
    # show the shape of the data
    list_of_i = []
    for i, (X, y) in enumerate(train_iter):
        if i < 3:
            print('the shape of the', i, 'batch of the train_iter is:', X.shape)
        else:
            pass
        list_of_i.append(i)
    print(f'The number of batches is: {np.array(list_of_i).shape}')
    to_device_intervals_total, forward_intervals_total, loss_intervals_total,\
          backward_intervals_total, optimize_intervals_total, test_intervals_total, epoch_intervals_total = train_func(net, train_iter, test_iter, num_epochs, lr, device, folder_path, sample_interval)

    # transfer the data to the numpy array
    to_device_data = np.array(to_device_intervals_total)
    forward_time = np.array(forward_intervals_total)
    loss_time = np.array(loss_intervals_total)
    backward_time = np.array(backward_intervals_total)
    optimize_time = np.array(optimize_intervals_total)
    test_time = np.array(test_intervals_total)
    epoch_time = np.array(epoch_intervals_total)

    # save the data
    np.save(folder_path/'to_device.npy', to_device_data, allow_pickle=True)
    np.save(folder_path/'forward.npy', forward_time, allow_pickle=True)
    np.save(folder_path/'loss.npy', loss_time, allow_pickle=True)
    np.save(folder_path/'backward.npy', backward_time, allow_pickle=True)
    np.save(folder_path/'optimize.npy', optimize_time, allow_pickle=True)
    np.save(folder_path/'test.npy', test_time, allow_pickle=True)
    np.save(folder_path/'epoch.npy', epoch_time, allow_pickle=True)

    # # transfer the data to dataframe
    # to_device_df = pd.DataFrame(to_device_data, columns=['start_time', 'end_time'])
    # forward_df = pd.DataFrame(forward_time, columns=['start_time', 'end_time'])
    # loss_df = pd.DataFrame(loss_time, columns=['start_time', 'end_time'])
    # backward_df = pd.DataFrame(backward_time, columns=['start_time', 'end_time'])
    # optimize_df = pd.DataFrame(optimize_time, columns=['start_time', 'end_time'])
    # test_df = pd.DataFrame(test_time, columns=['start_time', 'end_time'])
    # epoch_df = pd.DataFrame(epoch_time, columns=['start_time', 'end_time'])

    # # save the data
    # to_device_df.to_csv(folder_path/'to_device.csv', index=False)
    # forward_df.to_csv(folder_path/'forward.csv', index=False)
    # loss_df.to_csv(folder_path/'loss.csv', index=False)
    # backward_df.to_csv(folder_path/'backward.csv', index=False)
    # optimize_df.to_csv(folder_path/'optimize.csv', index=False)
    # test_df.to_csv(folder_path/'test.csv', index=False)
    # epoch_df.to_csv(folder_path/'epoch.csv', index=False)


    
