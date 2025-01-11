'''
This code is to store all the models that we have used in our project.

The models are:
    1. AlexNet
    2. ResNet(18, 34, 50)
    3. VGG(11, 13, 16)
    4. MobileNetV1, MobileNetV2
    5. GoogleNet(with 9 different modifications)

And for each model, we try to capture each layer running time, with the interpolation method as well as
the real time power, we can calculate the energy consumption of each layer, and then the whole model.

For the time period, we record 6 different steps:
    Training:
        1. Data loading time(to_device)
        2. Forward propagation time(forward)
        3. Loss calculation time(loss)
        4. Backward propagation time(backward)
        5. Optimizer update time(optimize)
    Testing:
        6. Forward propagation time(test)

For each layer in the forward process, we try to record the time of each layer, 
and then calculate the energy with interpolation method.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from d2l import torch as d2l
import numpy as np
from functools import wraps
import time


# create a decorator to record the time of each layer
def timeit(layer_name):
    """装饰器，用于记录方法的执行时间，并存储到 self.layer_time 中"""
    def decorator(func):
        @wraps(func)  # 使用 wraps 修饰 wrapper 函数
        def wrapper(self, *args, **kwargs):
            start_t = time.time()
            result = func(self, *args, **kwargs)
            torch.cuda.synchronize()  # 确保时间精确性
            end_t = time.time()
            self.layer_time[layer_name] = [start_t, end_t]
            return result
        return wrapper
    return decorator

############################################################################################################
'''AlexNet'''
class AlexNet(nn.Module):
    def __init__(self, img_channel, num_labels):
        super(self).__init__()
        # define the layers
        self.conv1 = nn.Conv2d(img_channel, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(4096, num_labels)
        
        # record the time of each layer
        self.layer_time = {}

    @timeit('conv1')
    def forward_conv1(self, x):
        return self.conv1(x)

    @timeit('relu1')
    def forward_relu1(self, x):
        return self.relu1(x)

    @timeit('pool1')
    def forward_pool1(self, x):
        return self.pool1(x)
        
    @timeit('conv2')
    def forward_conv2(self, x):
        return self.conv2(x)

    @timeit('relu2')
    def forward_relu2(self, x):
        return self.relu2(x)

    @timeit('pool2')
    def forward_pool2(self, x):
        return self.pool2(x)

    @timeit('conv3')
    def forward_conv3(self, x):
        return self.conv3(x)

    @timeit('relu3')
    def forward_relu3(self, x):
        return self.relu3(x)

    @timeit('conv4')
    def forward_conv4(self, x):
        return self.conv4(x)

    @timeit('relu4')
    def forward_relu4(self, x):
        return self.relu4(x)

    @timeit('conv5')
    def forward_conv5(self, x):
        return self.conv5(x)

    @timeit('relu5')
    def forward_relu5(self, x):
        return self.relu5(x)

    @timeit('pool5')
    def forward_pool5(self, x):
        return self.pool5(x)

    @timeit('adaptive_pool')
    def forward_adaptive_pool(self, x):
        return self.adaptive_pool(x)

    @timeit('flatten')
    def forward_flatten(self, x):
        return self.flatten(x)

    @timeit('fc1')
    def forward_fc1(self, x):
        return self.fc1(x)

    @timeit('relu6')
    def forward_relu6(self, x):
        return self.relu6(x)

    @timeit('dropout1')
    def forward_dropout1(self, x):
        return self.dropout1(x)

    @timeit('fc2')
    def forward_fc2(self, x):
        return self.fc2(x)

    @timeit('relu7')
    def forward_relu7(self, x):
        return self.relu7(x)

    @timeit('dropout2')
    def forward_dropout2(self, x):
        return self.dropout2(x)

    @timeit('fc3')
    def forward_fc3(self, x):
        return self.fc3(x)

    def forward(self, x):
        x = self.forward_conv1(x)
        x = self.forward_relu1(x)
        x = self.forward_pool1(x)
        
        x = self.forward_conv2(x)
        x = self.forward_relu2(x)
        x = self.forward_pool2(x)
        
        x = self.forward_conv3(x)
        x = self.forward_relu3(x)
        
        x = self.forward_conv4(x)
        x = self.forward_relu4(x)
        
        x = self.forward_conv5(x)
        x = self.forward_relu5(x)
        x = self.forward_pool5(x)
        
        x = self.forward_adaptive_pool(x)
        x = self.forward_flatten(x)
        
        x = self.forward_fc1(x)
        x = self.forward_relu6(x)
        x = self.forward_dropout1(x)
        
        x = self.forward_fc2(x)
        x = self.forward_relu7(x)
        x = self.forward_dropout2(x)
        
        x = self.forward_fc3(x)
        return x

def alexnet(img_channel, num_labels):
    net = AlexNet(img_channel, num_labels)
    return net


############################################################################################################
'''ResNet'''
'''ResNet18'''
class ResidualBlock(nn.Module):  # residual block for resnet18 and resnet34
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # Define the layers 
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        # Dictionary to record time for each sub-layer
        self.layer_time = {}

    @timeit('conv1')
    def forward_conv1(self, x):
        return self.conv1(x)

    @timeit('bn1') 
    def forward_bn1(self, x):
        return self.bn1(x)

    @timeit('relu1')
    def forward_relu1(self, x):
        return self.relu1(x)

    @timeit('conv2')
    def forward_conv2(self, x):
        return self.conv2(x)

    @timeit('bn2')
    def forward_bn2(self, x):
        return self.bn2(x)

    @timeit('conv3')
    def forward_conv3(self, x):
        return self.conv3(x)

    @timeit('residual_add_relu2')
    def forward_residual(self, out, x):
        out = out + x
        return self.relu2(out)

    def forward(self, X):
        out = self.forward_conv1(X)
        out = self.forward_bn1(out)
        out = self.forward_relu1(out)
        out = self.forward_conv2(out)
        out = self.forward_bn2(out)
        
        if self.conv3:
            X = self.forward_conv3(X)
            
        out = self.forward_residual(out, X)
        return out
    
    
def resnet18(img_channel, num_labels):
    b1 = nn.Sequential(
        nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResidualBlock(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(ResidualBlock(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))

    net = nn.Sequential(
                        b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(),
                        nn.Linear(512, num_labels)
                    )
    return net

'''ResNet34'''
def resnet34(img_channel, num_labels):
    # blk = Residual(3,6, use_1x1conv=True, strides=2)

    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def resnet_block(input_channels, num_channels, num_residuals,
                    first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(ResidualBlock(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(ResidualBlock(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 4))
    b4 = nn.Sequential(*resnet_block(128, 256, 6))
    b5 = nn.Sequential(*resnet_block(256, 512, 3))

    net = nn.Sequential(b1, b2, b3, b4, b5,
                        nn.AdaptiveAvgPool2d((1,1)),
                        nn.Flatten(), nn.Linear(512, num_labels))
    return net

'''ResNet50'''
class Residual50(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels , kernel_size=1, stride=strides, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels )
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_channels , num_channels , kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels )
        self.relu2 = nn.ReLU()  
        self.conv3 = nn.Conv2d(num_channels , num_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.ds_conv = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, bias=False)
        self.ds_bn = nn.BatchNorm2d(num_channels)
        
        if use_1x1conv or strides != 1:
            self.downsample = nn.Sequential(
                self.ds_conv, self.ds_bn
            )
        else:
            self.downsample = None

        

        # Dictionary to record time for each sub-layer
        self.layer_time = {}

    @timeit('conv1')
    def forward_conv1(self, x):
        return self.conv1(x)

    @timeit('bn1')
    def forward_bn1(self, x):
        return self.bn1(x)

    @timeit('relu1')
    def forward_relu1(self, x):
        return self.relu1(x)

    @timeit('conv2')
    def forward_conv2(self, x):
        return self.conv2(x)

    @timeit('bn2')
    def forward_bn2(self, x):
        return self.bn2(x)

    @timeit('relu2')
    def forward_relu2(self, x):
        return self.relu2(x)

    @timeit('conv3')
    def forward_conv3(self, x):
        return self.conv3(x)

    @timeit('bn3')
    def forward_bn3(self, x):
        return self.bn3(x)

    @timeit('ds_conv')
    def downsample_conv(self, x):
        return self.ds_conv(x)
    
    @timeit('ds_bn')
    def downsample_bn(self, x):
        return self.ds_bn(x)

    def downsample(self, x):
        x = self.downsample_conv(x)
        return self.downsample_bn(x)

    @timeit('final_relu')
    def forward_final_relu(self, x):
        return F.relu(x)

    def forward(self, X):
        Y = self.forward_conv1(X)
        Y = self.forward_bn1(Y)
        Y = self.forward_relu1(Y)
        
        Y = self.forward_conv2(Y)
        Y = self.forward_bn2(Y)
        Y = self.forward_relu2(Y)
        
        Y = self.forward_conv3(Y)
        Y = self.forward_bn3(Y)
        
        if self.downsample:
            X = self.downsample(X)
            
        Y += X
        return self.forward_final_relu(Y)

def resnet50(img_channel, num_labels):
    b1 = nn.Sequential(
        nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual50(input_channels, num_channels, use_1x1conv=True, strides=2))
            else:
                blk.append(Residual50(num_channels, num_channels))
        return blk

    b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 4))
    b4 = nn.Sequential(*resnet_block(128, 256, 6))
    b5 = nn.Sequential(*resnet_block(256, 512, 3))

    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(), nn.Linear(512, num_labels)
    )
    return net


############################################################################################################
'''VGG'''
'''VGG11'''
class VGGBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super().__init__()
        layers = []
        self.num_convs = num_convs
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for _ in range(num_convs):
            layers.append(self.conv)
            layers.append(self.relu)
            in_channels = out_channels
        layers.append(self.max_pool)

    @timeit('conv')
    def forward_conv(self, x, layer_idx):
        return self.layers[layer_idx](x)
        
    @timeit('relu') 
    def forward_relu(self, x, layer_idx):
        return self.layers[layer_idx](x)
        
    @timeit('maxpool')
    def forward_pool(self, x):
        return self.layers[-1](x)

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.forward_conv(x, i*2)
            x = self.forward_relu(x, i*2+1)
        x = self.forward_pool(x)
        return x


def vgg11(input_channels, output_channels):
    # VGG-11 architecture
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
    in_channels = input_channels  # For RGB images
    # Create convolutional layers
    conv_layers = []
    for num_convs, out_channels in conv_arch:
        conv_layers.append(VGGBlock(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
            *conv_layers, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, output_channels)  # Output layer for 1000 classes
        )
    return net

'''VGG13'''
def vgg13(input_channels, output_channels):
    # VGG-13 architecture
    conv_arch = [(2, 64), (2, 128), (2, 256), (2, 512), (2, 512)]
    in_channels = input_channels  # For RGB images
    # Create convolutional layers
    conv_layers = []
    for num_convs, out_channels in conv_arch:
        conv_layers.append(VGGBlock(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net = nn.Sequential(
            *conv_layers, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, output_channels)  # Output layer for 1000 classes
        )
    return net

'''VGG16'''
def vgg16(input_channels, output_channels):
    # VGG-16 architecture
    conv_arch = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
    in_channels = input_channels  # For RGB images
    # Create convolutional layers
    conv_layers = []
    for num_convs, out_channels in conv_arch:
        conv_layers.append(VGGBlock(num_convs, in_channels, out_channels))
        in_channels = out_channels

    net =  nn.Sequential(
            *conv_layers, nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, output_channels)  # Output layer for 1000 classes
        )
    return net

############################################################################################################
'''GoogleNet'''
'''GoogleNet_original'''
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_conv = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_conv1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_conv2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_conv1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_conv2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_conv = nn.Conv2d(in_channels, c4, kernel_size=1)

        # initial the relu layer
        self.p1_relu = nn.ReLU()
        self.p2_relu1 = nn.ReLU()
        self.p2_relu2 = nn.ReLU()
        self.p3_relu1 = nn.ReLU()
        self.p3_relu2 = nn.ReLU()
        self.p4_relu = nn.ReLU()

        self.layer_time = {}

    @timeit('p1_conv')
    def forward_p1_conv(self, x):
        return self.p1_conv(x)

    @timeit('p1_relu')
    def forward_p1_relu(self, x):
        return self.p1_relu(x)

    @timeit('p2_1_conv')
    def forward_p2_1_conv(self, x):
        return self.p2_conv1(x)

    @timeit('p2_1_relu')
    def forward_p2_1_relu(self, x):
        return self.p2_relu1(x)

    @timeit('p2_2_conv')
    def forward_p2_2_conv(self, x):
        return self.p2_conv2(x)

    @timeit('p2_2_relu')
    def forward_p2_2_relu(self, x):
        return self.p2_relu2(x)

    @timeit('p3_1_conv')
    def forward_p3_1_conv(self, x):
        return self.p3_conv1(x)

    @timeit('p3_1_relu')
    def forward_p3_1_relu(self, x):
        return self.p3_relu1(x)

    @timeit('p3_2_conv')
    def forward_p3_2_conv(self, x):
        return self.p3_conv2(x)

    @timeit('p3_2_relu')
    def forward_p3_2_relu(self, x):
        return self.p3_relu2(x)

    @timeit('p4_pool')
    def forward_p4_pool(self, x):
        return self.p4_mp(x)

    @timeit('p4_conv')
    def forward_p4_conv(self, x):
        return self.p4_conv(x)

    @timeit('p4_relu')
    def forward_p4_relu(self, x):
        return self.p4_relu(x)
    
    @timeit('concat')
    def forward_concat(self, p1_out, p2_out, p3_out, p4_out):
        return torch.cat((p1_out, p2_out, p3_out, p4_out), dim=1)
    
    def forward(self, x):
        p1_out = self.forward_p1_conv(x)
        p1_out = self.forward_p1_relu(p1_out)

        p2_out = self.forward_p2_1_conv(x)
        p2_out = self.forward_p2_1_relu(p2_out)
        p2_out = self.forward_p2_2_conv(p2_out)
        p2_out = self.forward_p2_2_relu(p2_out)

        p3_out = self.forward_p3_1_conv(x)
        p3_out = self.forward_p3_1_relu(p3_out)
        p3_out = self.forward_p3_2_conv(p3_out)
        p3_out = self.forward_p3_2_relu(p3_out)

        p4_out = self.forward_p4_pool(x)
        p4_out = self.forward_p4_conv(p4_out)
        p4_out = self.forward_p4_relu(p4_out)

        out = self.forward_concat(p1_out, p2_out, p3_out, p4_out)
        # 在通道维度上连结输出
        return out

def Googlenet(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, num_labels))
    return net

'''GoogleNet_mod1'''
class Inception_mod1(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception_mod1, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p2, p3, p4), dim=1)
    
def Googlenet_mod1(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod1(192, 64, (96, 128), (16, 32), 32),
                   Inception_mod1(256-64, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod1(480-128, 192, (96, 208), (16, 48), 64),
                   Inception_mod1(512-192, 160, (112, 224), (24, 64), 64),
                   Inception_mod1(512-160, 128, (128, 256), (24, 64), 64),
                   Inception_mod1(512-128, 112, (144, 288), (32, 64), 64),
                   Inception_mod1(528-112, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod1(832-256, 256, (160, 320), (32, 128), 128),
                   Inception_mod1(832-256, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024-384, num_labels))
    return net

'''GoogleNet_mod2'''
class Inception_mod2(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception_mod2, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        # p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p3, p4), dim=1)
    
def Googlenet_mod2(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod2(192, 64, (96, 128), (16, 32), 32),
                   Inception_mod2(256-128, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod2(480-192, 192, (96, 208), (16, 48), 64),
                   Inception_mod2(512-208, 160, (112, 224), (24, 64), 64),
                   Inception_mod2(512-224, 128, (128, 256), (24, 64), 64),
                   Inception_mod2(512-256, 112, (144, 288), (32, 64), 64),
                   Inception_mod2(528-288, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod2(832-320, 256, (160, 320), (32, 128), 128),
                   Inception_mod2(832-320, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024-384, num_labels))
    return net

'''GoogleNet_mod3'''
class Inception_mod3(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception_mod3, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        # self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        # self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p4), dim=1)
    
def Googlenet_mod3(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod3(192, 64, (96, 128), (16, 32), 32),
                   Inception_mod3(256-32, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod3(480-96, 192, (96, 208), (16, 48), 64),
                   Inception_mod3(512-48, 160, (112, 224), (24, 64), 64),
                   Inception_mod3(512-64, 128, (128, 256), (24, 64), 64),
                   Inception_mod3(512-64, 112, (144, 288), (32, 64), 64),
                   Inception_mod3(528-64, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod3(832-128, 256, (160, 320), (32, 128), 128),
                   Inception_mod3(832-128, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024-128, num_labels))
    return net

'''GoogleNet_mod4'''
class Inception_mod4(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception_mod4, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3), dim=1)
    
def Googlenet_mod4(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod4(192, 64, (96, 128), (16, 32), 32),
                   Inception_mod4(256-32, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod4(480-64, 192, (96, 208), (16, 48), 64),
                   Inception_mod4(512-64, 160, (112, 224), (24, 64), 64),
                   Inception_mod4(512-64, 128, (128, 256), (24, 64), 64),
                   Inception_mod4(512-64, 112, (144, 288), (32, 64), 64),
                   Inception_mod4(528-64, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod4(832-128, 256, (160, 320), (32, 128), 128),
                   Inception_mod4(832-128, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024-128, num_labels))
    return net

'''GoogleNet_mod5'''
class Inception_mod5(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, **kwargs):
        super(Inception_mod5, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        # self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        # self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat([p2], dim=1)
    
def Googlenet_mod5(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod5(192, (96, 128)),
                   Inception_mod5(128, (128, 192)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod5(192, (96, 208)),
                   Inception_mod5(208, (112, 224)),
                   Inception_mod5(224, (128, 256)),
                   Inception_mod5(256, (144, 288)),
                   Inception_mod5(288, (160, 320)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod5(320, (160, 320)),
                   Inception_mod5(320, (192, 384)),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(384, num_labels))
    return net

'''GoogleNet_mod6'''
class Inception_mod6(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, **kwargs):
        super(Inception_mod6, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        # self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        # p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p2, p3), dim=1)
    
def Googlenet_mod6(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod6(192, (96, 128)),
                   Inception_mod6(128*2, (128, 192)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod6(192*2, (96, 208)),
                   Inception_mod6(208*2, (112, 224)),
                   Inception_mod6(224*2, (128, 256)),
                   Inception_mod6(256*2, (144, 288)),
                   Inception_mod6(288*2, (160, 320)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod6(320*2, (160, 320)),
                   Inception_mod6(320*2, (192, 384)),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(384*2, num_labels))
    return net

'''GoogleNet_mod7'''
class Inception_mod7(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, **kwargs):
        super(Inception_mod7, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        # self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p4_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

    def forward(self, x):
        # p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # 在通道维度上连结输出
        return torch.cat((p2, p3, p4), dim=1)
    
def Googlenet_mod7(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod7(192, (96, 128)),
                   Inception_mod7(128*3, (128, 192)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod7(192*3, (96, 208)),
                   Inception_mod7(208*3, (112, 224)),
                   Inception_mod7(224*3, (128, 256)),
                   Inception_mod7(256*3, (144, 288)),
                   Inception_mod7(288*3, (160, 320)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod7(320*3, (160, 320)),
                   Inception_mod7(320*3, (192, 384)),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(384*3, num_labels))
    return net

'''GoogleNet_mod8'''
class Inception_mod8(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c2, **kwargs):
        super(Inception_mod8, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p1_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p4_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

    def forward(self, x):
        p1 = F.relu(self.p1_2(F.relu(self.p1_1(x))))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(F.relu(self.p4_1(x))))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
    
def Googlenet_mod8(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b3 = nn.Sequential(Inception_mod8(192, (96, 128)),
                   Inception_mod8(128*4, (128, 192)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b4 = nn.Sequential(Inception_mod8(192*4, (96, 208)),
                   Inception_mod8(208*4, (112, 224)),
                   Inception_mod8(224*4, (128, 256)),
                   Inception_mod8(256*4, (144, 288)),
                   Inception_mod8(288*4, (160, 320)),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b5 = nn.Sequential(Inception_mod8(320*4, (160, 320)),
                   Inception_mod8(320*4, (192, 384)),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(384*4, num_labels))
    return net

'''GoogleNet_mod9'''
def Googlenet_mod9(img_channel, num_labels):
    b1 = nn.Sequential(nn.Conv2d(img_channel, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    # b3 = nn.Sequential(Inception_mod8(192, (96, 128)),
    #                Inception_mod8(128*4, (128, 192)),
    #                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    # b4 = nn.Sequential(Inception_mod8(192*4, (96, 208)),
    #                Inception_mod8(208*4, (112, 224)),
    #                Inception_mod8(224*4, (128, 256)),
    #                Inception_mod8(256*4, (144, 288)),
    #                Inception_mod8(288*4, (160, 320)),
    #                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    # b5 = nn.Sequential(Inception_mod8(320*4, (160, 320)),
    #                Inception_mod8(320*4, (192, 384)),
    #                nn.AdaptiveAvgPool2d((1,1)),
    #                nn.Flatten())
    
    b5 = nn.Sequential(
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

    net = nn.Sequential(b1, b2, b5, nn.Linear(192, num_labels))
    return net

############################################################################################################
'''MobileNet'''
'''MobileNetV1'''
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layer_time = {}

    @timeit('depthwise')
    def forward_depthwise(self, x):
        return self.depthwise(x)

    @timeit('pointwise') 
    def forward_pointwise(self, x):
        return self.pointwise(x)
        
    def forward(self, x):
        x = self.forward_depthwise(x)
        x = self.forward_pointwise(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),

            # Typically, 5 Depthwise Separable Convolutions are repeated here, each with stride 1
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],

            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, output_channels)
        )

    def forward(self, x):
        return self.model(x)

def MobileNetV1(input_channels, output_channels):
    net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            DepthwiseSeparableConv(32, 64, 1),
            DepthwiseSeparableConv(64, 128, 2),
            DepthwiseSeparableConv(128, 128, 1),
            DepthwiseSeparableConv(128, 256, 2),
            DepthwiseSeparableConv(256, 256, 1),
            DepthwiseSeparableConv(256, 512, 2),

            # Typically, 5 Depthwise Separable Convolutions are repeated here, each with stride 1
            *[DepthwiseSeparableConv(512, 512, 1) for _ in range(5)],

            DepthwiseSeparableConv(512, 1024, 2),
            DepthwiseSeparableConv(1024, 1024, 1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, output_channels)
        )
    
    return net

'''MobileNetV2'''
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = self.stride == 1 and in_channels == out_channels

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.relu2 = nn.ReLU6(inplace=True)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.layers = nn.Sequential(
            self.conv1, self.bn1, self.relu1,
            self.conv2, self.bn2, self.relu2,
            self.conv3, self.bn3
        )

    @timeit('conv1')
    def forward_conv1(self, x):
        return self.conv1(x)

    @timeit('bn1')
    def forward_bn1(self, x):
        return self.bn1(x)
    
    @timeit('relu1')  
    def forward_relu1(self, x):
        return self.relu1(x)

    @timeit('conv2')
    def forward_conv2(self, x):
        return self.conv2(x)

    @timeit('bn2')
    def forward_bn2(self, x):  
        return self.bn2(x)

    @timeit('relu2')
    def forward_relu2(self, x):
        return self.relu2(x)

    @timeit('conv3') 
    def forward_conv3(self, x):
        return self.conv3(x)

    @timeit('bn3')
    def forward_bn3(self, x):
        return self.bn3(x)

    def forward(self, x):
        if self.use_residual:
            identity = x
            x = self.forward_conv1(x)
            x = self.forward_bn1(x)
            x = self.forward_relu1(x)
            x = self.forward_conv2(x)
            x = self.forward_bn2(x)
            x = self.forward_relu2(x)
            x = self.forward_conv3(x)
            x = self.forward_bn3(x)
            return x + identity
        else:
            x = self.forward_conv1(x)
            x = self.forward_bn1(x)
            x = self.forward_relu1(x)
            x = self.forward_conv2(x)
            x = self.forward_bn2(x) 
            x = self.forward_relu2(x)
            x = self.forward_conv3(x)
            x = self.forward_bn3(x)
            return x

def MobileNetV2(input_channels, output_channels):
    first_layer = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

    inverted_residual_blocks = nn.Sequential(
            InvertedResidual(32, 16, 1, 1),
            InvertedResidual(16, 24, 2, 6),
            InvertedResidual(24, 24, 1, 6),
            InvertedResidual(24, 32, 2, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 32, 1, 6),
            InvertedResidual(32, 64, 2, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 64, 1, 6),
            InvertedResidual(64, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 96, 1, 6),
            InvertedResidual(96, 160, 2, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 160, 1, 6),
            InvertedResidual(160, 320, 1, 6)
        )

    last_layer = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, output_channels)
        )

    net = nn.Sequential(first_layer, inverted_residual_blocks, last_layer)
    return net