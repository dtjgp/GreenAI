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
    conv_arch = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
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