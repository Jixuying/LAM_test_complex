import torch
import numpy as np
import torch.nn as nn
from data_process import interpolation
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d
from scipy.io import loadmat,savemat
from tensorboardX import SummaryWriter
# import tensorwatch as tw
import torchvision.models
from torchviz import make_dot
import os.path

class SRCNN_ComplexNet(nn.Module):

    def __init__(self):
        # 这里 ComplexNet继承父类nn.Module中的init
        super(SRCNN_ComplexNet, self).__init__()  # https://www.runoob.com/python/python-func-super.html

        self.conv1 = ComplexConv2d(1, 32, 9, 1, 4)
        self.conv2 = ComplexConv2d(32, 16, 1, 1, 0)
        self.conv3 = ComplexConv2d(16, 1, 5, 1, 2)

    def forward(self, x):  # forward函数定义了网络的前向传播的顺序
        with torch.no_grad():
            xr = x[:, 0, :, :]  # x就是传进来的data
            # imaginary part to zero
            # xi = torch.zeros(xr.shape, dtype=xr.dtype, device=xr.device)
            xi = x[:, 1, :, :]

            xr = xr[:, None, :, :]
            xi = xi[:, None, :, :]
            # outputs = []
        xr, xi = self.conv1(xr, xi)
        # with torch.no_grad():
        #     outputs.append(xr)
        #     outputs.append(xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.conv2(xr, xi)
        # with torch.no_grad():
        #     outputs.append(xr)
        #     outputs.append(xi)
        xr, xi = complex_relu(xr, xi)
        xr, xi = self.conv3(xr, xi)
        # with torch.no_grad():
        #     outputs.append(xr)
        #     outputs.append(xi)
        # xr, xi = complex_relu(xr, xi)

        x_out = torch.zeros(len(xr[:, 0, 0, 0]), 2, len(xr[1, 0, :, 1]), len(xr[1, 0, 1, :]))
        x_out[:, 0, :, :] = xr[:, 0, :, :]
        x_out[:, 1, :, :] = xi[:, 0, :, :]

        # return x_out,outputs
        return x_out

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))