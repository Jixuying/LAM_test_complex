import torch
import numpy as np
import torch.nn as nn
from data_process import interpolation
import torch.nn.functional as F
from torchvision import datasets, transforms
from complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from complexFunctions import complex_relu, complex_max_pool2d
from scipy.io import loadmat,savemat
# from tensorboardX import SummaryWriter
# import tensorwatch as tw
import torchvision.models
# from torchviz import make_dot
import os.path

'''
复数DNCNN
'''
DNCNN_HIDDENS = 18
class DNCNN_ComplexNet(nn.Module):

  def __init__(self,BN = True,Dropout = False):
    #这里 ComplexNet继承父类nn.Module中的init
    super(DNCNN_ComplexNet, self).__init__()#https://www.runoob.com/python/python-func-super.html
    self.dobn = BN
    self.dodrop = Dropout  # 根据传入网络中的参数来决定是否执行dropout或者batch normalization
    self.hidden = []
    self.bns = []
    self.drops = []
    # self.conv1 = ComplexConv2d(1, 64, 3, 1, 1)
    # self.conv2 = ComplexConv2d(64, 64, 3, 1, 1)
    # self.conv3 = ComplexConv2d(64, 1, 3, 1, 1)

    self.conv1 = ComplexConv2d(1, 32, 3, 1, 1)
    self.conv2 = ComplexConv2d(32, 32, 3, 1, 1)
    self.conv3 = ComplexConv2d(32, 1, 3, 1, 1)

    for i in range(DNCNN_HIDDENS):
        # conv = ComplexConv2d(64, 64, 3, 1, 1)
        conv = ComplexConv2d(32, 32, 3, 1, 1)
        setattr(self, 'conv2_hideen%i' % i, conv)
        self.hidden.append(conv)
        if self.dobn:
            # bn = ComplexBatchNorm2d(64)
            bn = ComplexBatchNorm2d(32)
            setattr(self, 'bn%i' % i, bn)
            self.bns.append(bn)

    def forward(self, x,device): # forward函数定义了网络的前向传播的顺序
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
      for i in range(DNCNN_HIDDENS):
          xr, xi = self.hidden[i](xr, xi)
          # with torch.no_grad():
          #     outputs.append(xr)
          #     outputs.append(xi)
          if self.dobn:
              xr, xi = self.bns[i](xr, xi)
          xr, xi = complex_relu(xr, xi)

      xr, xi = self.conv3(xr, xi)
      # with torch.no_grad():
      #     outputs.append(xr)
      #     outputs.append(xi)
      # xr, xi = complex_relu(xr, xi)

  #     x_out = torch.zeros(len(xr[:, 0, 0, 0]), 2, len(xr[1,0,:,1]),len(xr[1,0,1,:]))
  #     x_out = x_out.to(device)
  #     x_out[:, 0, :, :] = xr[:, 0, :, :]
  #     x_out[:, 1, :, :] = xi[:, 0, :, :]

  #     x_diff = x - x_out
      # return x_diff,outputs
  #     return x_diff
      return x - torch.cat([xr,xi],dim=1)

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
