import time
import os
import sys
import h5py
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import scipy.misc
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import  argparse
from train import Model
# from plot import plot
from model import WaterFusion
import logging
from DeepWB import deep_wb_single_task
from dataset import TrainDataSet
# import utils
from thop import profile
import datetime

# from MSBDN.MSBDN import Net
argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='logs/')
argparser.add_argument('--train', action='store_true')
argparser.add_argument('--resume',action='store_true')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--ssim', type=float, default=300)
argparser.add_argument('--mse', type = float, default=20)
argparser.add_argument('--vgg', type = float, default=1)
argparser.add_argument('--egan', type=float, default=0.1)
argparser.add_argument('--w', type = float, default=None)
argparser.add_argument('--patchD_3', type = int, default=5)
args = argparser.parse_args()
WB_model = 'DeepWB/models'
MSBDN_model = 'networks/model'

train_dataset = '../underwater_data.h5'

test_data = 'UIEDBtest/'

output_folder = 'results/'

# ssh -L 18097:127.0.0.1:8097 jiangzhiying@172.31.73.75


net_awb = deep_wb_single_task.deepWBnet()
MSBDN = torch.load(os.path.join(MSBDN_model, 'model.pkl'), map_location=lambda storage, loc: storage)
model = WaterFusion(in_channels= 3, out_channels = 3, num_features = 64, growthrate = 32)

input = torch.randn(1, 3, 256, 256)
flops1, params1 = profile(net_awb, inputs=(input, ))
flops2, params2 = profile(MSBDN, inputs=(input, ))
pre1 = net_awb(input)
pre2 = MSBDN(input)
flops3, params3 = profile(model, inputs=(pre1,pre2, ))

print('Flops1: ',flops1)
print('Params1: ',params1)
print('Flops2: ',flops2)
print('Params2: ',params2)
print('Flops3: ',flops3)
print('Params3: ',params3)
flops = flops1 + flops2 + flops3
params = params1 + params2 + params3
print('Flops: ',flops)
print('Params: ',params)