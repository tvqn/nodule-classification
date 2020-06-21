import os
import sys
import pickle
sys.path.insert(0, 'model')
sys.path.insert(0, 'preprocessing')
sys.path.insert(0, 'running')

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from model import generate_model
from running import train as T

exp_path = './experiment'
modify = True
class opts:
    def __init__(self):
        self.mode = 'score'
        self.batch_size = 1
        self.no_cuda = False
        self.verbose = True
        
        self.sample_size  = 64
        self.sample_duration = 64
        self.n_classes = 2
        
        self.model_depth = 10
        self.name_architecture = 'bottleneck'
        self.model_name = 'resnet'        
        self.resnet_shortcut = 'B'
        self.resnext_cardinality = 32

path = os.path.join(exp_path, 'model_info.p')
if not os.path.exists(path) or modify:
    opt = opts()
    with open(path, 'wb') as f:
        pickle.dump(opt, f)
else:
    with open(path, 'rb') as f:
        opt = pickle.load(f)

model = generate_model(opt)

if opt.verbose:
    from torchsummary import summary
    summary(model, (3, 64, 64, 64))
# Data = {'train': 'S:\Google Drive\data3\\train', 'val': 'S:\Google Drive\data3\\val'}
Data = {'train': '/media/whale/Storage/Google Drive/data3/train', 'val': '/media/whale/Storage/Google Drive/data3/val'}
learning_rate = 1e-05
momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum= momentum)
weight = torch.tensor([17.04, 10]).cuda()
criterion = nn.CrossEntropyLoss(weight=weight)

T.train(model, criterion, optimizer, opt.batch_size, Data, size = 64, save_path = exp_path, num_epochs=2)