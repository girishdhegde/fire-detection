import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import torchvision.utils as vutils

# from model import yolo
from yolo_resnet import yolo
from utils import convert, iou, nms, yoloLoss
from dataset import dataset

# settings
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# model training device cuda or cpu
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch size
bs  = 4
# learning rate
lr = 1e-3
# starting epoch
start = 1
# lost epoch
end = 50
# model weight save path
# save_path = './weights/weight_res.pt'
save_path = './weights/weight.pt'
# log file path
# logfile = './log_res.txt'
logfile = './log.txt'
# weight load path
# load = './weights/weight_res.pt'
# load = './weights/weight.pt'
load = None
# image dataset path
img_path = './trainset/images/'
# label dataset path
lbl_path = './trainset/labels/'
# size of image
size = 448
# Target Grid 
S = 7
# Bounding Boxex per Grid Cell
B = 2
# Number of Classes
C = 80 # 2 fir fire smoke dataset 80 for coco
# Prediction Per Cell Vector Length
E = (C+B*5)

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------


# creating log file if not found
if not os.path.isfile(logfile):
    with open(logfile, 'w') as f:
        pass

# Creating DataLoader
trainset = dataset(img_path, lbl_path, (size, size), device)
loader   = DataLoader(trainset, batch_size=bs, shuffle=True, )
batches  = len(loader)

print('Total training samples: ', len(trainset))

# Initializing the model
net = yolo().to(device)
net.train()

# Initializing Adam Optimizer With Model Parameters
optimizer = optim.Adam(net.parameters(), lr=lr)

# Dynamic learning rate
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5, )

# LOading Pretrained Weight
if load is not None:
    state = torch.load(load)
    try:
        if state.__contains__('yolo'):
            net.load_state_dict(state['yolo'])
        else:
             net.load_state_dict(state)
        if state.__contains__('optimizer'):
            optimizer.load_state_dict(state['optimizer'])
    except:
        print('Unable to load model')
        

# Loss Function
criterion  = yoloLoss(S=S, B=B, C=C)

# train loop
for epoch in range(start, end):
    net_loss = 0
    startTime = time.time()
    for batch, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        out = net(data)
        loss = criterion(out, target)
        loss.backward()
        # Update Weights
        optimizer.step()
        net_loss += loss.item()
       
        print(f'[epoch][{epoch}/{end}]\t[batch][{batch+1}/{batches}]\t Loss: {loss.item()}')

    # save weights
    params = {'yolo': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(params, save_path)
    
    # Update Learning Rate
    scheduler.step()

    print(f'[epoch][{epoch}/{end}]\t Loss: {net_loss/len(loader)}, \tTime: {time.time()-startTime}s')

    # writing log
    with open(logfile, 'a') as f:
        f.write(f'[epoch][{epoch}/{end}]\t Loss: {net_loss/len(loader)}, \tTime: {time.time()-startTime}s\n')
