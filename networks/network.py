import torch
import torch.nn as nn
import numpy as np
import networks.piggyback_layers as nl
import math

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))
class FiLM(nn.Module):
    def __init__(self, nfilters):
        super().__init__()
        # Gamma
        # (num_batch, nfilters, height, width) * (1, nfilters, 1, 1) => (num_batch, nfilters, height, width)
        self.gamma = nn.Parameter(torch.ones(1, nfilters, 1, 1))
        # Beta
        self.beta = nn.Parameter(torch.zeros(1,nfilters, 1, 1))

    def forward(self, x):
        return x * self.gamma + self.beta

class FiLM_for_Linear(nn.Module):
    def __init__(self, nfilters):
        super().__init__()
        # Gamma
        self.gamma = nn.Parameter(torch.ones(nfilters))
        # Beta
        self.beta = nn.Parameter(torch.zeros(nfilters))

    def forward(self, x):
        return x * self.gamma + self.beta



class Conv_Net_FiLM_pooling(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.film1 = FiLM(32)
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.film2 = FiLM(32)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.film3 = FiLM(64)
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.film4 = FiLM(64)
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.film5 = FiLM(128)
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.film6 = FiLM(128)
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.film7 = FiLM_for_Linear(256)
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.film1(self.conv1(x)))
        act2=self.relu(self.film2(self.conv2(act1)))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.film3(self.conv3(h)))
        act4=self.relu(self.film4(self.conv4(act3)))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.film5(self.conv5(h)))
        act6=self.relu(self.film6(self.conv6(act5))) # act6 : (256,128,8,8)
        h=self.drop1(self.MaxPool(act6)) # h : (256,128,4,4)
        h=h.view(x.shape[0],-1) # h : (256, 2048)
        act7 = self.relu(self.film7(self.fc1(h))) # act7 : (256, 256)
        h = self.drop2(act7)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

class Conv_Net_pooling(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.conv1(x))
        act2=self.relu(self.conv2(act1))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.conv3(h))
        act4=self.relu(self.conv4(act3))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.conv5(h))
        act6=self.relu(self.conv6(act5)) # act6 : (256,128,8,8)
        h=self.drop1(self.MaxPool(act6)) # h : (256,128,4,4)
        h=h.view(x.shape[0],-1) # h : (256, 2048)
        act7 = self.relu(self.fc1(h)) # act7 : (256, 256)
        h = self.drop2(act7)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

class Conv_Net_wo_fc(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        # stride = 2, padding =0
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(size,3, padding=0) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(s,3, padding=0) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(64,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu((self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
        act2=self.relu((self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
        act3=self.relu((self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
        act4=self.relu((self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y
class Conv_Net_FiLM_remember_wo_batchnorm(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        self.conv1 = nn.Conv2d(ncha,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.film1=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film1.append(FiLM(32))
        self.conv2 = nn.Conv2d(32,32,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        s = s//2 # 16
        self.film2=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film2.append(FiLM(32))
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        self.film3=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film3.append(FiLM(64))
        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 16
        s = s//2 # 8
        self.film4=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film4.append(FiLM(64))
        self.conv5 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        self.film5=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film5.append(FiLM(128))
        self.conv6 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 8
        s = s//2 # 4
        self.film6=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film6.append(FiLM(128))
        self.fc1 = nn.Linear(s*s*128,256) # 2048
        self.drop1 = nn.Dropout(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.MaxPool = torch.nn.MaxPool2d(2)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, task_t, avg_act = False):
        act1=self.relu(self.film1[task_t](self.conv1(x)))
        act2=self.relu(self.film2[task_t](self.conv2(act1)))
        h=self.drop1(self.MaxPool(act2))
        act3=self.relu(self.film3[task_t](self.conv3(h)))
        act4=self.relu(self.film4[task_t](self.conv4(act3)))
        h=self.drop1(self.MaxPool(act4))
        act5=self.relu(self.film5[task_t](self.conv5(h)))
        act6=self.relu(self.film6[task_t](self.conv6(act5))) # act6 : (256,128,8,8)
        h=self.drop1(self.MaxPool(act6)) # h : (256,128,4,4)
        h=h.view(x.shape[0],-1) # h : (256, 2048)
        act7 = self.relu(self.fc1(h)) # act7 : (256, 256)
        h = self.drop2(act7)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y
class Conv_Net_FiLM_remember(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        # stride = 2, padding =0
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(size,3, padding=0) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.film1=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film1.append(FiLM(64))

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(s,3, padding=0) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.film2=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film2.append(FiLM(64))

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.film3=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film3.append(FiLM(64))

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.film4=torch.nn.ModuleList()
        for t,_ in self.task_info:
            self.film4.append(FiLM(64))

        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(64,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, task_t,  avg_act = False):
        act1=self.relu(self.film1[task_t](self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
        act2=self.relu(self.film2[task_t](self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
        act3=self.relu(self.film3[task_t](self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
        act4=self.relu(self.film4[task_t](self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        y = []
        for t, i in self.task_info:
            y.append(self.last[t](h))
        return y
class Conv_Net_FiLM_wo_fc(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        # stride = 2, padding =0
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(size,3, padding=0) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.film1 = FiLM(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=0, stride=2)
        s = compute_conv_output_size(s,3, padding=0) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.film2 = FiLM(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.film3 = FiLM(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=0,stride=2)
        s = compute_conv_output_size(s,3, padding=0)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.film4 = FiLM(64)

        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(64,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.film1(self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
        act2=self.relu(self.film2(self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
        act3=self.relu(self.film3(self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
        act4=self.relu(self.film4(self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y
class Conv_Net_FiLM_last(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        # stride = 2, padding =0
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.film1 = FiLM(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.film2 = FiLM(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.film3 = FiLM(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.film4 = FiLM(64)

        self.fc1 = nn.Linear(s*s*64,256)
        self.film5 = FiLM_for_Linear(256)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.film1(self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
        act2=self.relu(self.film2(self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
        act3=self.relu(self.film3(self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
        act4=self.relu(self.film4(self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        h = self.relu(self.film5(self.fc1(h))) # act5 : (256, 256)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

class Conv_Net_FiLM(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        # stride = 2, padding =0
        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.film1 = FiLM(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.film2 = FiLM(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.film3 = FiLM(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.film4 = FiLM(64)

        self.fc1 = nn.Linear(s*s*64,256)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.film1(self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
        act2=self.relu(self.film2(self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
        act3=self.relu(self.film3(self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
        act4=self.relu(self.film4(self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        h = self.relu(self.fc1(h)) # act5 : (256, 256)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

class Conv_Net(nn.Module):
    def __init__(self, inputsize, task_info):
        super().__init__()

        ncha,size,_=inputsize
        self.task_info = task_info

        self.conv1 = nn.Conv2d(ncha,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(size,3, padding=1) # 32
        self.batchnorm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1) # 32
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        s = compute_conv_output_size(s,3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(s*s*64,256)
        self.last=torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(256,n))
        self.relu = torch.nn.ReLU()

    def forward(self, x, avg_act = False):
        act1=self.relu(self.batchnorm1(self.conv1(x))) # act1 : (256,64,32,32)
        act2=self.relu(self.batchnorm2(self.conv2(act1))) # act2 : (256,64,32,32)
        act3=self.relu(self.batchnorm3(self.conv3(act2))) # act3 : (256,64,32,32)
        act4=self.relu(self.batchnorm4(self.conv4(act3))) # act4 : (256,64,32,32)

        h=act4.view(x.shape[0],-1) # h : (256, 65536)
        h = self.relu(self.fc1(h)) # act5 : (256, 256)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

class MLP_Net(torch.nn.Module):

    def __init__(self, inputsize, task_info, unitN = 400):
        super(MLP_Net,self).__init__()

        size,_=inputsize
        self.task_info=task_info
        self.relu=torch.nn.ReLU()
        self.drop=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(size*size,unitN)
        self.fc2=torch.nn.Linear(unitN,unitN)
        self.relu = torch.nn.ReLU()

        self.last=torch.nn.ModuleList()
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(unitN,n))

    def forward(self,x):
        h=x.view(x.size(0),-1)
        h=self.drop(self.relu(self.fc1(h)))
        h=self.drop(self.relu(self.fc2(h)))
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](h))

        return y

def conv_net_FiLM(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM(inputsize, task_info)
    return model

#Film layer after last fc layer
def conv_net_FiLM_last(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM_last(inputsize, task_info)
    return model

def conv_net_FiLM_pooling(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM_pooling(inputsize, task_info)
    return model

def conv_net(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net(inputsize, task_info)
    return model

def conv_net_pooling(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_pooling(inputsize, task_info)
    return model

def conv_net_FiLM_wo_fc(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM_wo_fc(inputsize, task_info)
    return model

def conv_net_wo_fc(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_wo_fc(inputsize, task_info)
    return model

def conv_net_FiLM_remember(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM_remember(inputsize, task_info)
    return model

def conv_net_FiLM_remember_wo_batchnorm(task_info):
    inputsize = (3, 32, 32)
    model = Conv_Net_FiLM_remember_wo_batchnorm(inputsize, task_info)
    return model
def MLP(task_info):
    inputsize = (28, 28)
    model = MLP_Net(inputsize, task_info)

    return model



"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
class BasicBlock_FiLM_remember(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, task_info, in_channels, out_channels, stride=1):
        super().__init__()

        self.film1 = torch.nn.ModuleList()
        for t,_ in task_info:
            self.film1.append(FiLM(out_channels))
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FiLM(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
            FiLM(out_channels * BasicBlock_FiLM.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock_FiLM.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
                FiLM(out_channels * BasicBlock_FiLM.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BasicBlock_PiggyBack(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, mask_init, mask_scale, threshold_fn):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nl.ElementWiseConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False,
                mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nl.ElementWiseConv2d(out_channels, out_channels * BasicBlock_PiggyBack.expansion, kernel_size=3, padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn),
            nn.BatchNorm2d(out_channels * BasicBlock_PiggyBack.expansion),
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock_PiggyBack.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nl.ElementWiseConv2d(in_channels, out_channels * BasicBlock_PiggyBack.expansion, kernel_size=1, stride=stride, bias=False, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn),
                nn.BatchNorm2d(out_channels * BasicBlock_PiggyBack.expansion),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
class BasicBlock_FiLM(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FiLM(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
            FiLM(out_channels * BasicBlock_FiLM.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock_FiLM.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
                FiLM(out_channels * BasicBlock_FiLM.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
class ResNet_FiLM_remember(nn.Module):

    def __init__(self, task_info, block, num_block, num_classes=100, init_weights=True):
        super().__init__()

        self.in_channels = 64
        self.task_info = task_info

        self.film1 = torch.nn.ModuleList()
        for t, _ in task_info:
            self.film1.append(FiLM(64))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            self.film1,
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        y = []
        for t, _ in self.task_info:
            y.append(self.last[t](output))
        return y

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet_FiLM(nn.Module):

    def __init__(self, task_info, block, num_block, num_classes=100, init_weights=True):
        super().__init__()

        self.in_channels = 64
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            FiLM(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))

        # weights initialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        y = []
        for t, _ in self.task_info:
            y.append(self.last[t](output))
        return y

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class ResNet_PiggyBack(nn.Module):

    def __init__(self, task_info, block, num_block, mask_init, mask_scale, threshold_fn, num_classes=100, init_weights=True):
        super().__init__()

        self.in_channels = 64
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nl.ElementWiseConv2d(3, 64, kernel_size=3, padding=1, bias=False,
                mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, mask_init, mask_scale, threshold_fn)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, mask_init, mask_scale, threshold_fn)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, mask_init, mask_scale, threshold_fn)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, mask_init, mask_scale, threshold_fn)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))

        # weights initialization
        if init_weights:
            self._initialize_weights()

        for m in self.modules():
            if isinstance(m, nl.ElementWiseConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_channels, num_blocks, stride, mask_init, mask_scale, threshold_fn):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, mask_init, mask_scale, threshold_fn))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        y = []
        for t, _ in self.task_info:
            y.append(self.last[t](output))
        return y

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
class ResNet(nn.Module):

    def __init__(self, task_info, block, num_block, num_classes=100, init_weights=True):
        super().__init__()

        self.in_channels = 64
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))

        # weights initialization
        if init_weights:
            self._initialize_weights()


    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        y = []
        for t,_ in self.task_info:
            y.append(self.last[t](output))

        return y

    # define weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def resnet18_FiLM(task_info):
    """ return a ResNet 18 object
    """
    return ResNet_FiLM(task_info, BasicBlock_FiLM, [2, 2, 2, 2])
def resnet18(task_info):
    """ return a ResNet 18 object
    """
    return ResNet(task_info, BasicBlock, [2, 2, 2, 2])

def resnet18_piggyback(task_info):
    mask_init='1s'
    mask_scale=1e-2
    threshold_fn = 'binarizer'
    return ResNet_PiggyBack(task_info, BasicBlock_PiggyBack, [2,2,2,2], mask_init, mask_scale, threshold_fn)

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

class VGG(nn.Module):

    def __init__(self, task_info, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.task_info = task_info


        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            #nl.ElementWiseLinear(
            #    512 * 4 * 4, 4096, mask_init=mask_init, mask_scale=mask_scale,
            #    threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #nl.ElementWiseLinear(
            #    4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
            #    threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout()
        )

        self.last = torch.nn.ModuleList()
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(4096, n))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = []
        for t,_ in self.task_info:
            y.append(self.last[t](x))

        return y
class VGG_PiggyBack(nn.Module):

    def __init__(self, task_info, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG_PiggyBack, self).__init__()
        self.features = features
        self.task_info = task_info


        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            #nl.ElementWiseLinear(
            #    512 * 4 * 4, 4096, mask_init=mask_init, mask_scale=mask_scale,
            #    threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #nl.ElementWiseLinear(
            #    4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
            #    threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout()
        )

        self.last = torch.nn.ModuleList()
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(4096, n))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = []
        for t,_ in self.task_info:
            y.append(self.last[t](x))

        return y

class VGG_FiLM(nn.Module):

    def __init__(self, task_info, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG_FiLM, self).__init__()
        self.features = features
        self.task_info = task_info


        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )

        self.last = torch.nn.ModuleList()
        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(4096, n))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        y = []
        for t,_ in self.task_info:
            y.append(self.last[t](x))

        return y

def make_layers(layer_type, cfg, mask_init, mask_scale, threshold_fn, batch_norm=False):
    layers = []
    in_channels = 3
    if layer_type == 'piggyback':
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nl.ElementWiseConv2d(
                    in_channels, v, kernel_size=3, padding=1,
                    mask_init=mask_init, mask_scale=mask_scale,
                    threshold_fn=threshold_fn)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    if layer_type == 'film':
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), FiLM(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, FiLM(v), nn.ReLU(inplace=True)]
                in_channels = v
    if layer_type == 'original':
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [32,32,'M', 64,64,128,128,128,'M',256,256,256,512,512,512,'M']
}


def vgg16_piggyback(task_info):
    """VGG 16-layer model (configuration "D")."""
    mask_init = '1s'
    mask_scale = 1e-2
    threshold_fn = 'binarizer'
    layer_type = 'piggyback'

    model = VGG_PiggyBack(task_info, make_layers(layer_type, cfg['F'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn)
    return model

def vgg16_film(task_info):
    """VGG 16-layer model (configuration "D")."""

    mask_init = '1s'
    mask_scale = 1e-2
    threshold_fn = 'binarizer'
    layer_type = 'film'
    model = VGG_FiLM(task_info, make_layers(layer_type, cfg['F'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn)
    return model

def vgg16_original(task_info):
    """VGG 16-layer model (configuration "D")."""

    mask_init = '1s'
    mask_scale = 1e-2
    threshold_fn = 'binarizer'
    layer_type = 'original'
    model = VGG(task_info, make_layers(layer_type, cfg['F'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn)
    return model

def vgg16_bn(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn, batch_norm=True),
                mask_init, mask_scale, threshold_fn, **kwargs)
    return model
