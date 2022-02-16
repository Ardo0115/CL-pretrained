import torch
import torch.nn as nn
import numpy as np

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

class BasicBlock_FiLM(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            FiLM(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
            FiLM(out_channels * BasicBlock_FiLM.expansion)
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock_FiLM.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock_FiLM.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock_FiLM.expansion),
                FiLM(out_channels * BasicBlock_FiLM.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias=False로 설정합니다.
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )

        # identity mapping, input과 output의 feature map size, filter 수가 동일한 경우 사용.
        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        # projection mapping using 1x1conv
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        self.relu = nn.ReLU()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels*BottleNeck.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x

class ResNet_FiLM(nn.Module):
    def __init__(self, task_info, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        
        self.in_channels=64
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            FiLM(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))


        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](x))

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
    def __init__(self, task_info, block, num_block, num_classes=10, init_weights=True):
        super().__init__()

        
        self.in_channels=64
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))


        # weights inittialization
        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self,x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        x = self.conv3_x(output)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        y = []
        for t,i in self.task_info:
            y.append(self.last[t](x))

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

    def forward(self, x, avg_act = False):
        y = []
        for t, i in self.task_info:
            act1=self.relu(self.film1[t](self.batchnorm1(self.conv1(x)))) # act1 : (256,64,32,32)
            act2=self.relu(self.film2[t](self.batchnorm2(self.conv2(act1)))) # act2 : (256,64,32,32)
            act3=self.relu(self.film3[t](self.batchnorm3(self.conv3(act2)))) # act3 : (256,64,32,32)
            act4=self.relu(self.film4[t](self.batchnorm4(self.conv4(act3)))) # act4 : (256,64,32,32)

            h=act4.view(x.shape[0],-1) # h : (256, 65536)
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

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34(task_info):
    model = ResNet(task_info, BasicBlock, [3,4,6,3])
    return model
def resnet34_FiLM(task_info):
    model = ResNet_FiLM(task_info, BasicBlock_FiLM, [3,4,6,3])
    return model

def resnet50(task_info):
    model = ResNet(task_info, BottleNeck, [3,4,6,3])
    return model

def resnet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])

def MLP(task_info):
    inputsize = (28, 28)
    model = MLP_Net(inputsize, task_info)

    return model

