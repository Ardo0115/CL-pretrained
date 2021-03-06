import torch
import torch.nn as nn
import numpy as np
import networks.piggyback_layers as nl
import math
from torch.nn.functional import relu, avg_pool2d

BN_MOMENTUM=0.05
BN_AFFINE=True

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

def MLP(task_info):
    inputsize = (28, 28)
    model = MLP_Net(inputsize, task_info)

    return model



def resnet18(task_info):
    """ return a ResNet 18 object
    """
    return ResNet(task_info, BasicBlock, [2, 2, 2, 2])

def resnet18_LMC(task_info):
    nclasses=100
    nf=20
    config={'dropout': 0.05}
    net = ResNet_LMC(task_info, BasicBlock_LMC, [2, 2, 2, 2], nclasses, nf, config=config)
    return net

def resnet18_small(task_info):
    """ return a ResNet 18 object
    """
    return ResNet_Small(task_info, BasicBlock_Small, [2, 2, 2, 2])

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv3x3_LMC(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock_LMC(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, config={}):
        super(BasicBlock_LMC, self).__init__()
        self.conv1 = conv3x3_LMC(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3_LMC(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes, affine=False, track_running_stats=False, momentum=BN_MOMENTUM)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out
class BasicBlock_Small(nn.Module):
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

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, task_info, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.task_info = task_info
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()
        for t, n in self.task_info:
            self.last.append(torch.nn.Linear(512 * block.expansion, n))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = []
        for t, _ in self.task_info:
            y.append(self.last[t](x))
        return y

    def forward(self, x):
        return self._forward_impl(x)

class ResNet_LMC(nn.Module):
    def __init__(self, task_info, block, num_blocks, num_classes, nf, config={}):
        super(ResNet_LMC, self).__init__()
        self.in_planes = nf
        self.save_acts = False
        self.acts = {}
        self.task_info = task_info

        self.conv1 = conv3x3_LMC(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, affine=BN_AFFINE, track_running_stats=False, momentum=BN_MOMENTUM)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, config=config)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, config=config)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, config=config)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, config=config)
        # self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)
        self.last = torch.nn.ModuleList()
        for t, n in self.task_info:
            self.last.append(torch.nn.Linear(nf * 8 * block.expansion, n))

    def _make_layer(self, block, planes, num_blocks, stride, config):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, config=config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        if self.save_acts:
            self.acts['block 1'] = out.detach().clone()

        out = self.layer2(out)
        if self.save_acts:
            self.acts['block 2'] = out.detach().clone()

        out = self.layer3(out)
        if self.save_acts:
            self.acts['block 3'] = out.detach().clone()

        out = self.layer4(out)
        if self.save_acts:
            self.acts['block 4'] = out.detach().clone()

        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = []
        for t, _ in self.task_info:
            y.append(self.last[t](out))
        return y
class ResNet_Small(nn.Module):

    def __init__(self, task_info, block, num_block, num_classes=100, init_weights=True):
        super().__init__()
        nf = 20
        self.in_channels = nf
        self.task_info = task_info

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, nf, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(nf),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, nf, num_block[0], 1)
        self.conv3_x = self._make_layer(block, nf * 2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, nf * 4, num_block[2], 2)
        self.conv5_x = self._make_layer(block, nf * 8, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last = torch.nn.ModuleList()

        for t,n in self.task_info:
            self.last.append(torch.nn.Linear(nf * 8 * block.expansion, n))

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
def _resnet(task_info, block, layers, **kwargs):
    model = ResNet(task_info, block, layers, **kwargs)
    return model


def resnet18(task_info, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(task_info, BasicBlock, [2, 2, 2, 2],
                   **kwargs)


