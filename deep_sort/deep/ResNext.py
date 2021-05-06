import torch
import torch.nn as nn

class GroupBlock(nn.Module):
    def __init__(self, input_size, stride, cardinality = 32, bottleneck_width = 4, expansion = 2): #to understand *args,**kwargs
        super(GroupBlock, self).__init__()
        group_channels = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(input_size, group_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(group_channels)
        self.conv2 = nn.Conv2d(group_channels, group_channels, kernel_size=3, stride=stride, padding=1, groups=cardinality)
        self.bn2 = nn.BatchNorm2d(group_channels)
        self.conv3 = nn.Conv2d(group_channels, group_channels*expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(group_channels*expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride !=1 or input_size != group_channels*expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(input_size, group_channels*expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(group_channels*expansion)
            )
        
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.relu(self.bn3(self.conv3(output)))
        shortcut = self.shortcut(x)
        output = self.relu(shortcut + output)
        return output

class ResNext(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes = 10):
        super(ResNext, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.input_size = 64
        self.expansion = 2

        self.conv1 = nn.Conv2d(3, 64, kernel_size= 7, stride= 2, padding=3)  #to understand 7x7
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)#论文有，代码没有？

        self.layer1 = self._make_layer_(num_blocks[0], 1)
        self.layer2 = self._make_layer_(num_blocks[1], 2)
        self.layer3 = self._make_layer_(num_blocks[2], 2)
        self.layer4 = self._make_layer_(num_blocks[3], 2) #to understand stride

        self.linear = nn.Linear(self.cardinality*self.bottleneck_width*2, num_classes)
        self.avg_pool = nn.AvgPool2d(kernel_size=2
        )

    def _make_layer_(self, blocks, stride):
        layers = []
        strides = []
        strides = [stride] + [1]*(blocks-1)
        for stride in strides: #to understand difference here between the original(主要是stride不一样)
            layers.append(GroupBlock(self.input_size, stride, self.cardinality, self.bottleneck_width, self.expansion))
            self.input_size = self.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width = 2 * self.bottleneck_width
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.maxpool(self.bn1(self.conv1(x)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.avg_pool(output)#与F.avg_pool2d区别
        #根据debug看一下这里少一个view...其实就是resize一下
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

def ResNext50_32x4d():
    return ResNext(num_blocks = [3,4,6,3], cardinality = 32, bottleneck_width = 4, num_classes=776) #to change num_classes


net = ResNext50_32x4d()
x = torch.randn(4,3,128,64)
y = net(x)


