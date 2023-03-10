'''
ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """
    对于浅层网络，如ResNet-18/34等，用基本的Block
    基础模块没有压缩,所以expansion=1
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        # 如果输入输出维度不等，则使用1x1卷积层来改变维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
    def forward(self, x):
        out = self.features(x)
#         print(out.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out
    

class Bottleneck(nn.Module):
    """
    对于深层网络，我们使用BottleNeck，论文中提出其拥有近似的计算复杂度，但能节省很多资源
    zip_channels: 压缩后的维数，最后输出的维数是 expansion * zip_channels
    针对ResNet50/101/152的网络结构,主要是因为第三层是第二层的4倍的关系所以expansion=4
    """
    expansion = 4

    def __init__(self, in_channels, zip_channels, stride=1):
        super(Bottleneck, self).__init__()
        out_channels = self.expansion * zip_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, zip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, zip_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(zip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(zip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.features(x)
#         print(out.shape)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    """
    不同的ResNet架构都是统一的一层特征提取、四层残差，不同点在于每层残差的深度。
    对于cifar10，feature map size的变化如下：
    (32, 32, 3) -> [Conv2d] -> (32, 32, 64) -> [Res1] -> (32, 32, 64) -> [Res2]
 -> (16, 16, 128) -> [Res3] -> (8, 8, 256) ->[Res4] -> (4, 4, 512) -> [AvgPool]
 -> (1, 1, 512) -> [Reshape] -> (512) -> [Linear] -> (10)
    """
    def __init__(self, block, num_blocks, num_classes=10, verbose = False,  init_weights=True,threshhold = [0.1,0.1,0.1,0.1]):
        super(ResNet, self).__init__()

        self.exit_1 = nn.Sequential(
            nn.Linear(256 * 8* 8, 512 * block.expansion),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.exit_2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512 * block.expansion),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.exit_3 = nn.Sequential(
            nn.Linear(1024 * 2 * 2, 512 * block.expansion),
            nn.Linear(512 * block.expansion, num_classes)
        )
        self.exit_4 = nn.Sequential(
            nn.Linear(2048 * 1 * 1, 512 * block.expansion),
            nn.Linear(512 * block.expansion, num_classes)
        )

        self.threshold = threshhold

        self.verbose = verbose
        self.in_channels = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        #使用_make_layer函数生成上表对应的conv2_x, conv3_x, conv4_x, conv5_x的结构
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # cifar10经过上述结构后，到这里的feature map size是 4 x 4 x 512 x expansion
        # 所以这里用了 4 x 4 的平均池化
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.classifer = nn.Linear(512 * block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()


    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block要进行降采样
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 如果是Bottleneck Block的话需要对每层输入的维度进行压缩，压缩后再增加维数
            # 所以每层的输入维数也要跟着变
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x,location = 0):
        out = self.features(x)
        if self.verbose:
            print('block 1 output: {}'.format(out.shape))
        out = self.layer1(out)
        x_1 = self.avg_pool(out)
        x_1 = self.exit_1(x_1.view(x.size()[0], -1))

        if self.entrophy(x_1,self.threshold[0]) or location==1:
            #print('exit_1\n')
            fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
            fileObject1.write('1')
            fileObject5.write('1')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_1



        if self.verbose:
            print('block 2 output: {}'.format(out.shape))
        out = self.layer2(out)
        x_2 = self.avg_pool(out)
        x_2 = self.exit_2(x_2.view(x.size()[0], -1))

        if self.entrophy(x_2,self.threshold[1]) or location==2:
            #print('exit_1\n')
            fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
            fileObject1.write('2')
            fileObject5.write('2')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_2


        if self.verbose:
            print('block 3 output: {}'.format(out.shape))
        out = self.layer3(out)
        x_3 = self.avg_pool(out)
        x_3 = self.exit_3(x_3.view(x.size()[0], -1))

        if self.entrophy(x_3, self.threshold[2]) or location == 3:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
            fileObject1.write('3')
            fileObject5.write('3')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_3

        if self.verbose:
            print('block 4 output: {}'.format(out.shape))
        out = self.layer4(out)

        x_4 = self.avg_pool(out)
        x_4 = self.exit_4(x_4.view(x.size()[0], -1))

        if self.entrophy(x_4, self.threshold[3]) or location == 4:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
            fileObject1.write('4')
            fileObject5.write('4')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_4

        if self.verbose:
            print('block 5 output: {}'.format(out.shape))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifer(out)
        fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
        fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
        fileObject1.write('0')
        fileObject5.write('0')
        fileObject1.write('\t')
        fileObject5.write('\t')
        fileObject1.close()
        fileObject5.close()
        return out

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

    def entrophy(self, x, threshold=0):
        #fileObject = open("eixt-0.txt", "a")

        x = F.softmax(x)
        x = x.cpu().detach().numpy()[0]
        log_probs = np.log(x)
        entropy = -1 * np.sum(x * log_probs) / np.log(10)
        #fileObject.write(str(entropy))
        #fileObject.write('\n')

        #fileObject.close()
        if entropy < threshold:
            return True
        else:
            return False
def ResNet18(verbose=False):
    return ResNet(BasicBlock, [2,2,2,2],verbose=verbose)

def ResNet34(verbose=False):
    return ResNet(BasicBlock, [3,4,6,3],verbose=verbose)

def ResNet50(verbose=False):
    return ResNet(Bottleneck, [3,4,6,3],verbose=verbose)

def ResNet101(verbose=False):
    return ResNet(Bottleneck, [3,4,23,3],verbose=verbose)

def ResNet152(verbose=False):
    return ResNet(Bottleneck, [3,8,36,3],verbose=verbose)

if __name__ == '__main__':
    net = ResNet50()
    x = torch.randn(1,3,32,32)
    #y = net(x,4)
    #print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(1,3,32,32))

# test()