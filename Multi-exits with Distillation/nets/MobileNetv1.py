'''
MobileNetv1 in pytorch

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

exit_point_v1 = [1,3,6,9]

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self,in_channels,out_channels,stride=1):
        super(Block,self).__init__()
        # groups参数就是深度可分离卷积的关键
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,
                               padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
    def forward(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x
        
# 深度可分离卷积 DepthWise Separable Convolution
class MobileNet(nn.Module):
    # (128,2) means conv channel=128, conv stride=2, by default conv stride=1
    cfg = [64,(128,2),128,(256,2),256,(512,2),512,512,512,512,512,(1024,2),1024]
    
    def __init__(self, num_classes=10,alpha=1.0,beta=1.0,init_weights=True, threshhold = [0.1,0.1,0.1,0.1]):
        super(MobileNet,self).__init__()
        self.threshold = threshhold

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.avg = nn.AvgPool2d(kernel_size=2)
        self.layers = self._make_layers(in_channels=32)
        self.linear = nn.Linear(1024,num_classes)

        self.exit_1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.Linear(1024, num_classes)
        )
        self.exit_2 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Linear(512, num_classes)
        )
        self.exit_3 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.Linear(512, num_classes)
        )
        self.exit_4 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.Linear(512, num_classes)
        )
    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x,int) else x[0]
            stride = 1 if isinstance(x,int) else x[1]
            layers.append(Block(in_channels,out_channels,stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
        if init_weights:
            self._initialize_weights()

    def forward(self,x,location = 0):
        x = self.conv1(x)
        x = self.layers[0:exit_point_v1[0] + 1](x)
        x_1 = self.avg(x)
        x_1 = self.exit_1(x_1.view(x_1.size()[0], -1))
        if self.entrophy(x_1, self.threshold[0]) or location == 1:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-v1-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-v1-top5-0.1.txt", "a")
            fileObject1.write('1')
            fileObject5.write('1')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_1

        x = self.layers[exit_point_v1[0]+1:exit_point_v1[1]+1](x)
        x_2 = self.avg(x)
        x_2 = self.exit_2(x_2.view(x_2.size()[0], -1))
        if self.entrophy(x_2, self.threshold[0]) or location == 2:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-v1-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-v1-top5-0.1.txt", "a")
            fileObject1.write('2')
            fileObject5.write('2')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_2

        x = self.layers[exit_point_v1[1] + 1:exit_point_v1[2] + 1](x)
        x_3 = self.avg(x)
        x_3 = self.exit_3(x_3.view(x_3.size()[0], -1))
        if self.entrophy(x_3, self.threshold[0]) or location == 3:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-v1-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-v1-top5-0.1.txt", "a")
            fileObject1.write('3')
            fileObject5.write('3')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_3

        x = self.layers[exit_point_v1[2] + 1:exit_point_v1[3] + 1](x)
        x_4 = self.avg(x)
        x_4 = self.exit_4(x_4.view(x_4.size()[0], -1))
        if self.entrophy(x_4, self.threshold[0]) or location == 4:
            # print('exit_1\n')
            fileObject1 = open("eixt-acc-v1-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-v1-top5-0.1.txt", "a")
            fileObject1.write('4')
            fileObject5.write('4')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_4
        x = self.layers[exit_point_v1[3] + 1:](x)
        x = self.avg(x)
        x = x.view(x.size()[0],-1)
        x = self.linear(x)
        fileObject1 = open("eixt-acc-v1-top1-0.1.txt", "a")
        fileObject5 = open("eixt-acc-v1-top5-0.1.txt", "a")
        fileObject1.write('0')
        fileObject5.write('0')
        fileObject1.write('\t')
        fileObject5.write('\t')
        fileObject1.close()
        fileObject5.close()
        return x

    def _initialize_weights(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)

    def entrophy(self, x, threshold=0):
        fileObject = open("eixt-0.txt", "a")

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

if __name__ == '__main__':
    net = MobileNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(1,3,32,32))
    
# test()