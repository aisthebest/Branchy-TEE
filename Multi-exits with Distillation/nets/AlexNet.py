'''
AlexNet in Pytorch
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# 定义2012的AlexNet
class AlexNet(nn.Module):
    def __init__(self,num_classes=10,threshhold = [0.1,0.1,0.1]):
        super(AlexNet,self).__init__()
        self.threshold = threshhold
        # 五个卷积层 输入 32 * 32 * 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),   # (32-3+2)/1+1 = 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (32-2)/2+1 = 16
        )
        self.conv2 = nn.Sequential(  # 输入 16 * 16 * 6
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=1),  # (16-3+2)/1+1 = 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (16-2)/2+1 = 8
        )
        self.conv3 = nn.Sequential(  # 输入 8 * 8 * 16
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), # (8-3+2)/1+1 = 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (8-2)/2+1 = 4
        )
        self.conv4 = nn.Sequential(  # 输入 4 * 4 * 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # (4-3+2)/1+1 = 4
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (4-2)/2+1 = 2
        )
        self.conv5 = nn.Sequential(  # 输入 2 * 2 * 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),# (2-3+2)/1+1 = 2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # (2-2)/2+1 = 1
        )                            # 最后一层卷积层，输出 1 * 1 * 128
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,num_classes)
        )

        self.exit_1 = nn.Sequential(

            nn.Linear(8*8*16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

        self.exit_2 = nn.Sequential(
            nn.Linear(4 * 4 * 32, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

        self.exit_3 = nn.Sequential(
            nn.Linear(2 * 2 * 64, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x,location = 0):
        x = self.conv1(x)
        x = self.conv2(x)
        x_1 = self.exit_1(x.view(x.size()[0], -1))
        if self.entrophy(x_1,self.threshold[0]) or location==1:
            fileObject1 = open("distillation-eixt-acc-alexnet-top1-0.2.txt", "a")
            fileObject5 = open("distillation-eixt-acc-alexnet-top5-0.2.txt", "a")
            fileObject1.write('1')
            fileObject5.write('1')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_1

        x = self.conv3(x)
        x_2 = self.exit_2(x.view(x.size()[0], -1))

        if self.entrophy(x_2,self.threshold[1]) or location==2:
            fileObject1 = open("distillation-eixt-acc-alexnet-top1-0.2.txt", "a")
            fileObject5 = open("distillation-eixt-acc-alexnet-top5-0.2.txt", "a")
            fileObject1.write('2')
            fileObject5.write('2')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            return x_2

        x = self.conv4(x)

        x_3= self.exit_3(x.view(x.size()[0], -1))

        if self.entrophy(x_3,self.threshold[2])or location==3:
            fileObject1 = open("distillation-eixt-acc-alexnet-top1-0.2.txt", "a")
            fileObject5 = open("distillation-eixt-acc-alexnet-top5-0.2.txt", "a")
            fileObject1.write('3')
            fileObject5.write('3')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()

            return x_3

        x = self.conv5(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        fileObject1 = open("distillation-eixt-acc-alexnet-top1-0.2.txt", "a")
        fileObject5 = open("distillation-eixt-acc-alexnet-top5-0.2.txt", "a")
        fileObject1.write('0')
        fileObject5.write('0')
        fileObject1.write('\t')
        fileObject5.write('\t')
        fileObject1.close()
        fileObject5.close()

        return x

    def entrophy(self,x,threshold =0):
        fileObject = open("eixt-0.txt", "a")

        x = F.softmax(x)
        x = x.cpu().detach().numpy()[0]
        log_probs = np.log(x)
        entropy = -1 * np.sum(x * log_probs) / np.log(10)
        fileObject.write(str(entropy))
        fileObject.write('\n')

        fileObject.close()
        if entropy < threshold:
            return True
        else:
            return False
'''
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size()[0],-1)
        x = self.fc(x)
        return x
'''
if __name__ == '__main__':
    net = AlexNet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(1,3,32,32))