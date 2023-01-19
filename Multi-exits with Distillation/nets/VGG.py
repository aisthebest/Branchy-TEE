'''
VGG11,13,16,19 in pytorch
'''

#from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
exit_point_19 = [6,13,26,39]
class VGG(nn.Module):
    def __init__(self, vggname = 'VGG16',num_classes=10, init_weights=True,threshhold = [0.3,0.3,0.3,0.3]):
        super(VGG,self).__init__()
        self.threshold = threshhold
        self.features = self._make_layers(cfg[vggname])
        self.classifier = nn.Linear(512,num_classes)

        self.exit_1 = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),
            nn.Linear(512,num_classes)
        )
        self.exit_2 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.Linear(512, num_classes)
        )
        self.exit_3 = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.Linear(512, num_classes)
        )
        self.exit_4 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.Linear(512, num_classes)
        )
    def _make_layers(self,cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M': # 最大池化层
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
            else:
                layers += [nn.Conv2d(in_channels,out_channels=x,kernel_size=3,padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1,stride=1)]
        return nn.Sequential(*layers)
        if init_weights:
            self. _initialize_weight()
    def forward(self,x,location = 0):
        x = self.features[0:exit_point_19[0]+1](x)
        x_1 = self.exit_1(x.view(x.size()[0], -1))
        if self.entrophy(x_1,self.threshold[0]) or location==1:
            #print('exit_1\n')
            '''
            fileObject1 = open("eixt-acc-vgg-19-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-vgg-19-top5-0.1.txt", "a")
            fileObject1.write('1')
            fileObject5.write('1')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            '''
            return x_1

        x = self.features[exit_point_19[0]+1:exit_point_19[1]+1](x)
        x_2 = self.exit_2(x.view(x.size()[0], -1))
        if self.entrophy(x_2, self.threshold[1]) or location == 2:
            # print('exit_1\n')
            '''
            fileObject1 = open("eixt-acc-vgg-19-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-vgg-19-top5-0.1.txt", "a")
            fileObject1.write('2')
            fileObject5.write('2')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            '''
            return x_2
        x = self.features[exit_point_19[1] + 1:exit_point_19[2] + 1](x)
        x_3 = self.exit_3(x.view(x.size()[0], -1))
        if self.entrophy(x_3, self.threshold[2]) or location == 3:
            '''
            fileObject1 = open("eixt-acc-vgg-19-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-vgg-19-top5-0.1.txt", "a")
            fileObject1.write('3')
            fileObject5.write('3')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            '''
            return x_3

        x = self.features[exit_point_19[2] + 1:exit_point_19[3] + 1](x)
        x_4 = self.exit_4(x.view(x.size()[0], -1))
        if self.entrophy(x_4, self.threshold[3]) or location == 4:
            '''
            fileObject1 = open("eixt-acc-vgg-19-top1-0.1.txt", "a")
            fileObject5 = open("eixt-acc-vgg-19-top5-0.1.txt", "a")
            fileObject1.write('4')
            fileObject5.write('4')
            fileObject1.write('\t')
            fileObject5.write('\t')
            fileObject1.close()
            fileObject5.close()
            '''
            return x_4

        x = self.features[exit_point_19[3] + 1:](x)
        x = x.view(x.size()[0],-1)
        x = self.classifier(x)
        '''
        fileObject1 = open("eixt-acc-vgg-19-top1-0.1.txt", "a")
        fileObject5 = open("eixt-acc-vgg-19-top5-0.1.txt", "a")
        fileObject1.write('0')
        fileObject5.write('0')
        fileObject1.write('\t')
        fileObject5.write('\t')
        fileObject1.close()
        fileObject5.close()
        '''
        return x
    # 初始化参数
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # xavier is used in VGG's paper
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
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
if __name__ == '__main__':
    net = VGG('VGG19')
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())
    from torchinfo import summary
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    summary(net,(1,3,32,32))
    
# test()