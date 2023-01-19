import os
import torch
import torch.nn as nn
import argparse
import time
#import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import datasets, transforms
import torch.nn.functional as F

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    start = time.perf_counter()
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--cuda', action='store_true', default=False, help=' use GPU?')
    parser.add_argument('--batch-size', default=64, type=int, help="Batch Size for Test")
    parser.add_argument('--num-workers', default=2, type=int, help='num-workers')
    parser.add_argument('--net', type=str, choices=['LeNet5', 'AlexNet', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34',
                                                    'DenseNet', 'MobileNetv1', 'MobileNetv2'], default='VGG19',
                        help='net type')
    args = parser.parse_args()

    # Model
    print('==> Building model..')
    if args.net == 'VGG16':
        from VGG import VGG

        net = VGG('VGG16')
    elif args.net == 'VGG19':
        from VGG import VGG

        net = VGG('VGG19')
    elif args.net == 'ResNet18':
        from nets.ResNet import ResNet18

        net = ResNet18()
    elif args.net == 'ResNet34':
        from nets.ResNet import ResNet34

        net = ResNet34()
    elif args.net == 'LeNet5':
        from nets.LeNet5 import LeNet5

        net = LeNet5()
    elif args.net == 'AlexNet':
        from nets.AlexNet import AlexNet

        net = AlexNet()
    elif args.net == 'DenseNet':
        from nets.DenseNet import densenet_cifar

        net = densenet_cifar()
    elif args.net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet

        net = MobileNet()
    elif args.net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2

        net = MobileNetV2()

    if args.cuda and torch.cuda.is_available():
        device = 'cuda'
        net = torch.nn.DataParallel(net)
    else:
        device = 'cpu'
        net = net.to(device)
        net = torch.nn.DataParallel(net)

    criterion = nn.CrossEntropyLoss()

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net), map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    img = Image.open("./airplane.jpg").convert('RGB')  # 读取图像
    trans = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                     std=(0.5, 0.5, 0.5)),
                                ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    # 扩展后，为[1，1，28，28]
    output = net(img).to(device)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    print("概率", prob)
    value, predicted = torch.max(output.data, 1)
    print("类别", predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print("分类", pred_class)

    end = time.perf_counter()
    runTime = end - start
    runTime_ms = runTime * 1000
    # 输出运行时间
    print("运行时间：", runTime, "秒")
    print("运行时间：", runTime_ms, "毫秒")
