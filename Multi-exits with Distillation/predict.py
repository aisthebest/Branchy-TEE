import os
import torch
import torch.nn as nn
import argparse
import time
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
    if args.net == 'VGG19':
        from nets.VGG import VGG

        net = VGG('VGG19')

    elif args.net == 'AlexNet':
        from nets.AlexNet import AlexNet

        net = AlexNet()

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
    load = time.perf_counter()
    runTime = load - start

    print("模型加载时间：", runTime, "秒") 
    output = net(img)
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
