
import os
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from dataloader import get_test_dataloader
import torch.backends.cudnn as cudnn
import time
def eval_top1(outputs, label):
    total = outputs.shape[0]
    outputs = torch.softmax(outputs, dim=-1)
    _, pred_y = outputs.data.max(dim=1) # 得到概率
    correct = (pred_y == label).sum().data
    return correct / total

def eval_top5(outputs, label):
    total = outputs.shape[0]
    outputs = torch.softmax(outputs, dim=-1)
    pred_y = np.argsort(-outputs.cpu().numpy())
    pred_y_top5 = pred_y[:,:5]
    correct = 0
    for i in range(total):
        if label[i].cpu().numpy() in pred_y_top5[i]:
            correct += 1
    return correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test')
    parser.add_argument('--cuda', action='store_true', default=False, help =' use GPU?')
    parser.add_argument('--batch-size', default=100, type=int, help = "Batch Size for Test")
    parser.add_argument('--num-workers', default=2, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19','ResNet18','ResNet34',   
                                                       'DenseNet','MobileNetv1','MobileNetv2'], default='ResNet152', help='net type')
    args = parser.parse_args()
    testloader = get_test_dataloader(batch_size=1)
    # Model
    print('==> Building model..')
    if args.net == 'VGG16':
        from nets.VGG import VGG
        net = VGG('VGG16')
    elif args.net == 'VGG19':
        from nets.VGG import VGG
        net = VGG('VGG19')
    elif args.net == 'ResNet18':
        from nets.ResNet import ResNet18
        net = ResNet18()
    elif args.net == 'ResNet34':
        from nets.ResNet import ResNet34
        net = ResNet34()

    elif args.net == 'ResNet50':
        from nets.ResNet import ResNet50
        net = ResNet50()

    elif args.net == 'ResNet101':
        from nets.ResNet import ResNet101
        net = ResNet101()
    elif args.net == 'ResNet152':
        from nets.ResNet import ResNet152
        net = ResNet152()

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
        cudnn.benchmark = True
    else:
        device = 'cpu'
        net = net.to(device)
        net = torch.nn.DataParallel(net)

   # device = torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net), map_location=device)
    net.load_state_dict(checkpoint['net'])

    
    epoch_step_test = len(testloader)
    if epoch_step_test == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集，或者减小batchsize")
        
    net.eval()
    test_acc_top1 = 0
    test_acc_top5 = 0
    print('Start Test')
    #--------------------------------
    #   相同方法，同train
    #--------------------------------
    start = time.clock()
    #fileObject0 = open("exit_location_0.txt", "a")
   # fileObject1 = open("exit_location_1.txt", "a")
   # fileObject2 = open("exit_location_2.txt", "a")
   # fileObject3 = open("exit_location_3.txt", "a")
   # fileObject4 = open("exit_location_4.txt", "a")

    with tqdm(total=epoch_step_test,desc=f'Test Acc',postfix=dict,mininterval=0.3) as pbar2:
        for step,(im,label) in enumerate(testloader,start=0):
            im = im.to(device)
            label = label.to(device)
            with torch.no_grad():
                if step >= epoch_step_test:
                    break
                
                # 释放内存
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   前向传播
                #----------------------#
                outputs = net(im,0).to(device)
#               location_exit = outputs[1]
#               outputs =outputs[0]
                ss = criterion(outputs,label)
                test_acc_top1 += eval_top1(outputs,label)
                test_acc_top5 += eval_top5(outputs,label)
                pbar2.set_postfix(**{'Test Acc Top1': test_acc_top1.item()/(step+1),
                            'Test Acc Top5': test_acc_top5 / (step + 1)})
                pbar2.update(1)
                fileObject1 = open("eixt-acc-resnet-152-top1-0.1.txt", "a")
                fileObject5 = open("eixt-acc-resnet-152-top5-0.1.txt", "a")
                fileObject1.write(str(eval_top1(outputs, label).item()))
                fileObject1.write('\n')
                fileObject5.write(str(eval_top5(outputs,label)))
                fileObject5.write('\n')
                fileObject1.close()
                fileObject5.close()
    top1 = test_acc_top1.item()/ len(testloader)
    top5 = test_acc_top5 / len(testloader)
    print("top-1 accuracy = %.2f%%" % (top1*100))
    print("top-5 accuracy = %.2f%%" % (top5*100))
    end = time.clock()
    runTime = end - start
    print("运行时间：", runTime)
