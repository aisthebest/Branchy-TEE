'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

import os
import argparse
from utils import get_acc,EarlyStopping
from dataloader import get_test_dataloader, get_training_dataloader
from tqdm import tqdm


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=True, help =' use GPU?')
    parser.add_argument('--batch-size', default=64, type=int, help = "Batch Size for Training")
    parser.add_argument('--num-workers', default=2, type=int, help = 'num-workers')
    parser.add_argument('--net', type = str, choices=['LeNet5', 'AlexNet', 'VGG16','VGG19','ResNet18','ResNet34',   
                                                       'DenseNet','MobileNetv1','MobileNetv2'], default='AlexNet', help='net type')
    parser.add_argument('--epochs', type = int, default=100, help = 'Epochs')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--patience', '-p', type = int, default=7, help='patience for Early stop')
    parser.add_argument('--optim','-o',type = str, choices = ['sgd','adam','adamw'], default = 'adamw', help = 'choose optimizer')

    args = parser.parse_args()
    
    print(args)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Train Data
    trainloader = get_training_dataloader(batch_size = args.batch_size, num_workers = args.num_workers)
    testloader = get_test_dataloader(batch_size = args.batch_size, num_workers = args.num_workers, shuffle=False)
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
        from nets.teacher.AlexNet import AlexNet
        net_T = AlexNet()
    elif args.net == 'DenseNet':
        from nets.DenseNet import densenet_cifar
        net = densenet_cifar()
    elif args.net == 'MobileNetv1':
        from nets.MobileNetv1 import MobileNet
        net = MobileNet()
    elif args.net == 'MobileNetv2':
        from nets.MobileNetv2 import MobileNetV2
        net = MobileNetV2()

    if args.cuda:
        device = 'cuda'
        #device1 = 'cpu'
        #net_T = net_T.to(device1)
        net = torch.nn.DataParallel(net)
        net_T= torch.nn.DataParallel(net_T)
        # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/{}_ckpt.pth'.format(args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        args.lr = checkpoint['lr']


    early_stopping = EarlyStopping(patience = args.patience, verbose=True)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.KLDivLoss()
    if args.optim == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    elif args.optim == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.94,verbose=True,patience = 1,min_lr = 0.000001) # ?????????????????????

    epochs = args.epochs
    def train(epoch):
        epoch_step = len(trainloader)
        if epoch_step == 0:
            raise ValueError("????????????????????????????????????????????????????????????????????????batchsize")
        net.train()
        train_loss = 0
        train_acc = 0
        print('Start Train')

        checkpoint_T = torch.load('./checkpoint/checkpoint-native-dnn/{}_ckpt.pth'.format(args.net), map_location=device)

        net_T.load_state_dict(checkpoint_T['net'])


        net_T.eval()

        alpha = 0.95

        with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar:
            for step,(im,label) in enumerate(trainloader,start=0):
                #device = 'cuda'
                #device1 = 'cpu'
                #im_t = im.to(device1)
                #label_t = label.to(device1)
                im = im.to(device)
                label = label.to(device)
                soft_target = net_T(im)

                #---------------------
                #  ????????????
                #---------------------
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                #----------------------#
                #   ????????????
                #----------------------#
                optimizer.zero_grad()
                #----------------------#
                #   ????????????forward
                #----------------------#
                outputs_1 = net(im,1)
                outputs_2 = net(im, 2)
                outputs_3= net(im, 3)
                outputs_4 = net(im, 4)
                outputs = net(im, 0)


                #----------------------#
                #   ????????????-stu-loss
                #----------------------#
                loss_1 = 0.3*criterion(outputs_1,label)
                loss_2 = 0.3*criterion(outputs_2,label)
                loss_3 = 0.2*criterion(outputs_3,label)
                loss_4 = 0.2 * criterion(outputs_3, label)
                loss = 0.2* criterion(outputs,label)

                loss_s = loss+loss_1+loss_2+loss_3

                T = 4
                outputs_diss_1 = F.log_softmax(outputs_1/T,dim=1)
                outputs_diss_2 = F.log_softmax(outputs_2 / T, dim=1)
                outputs_diss_3 = F.log_softmax(outputs_3 / T, dim=1)
                #outputs_diss_1 = F.log_softmax(outputs_1 / T, dim=1)
                outputs_T = F.softmax(soft_target / T, dim=1)

                loss_t_1 = criterion2(outputs_diss_1, outputs_T) * T * T
                loss_t_2 = criterion2(outputs_diss_2, outputs_T) * T * T
                loss_t_3 = criterion2(outputs_diss_3, outputs_T) * T * T

                loss_t = loss_t_1+loss_t_2+loss_t_3

                loss = loss_s* (1 - alpha) + loss_t * alpha



                train_loss += loss.data
                train_acc += get_acc(outputs,label)
                #----------------------#
                #   ????????????
                #----------------------#
                # backward
                loss.backward()
                # ????????????
                optimizer.step()
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(**{'Train Loss' : train_loss.item()/(step+1),
                                    'Train Acc' :train_acc.item()/(step+1),  
                                    'Lr'   : lr})
                pbar.update(1)
        # train_loss = train_loss.item() / len(trainloader)
        # train_acc = train_acc.item() * 100 / len(trainloader)
        scheduler.step(train_loss)
        print('Finish Train')
    def test(epoch):
        #device = 'cuda'
        #device1 = 'cpu'
        global best_acc
        epoch_step_test = len(testloader)
        if epoch_step_test == 0:
                raise ValueError("????????????????????????????????????????????????????????????????????????batchsize")
        
        net.eval()
        test_loss = 0
        test_acc = 0
        print('Start Test')
        #--------------------------------
        #   ??????????????????train
        #--------------------------------
        with tqdm(total=epoch_step_test,desc=f'Epoch {epoch + 1}/{epochs}',postfix=dict,mininterval=0.3) as pbar2:
            for step,(im,label) in enumerate(testloader,start=0):

                im = im.to(device)
                label = label.to(device)

                with torch.no_grad():
                    if step >= epoch_step_test:
                        break
                    
                    # ????????????
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    #----------------------#
                    #   ????????????
                    #----------------------#
                    outputs = net(im)
                    loss = criterion(outputs,label)
                    test_loss += loss.data
                    test_acc += get_acc(outputs,label)
                    
                    pbar2.set_postfix(**{'Test Acc': test_acc.item()/(step+1),
                                'Test Loss': test_loss.item() / (step + 1)})
                    pbar2.update(1)
        lr = optimizer.param_groups[0]['lr']
        test_acc = test_acc.item() * 100 / len(testloader)
        # Save checkpoint.
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'lr': lr,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}_ckpt.pth'.format(args.net))
            best_acc = test_acc
            
        print('Finish Test')

        #early_stopping(test_loss, net)
        # ????????? early stopping ??????
        if early_stopping.early_stop:
            print("Early stopping")
            # ??????????????????
            exit()
        
    for epoch in range(start_epoch, epochs):
        train(epoch)
        test(epoch)
        
    torch.cuda.empty_cache()
    