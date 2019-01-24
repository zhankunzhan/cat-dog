from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import os
import torch.nn.functional as F
import pretrainedmodels

import xlrd
from skimage import io,transform
import glob
import numpy as np

from tkinter import _flatten
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import time
from read_img import read_img

from minibatches import minibatches
from data_enhancement import *
#2018.12.11tensorboard画图
from tensorboardX import SummaryWriter
writer = SummaryWriter('log')

'''以下为17年数据路径'''
path1_1="../data/training224"
path2="../data/YT2000.xls"
path3_3="../data/test224"
path4="../data/CS600.xls"

'''2019.01.14猫狗大战路径'''
path = '../data/catdogMinMin/'
def one_hot(values):
    n_values = 2
    return np.eye(n_values)[values]

def sp_fn(labels,forecast):
    m=0
    i=0
    k=0
    for a in labels:
        if labels[i] == 0 :
            m=m+1
            if forecast[i] == 0 :
                k=k+1
        i=i+1
    sp=k/m
    return sp

def AAA(a):
    aa=[]
    for i in a:        
        aa.append(i[1])
    return aa


def train_model(model, criterion, optimizer, scheduler,batch_size = 8, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
#            else:
#                model.train(False)  # Set model to evaluate mode
            
            """每走完一个大循环，将所有训练集数据打乱一次（每次的训练集都被打乱，输入的epoch改变）"""
            #打乱顺序
            num=data_train.shape[0]     #所有图片的数量，小版本数据集为455张图
            arry=np.arange(num)    #把0-454这些数在一个矩阵内按顺序分布
            np.random.shuffle(arry)        #把这454个数在矩阵内随机打乱，即此时arr这个矩阵内的455个数已经打乱
            x_train=data_train[arry]                #data每一个小图片矩阵都按照arr里面的排列顺序进行排列
            y_train=label_train[arry]              #label里面的每一个标签按照arr排列，与data匹配

            
            running_loss = 0.0
            running_corrects = 0
            
            for data, target in minibatches(data_train, label_train, batch_size, shuffle=True):
                
                if use_gpu:
                    data, target = torch.from_numpy(data).cuda(), torch.from_numpy(target).cuda()
                else:
                    data, target = torch.from_numpy(data), torch.from_numpy(target)
                
                data, target = Variable(data), Variable(target) #io.imshow(data[0].numpy())
                data = data.permute(0,3,1,2)    #io.imshow(data0[0].permute(1,2,0).numpy())

                optimizer.zero_grad()


                outputs = model(data)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, target)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == target.data)
                         
            epoch_loss = float(running_loss) / float(dataset_sizes)
            epoch_acc = float(running_corrects) / float(dataset_sizes)


            print('{} Loss: {:.6f} Acc: {:.5f}'.format(
                phase, epoch_loss, epoch_acc))
            
            
        hh1=[]
        jj1=[]
        kk1=[]
        mm1=[]
        # Each epoch has a training and validation phase
        for phase in ['val']:
            if phase == 'val':
                model.train(False)  # Set model to training mode
#            else:
#                model.train(False)  # Set model to evaluate mode

            loss_val = 0.0
            corrects_val = 0
            
            for x_data, y_target in minibatches(x_val, y_val, batch_size, shuffle=True):
                if use_gpu:
                    x_data, y_target = torch.from_numpy(x_data).cuda(), torch.from_numpy(y_target).cuda()
                else:
                    x_data, y_target = torch.from_numpy(x_data), torch.from_numpy(y_target)
                
                x_data, y_target = Variable(x_data), Variable(y_target) #io.imshow(data[0].numpy())
                x_data = x_data.permute(0,3,1,2)    #io.imshow(data0[0].permute(1,2,0).numpy())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs_val = model(x_data)
                outputs_val_softmax = F.softmax(outputs_val)
                _, preds_val = torch.max(outputs_val.data, 1)
                loss_val = criterion(outputs_val, y_target)

                # statistics
                loss_val += loss_val.data[0]
                corrects_val += torch.sum(preds_val == y_target.data)
                
                h1 = np.asarray(y_target).tolist()
                j1 = list(_flatten(np.asarray(preds_val).tolist()))
#                k1 = h1
                m1 = torch.tensor(outputs_val_softmax).cpu()
                m1 = m1.detach().numpy().tolist()
                
                hh1.extend(h1)
                jj1.extend(j1)
                mm1.extend(m1)
        
                
            epoch_val_loss = float(loss_val) / float(x_val_sizes)
            epoch_val_acc = float(corrects_val) / float(x_val_sizes)

            print('{} Loss: {:.6f} Acc: {:.5f}'.format(
                phase, epoch_val_loss, epoch_val_acc))
            
            kk1 = np.asarray(one_hot(hh1))
#            mm1 = np.asarray(AAA(mm1))
            mm1 = np.asarray(mm1)
            
            SE1=metrics.recall_score(hh1,jj1)
            SP1=sp_fn(hh1,jj1) 
        
            AUC=roc_auc_score(kk1,mm1)   
            AP1=average_precision_score(kk1,mm1)       
            
            print(" SE1: %f" % SE1)
            print(" SP1: %f" % SP1)
            print("AUC1: %f" % AUC)
            print(" AP1: %f" % AP1)
            #2018.12.11 画图
            writer.add_scalar('val/Loss', epoch_val_loss,num_epochs)
            
            print(time.strftime("%b %d %Y %H:%M:%S",time.localtime()))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights   
    #path = "E:\\陈枫\\pytorch\\moxin_baocun\\CF"
#    torch.save(model.load_state_dict(model.state_dict()), path)
#    torch.save(model.state_dict(), path)
    return model


if __name__ == '__main__':
    '''2019.01.14原来的初始化'''
#    data_train, label_train = read_img(path1_1,path2)
#    dataset_sizes = len(data_train)
#    
#    x_val , y_val = read_img(path3_3,path4)
#    x_val_sizes = len(x_val)
   
    '''2019.01.14猫狗大战的代码'''
    data,label=read_img(path)

    num_example=data.shape[0]     #所有图片的数量，数据集为3670张图
    arr=np.arange(num_example)    #把0-3669这些数在一个矩阵内按顺序分布
    np.random.shuffle(arr)        #把这3670个数在矩阵内随机打乱，即此时arr这个矩阵内的3670个数已经打乱
    datax=data[arr]                #data每一个小图片矩阵都按照arr里面的排列顺序进行排列
    labelx=label[arr]              #label里面的每一个标签按照arr排列，与data匹配
    
    #将所有数据分为训练集和验证集
    ratio=0.8
    s=np.int(num_example*ratio)  #s=2936
    data_train=datax[:s]             #data的前2936个图片矩阵
    label_train=labelx[:s]            #同理
    dataset_sizes = len(data_train)
    x_val=datax[s:]               #data的后734个图片矩阵
    y_val=labelx[s:]              #同理
    x_val_sizes = len(x_val)
    
    
    
    # use gpu or not
    use_gpu = torch.cuda.is_available()
    '''其他系列模式 2019.01.12'''
    # get model and replace the original fc layer with your fc layer
    model_ft = pretrainedmodels.alexnet(num_classes=1000, pretrained='imagenet')

    dim_feats = model_ft.last_linear.in_features # =2048
    nb_classes = 2
    model_ft.last_linear = nn.Linear(dim_feats, nb_classes).cuda()
    
    '''dpn系列使用以下方式 2019.01.12'''
    # get model and replace the original fc layer with your fc layer
#    model_ft = pretrainedmodels.dpn68(num_classes=1000)
#    dim_feats = model_ft.last_linear.in_channels
#    nb_classes = 2
#    model_ft.last_linear = nn.Conv2d(dim_feats, nb_classes, kernel_size=1, bias=True)
    
    
    if use_gpu:
        model_ft = model_ft.cuda()

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.2)

    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           batch_size =1,
                           num_epochs=155)





















