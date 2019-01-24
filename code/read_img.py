# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:15:06 2018

@author: Administrator
"""

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import xlrd
from skimage import io,transform
import glob
import numpy as np

import os
w=224
h=224
c=3

#读取图片
#def read_img(path1,path2):
#    imgs=[] 
#    labels=[]
#    for im in glob.glob(path1+'/*.jpg'):
#            #glob.glob(folder+'/*.jpg')是path1文件夹底下任何一个文件夹里面图片的根目录；eg：D:\\lower_photos\\tulips\\9976515506_d496c5e72c.jpg'
#            print('reading the images:%s'%(im))
#            img=io.imread(im)                 #读图，img为RGB的3通道数组
#            img=transform.resize(img,(w,h))   #img为200*200的3通道数组
##            img = img.permute(2,1,0)
#            imgs.append(img)
#    count1=0
#    for elem1 in imgs:
#        count1+=1
#    print("imgs 中有%d个元素"%count1)       
#    bk = xlrd.open_workbook(path2)
#    #shxrange = range(bk.nsheets)
#    try:
#        sh = bk.sheet_by_name("Sheet1")
#    except:
#        print("no sheet in %s named Sheet1" % path2)
#    #获取行数
#    nrows = sh.nrows
#    #获取列数
#    #ncols = sh.ncols
#    #获取各行数据
##    for i in range(0,nrows):              #16年数据用这个
#    for i in range(1,nrows):             #17年数据用这个
#        row_data = sh.row_values(i)
#        labels.append(row_data[1])
#    print(labels)
#    count2=0
#    for elem2 in labels:
#        count2+=1
#    print("imgs 中有%d个元素"%count1)
#    print("labels 中有%d个元素"%count2)
#    return np.asarray(imgs,np.float32),np.asarray(labels,np.int64)


'''2019.01.14 猫狗大战读取图片，按照文件夹顺序读取图片'''
#读取图片
def read_img(path):
    cate=[os.path.join(path,x) for x in os.listdir(path) if os.path.isdir(os.path.join(path,x))]
    print(cate)#zkz2018/11/09
    #os.listdir(path)找到path下的文件名；os.path.isdir(os.path.join(path,x))判断path下的x是否为文件夹；
    #os.path.join(path,x)把x加到path文件夹下，cate返回的也是这个，是根目录，eg：'D:\\lower_photos\\daisy'
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        #enumerate(cate)遍历了索引又遍历了元素
        #idx是一个数组，前633个数全是0，从634开始是1，有898个1：同理，接下来是2，3，4
        #folder是path文件夹底下5个文件的根目录
        for im in glob.glob(folder+'/*.jpg'):
            #glob.glob(folder+'/*.jpg')是path文件夹底下任何一个文件夹里面图片的根目录；eg：D:\\lower_photos\\tulips\\9976515506_d496c5e72c.jpg'
            print('reading the images:%s'%(im))
            img=io.imread(im)                 #读图，img为RGB的3通道数组
            img=transform.resize(img,(w,h))   #img为100*100的3通道数组
            imgs.append(img)                  #把img这个图片数组加到imgs里面
            labels.append(idx)                #把idx这个索引数组加到labels里面，当作标签
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int64)
