# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:16:12 2018

@author: Administrator
"""

#对训练集0进行90，80，270的旋转增强
def Augument0(data,label):
    label0=[]
    data0 =[]

    i=0
    for j in label:
#        if j == 1:
        label0.append(j)
        data0.append(data[i])
        i+=1
    rotate90=[]
    rotate180=[]
    rotate270=[]
    label_90=label0
    label_180=label0
    label_270=label0
    for jj in data0:
        rotate90.append(transform.rotate(jj, 90))
        rotate180.append(transform.rotate(jj, 180))
        rotate270.append(transform.rotate(jj, 270))
    data=list(data)
    label=list(label)
    data.extend(rotate90)  
    data.extend(rotate180) 
    data.extend(rotate270)
    label.extend(label_90)
    label.extend(label_180)
    label.extend(label_270)
    return np.asarray(data,np.float32), np.asarray(label,np.int64)    
#data_train,label_train=Augument0(data_train,label_train)
    
def fliplr_img(data,label):
    label0=[]
    data0 =[]

    i=0
    for j in label:
#        if j == 1:
        label0.append(j)
        data0.append(data[i])
        i+=1
    fli=[]
    for jj in data0:
        fli.append(np.fliplr(jj))
    data=list(data)
    label=list(label)
    data.extend(fli)
    label.extend(label0)
    return np.asarray(data,np.float32), np.asarray(label,np.int64)
#data_train,label_train=fliplr_img(data_train,label_train)


def add_zs(data,label):
    label0=[]
    data0 = data.copy()
    i=0
    for j in label:
    #   if j == 1:
        label0.append(j)
    zs = []
    for jj in data0:        
        # 随机生成500个椒盐
        rows, cols, dims = jj.shape
        for i in range(500):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            jj[x, y, :] = 1
        zs.append(jj)
    data=list(data)
    label=list(label)
    data.extend(zs)
    label.extend(label0)
    return np.asarray(data,np.float32),np.asarray(label,np.int64)