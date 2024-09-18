#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from skimage import io

#%%
Learning_Rate=1e-5
width=500  # image width
height=300 # image height
batchSize=6 # 6 for 8GB, 12 for 15GB, 50 for 40GB
noIteration= 10000

#%%
TrainFolderPath = "../data/train"
[imgName, mskName] = os.listdir(TrainFolderPath)

imgTiff = io.imread(os.path.join(TrainFolderPath, 'images.tif'))
mskTiff = io.imread(os.path.join(TrainFolderPath, 'masks.tif' ))

segClasses={'background':0,'bundle':1,'fibre':2,'out of field':3}
segWeight =[           0.8,         1,        1,             0.2]

#%%----------------Transform image---------------------
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize(0.473, 0.066)])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])

#%%-------------------Read image ----------------------
def ReadRandomImage(imgTiff,mskTiff): 
    idxImg= np.random.randint(0,len(imgTiff))
    Img   = np.uint8(imgTiff[idxImg])
    AnnMap= np.int32(mskTiff[idxImg])
    
    return transformImg(Img),transformAnn(AnnMap)

def LoadBatch(): # Load batch of images
    images = torch.zeros([batchSize,1,height,width])
    ann = torch.zeros([batchSize, height, width])
    for i in range(batchSize):
        images[i],ann[i]=ReadRandomImage(imgTiff,mskTiff)
    return images, ann

#%%--------------Load and set net and optimizer---------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) 
Net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
Net.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1)) 
Net=Net.to(device)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate)

#%%----------------Train------------------------------
modelName = 'HeartModel1'
lossRecord=[]
showImages= False

for itr in range(noIteration+1): 

    images,ann=LoadBatch() 
    
    if showImages:
        plt.figure()
        plt.imshow(images[0][0,:,:])
        plt.show()
    
        plt.figure()
        plt.imshow(ann[0])
        plt.show()
    
    torchImages=torch.autograd.Variable(images, requires_grad=False).to(device) 
    torchAnn   =torch.autograd.Variable(   ann, requires_grad=False).to(device) 
    
    Pred=Net(torchImages)['out'] 
    Net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(segWeight,device=device))
    Loss=criterion(Pred,torchAnn.long()) 
    Loss.backward() 
    optimizer.step() 
    seg = torch.argmax(Pred[0], 0).cpu().detach().numpy()  
    
    if showImages:
        plt.figure()
        plt.imshow(seg)
        plt.show()
    
    lossRecord.append(Loss.data.cpu().numpy())
    print(itr,") Loss=",Loss.data.cpu().numpy(),'MeanVal:',np.mean(lossRecord[-10:]))
    if itr > 10 and itr == noIteration:
        print("Saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(),  "Last iteration_"+ str(itr) +"_"+ modelName +".torch")
    if itr > 10 and np.mean(lossRecord[-10:]) < 0.01: 
        print("Threshold achieved, saving Model" +str(itr) + ".torch")
        torch.save(Net.state_dict(),   str(itr) +'_'+ modelName +".torch")
        break 

plt.figure()
plt.plot(lossRecord)
plt.show()