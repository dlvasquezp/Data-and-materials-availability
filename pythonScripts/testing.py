#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from skimage import io
from sklearn.metrics import classification_report

#%%
nameModel = "heartModel_V3.2.torch"
width=500
height=300

segClasses={'background':0,'bundle':1,'fibre':2,'out of field':3}
segWeight =[           0.8,         1,        1,             0.2]

#%%
transformImg=tf.Compose([tf.ToPILImage(),tf.Resize((height,width)),tf.ToTensor(),tf.Normalize(0.473, 0.066)])
transformAnn=tf.Compose([tf.ToPILImage(),tf.Resize((height,width),tf.InterpolationMode.NEAREST),tf.ToTensor()])
       
#%%
modelPath = "../torchModel/"+nameModel  # Path to trained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True) 
Net.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
Net.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1)) 
Net = Net.to(device)
Net.load_state_dict(torch.load(modelPath))
Net.eval() 

#%%
validationFolderPath = "../data/test"
TestFolder = os.listdir(validationFolderPath)

imgTiff = io.imread(os.path.join(validationFolderPath, 'images.tif'))
mskTiff = io.imread(os.path.join(validationFolderPath, 'masks.tif' ))

segList =[]
mskList =[]
for image,mask in zip(imgTiff,mskTiff):

    Img=np.uint8(image)
    height_orgin , widh_orgin = Img.shape
    Img = transformImg(Img)
    Img = torch.autograd.Variable(Img, requires_grad=False).to(device).unsqueeze(0)
    
    with torch.no_grad():
        Prd = Net(Img)['out']
    
    Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0])
    seg = torch.argmax(Prd, 0).cpu().detach().numpy()  
    seg = np.uint8(seg)

    AnnMap=np.uint8(mask)
    
    segList.append(seg)
    mskList.append(AnnMap)

    plt.figure()
    plt.imshow(AnnMap)
    plt.show()
    
    plt.figure()
    plt.imshow(seg)
    plt.show()
#%%   

segList = np.array(segList)
mskList = np.array(mskList)    
print(classification_report(mskList.flatten(),segList.flatten(),target_names=segClasses.keys()))