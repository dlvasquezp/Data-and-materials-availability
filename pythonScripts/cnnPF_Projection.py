# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
from skimage import io
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #Load OCT data   
    OCT = io.imread('../data/OCT/cnnOCT.tif')
    OCT = np.rot90(OCT,3)
    OCT = OCT[::2,::2,::2]
    
    dz, dx = 3.4, 8 #miu
    
    #Load segmented data
    labeledPF = io.imread('../data/OCT/labeledCnnPF.tif')
    labeledPF = np.rot90(labeledPF,3)
    
    #Define color maps
    cvals  = [0,1,2,3]
    colors = ['white','blue','orange','green']
    norm   = plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap   = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
    mapper = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    
    for slide in [50,70,100]: 
        oct0 = OCT[:,slide,:]
        img0 = labeledPF[:,slide,:]
        alpha_layer  = np.multiply((img0!=0),1)
        
        im = mapper.to_rgba(img0)
        im[...,-1] = alpha_layer
        
        plt.figure()
        plt.title('slide: '+ str(slide))
        plt.imshow(oct0[0:80:], cmap='gray')
        plt.imshow(  im[0:80,:])
        plt.show()
        
        areaList=[]
        for idx in [1,2,3]:
            areaList.append(np.sum(img0==idx)*dx*dz) #miu
            
        bar_colors = ['blue','orange','green']
        
        bar_labels = []
        for idx,areaTemp in zip([1,2,3],areaList):
            bar_labels.append('PF#{:d} : {:.3f}'.format(idx,areaTemp))
            
        plt.figure()
        plt.title('slide: '+ str(slide))
        plt.bar([1,2,3],areaList,label=bar_labels, color=bar_colors)
        plt.legend(title='Projected area [µm^2]')
        plt.show()
