# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
from skimage import io
import matplotlib.pyplot as plt

if __name__ == '__main__':

    #Load OCT data   
    OCT = io.imread('../data/OCT/invivoOCT.tif')
    OCT = np.transpose(OCT, (1,0,2))
    
    dz, dx = 1.7, 4 #miu
    
    #Load segmented data
    labeledPF = io.imread('../data/OCT/labeledInvivoPF.tif')
    labeledPF = np.transpose(labeledPF, (1,0,2))
    
    #Define color maps
    cvals  = [0,1,2]
    colors = ['white','blue','green']
    norm   = plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap   = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
    mapper = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
    
    for slide in [384,111]: 
        oct0 = OCT[:,slide,:]
        img0 = labeledPF[:,slide,:]
        alpha_layer  = np.multiply((img0!=0),1)
        
        im = mapper.to_rgba(img0)
        im[...,-1] = alpha_layer
        
        plt.figure()
        plt.title('slide: '+ str(slide))
        plt.imshow(oct0[0:160:], cmap='gray')
        plt.imshow(  im[0:160,:])
        plt.show()
        
        areaList=[]
        for idx in [1,2]:
            areaList.append(np.sum(img0==idx)*dx*dz) #miu
            
        bar_colors = ['blue','green']
        
        bar_labels = []
        for idx,areaTemp in zip([1,2],areaList):
            bar_labels.append('PF#{:d} : {:.3f}'.format(idx,areaTemp))
            
        plt.figure()
        plt.title('slide: '+ str(slide))
        plt.bar([1,2],areaList,label=bar_labels, color=bar_colors)
        plt.legend(title='Projected area [µm^2]')
        plt.show()
