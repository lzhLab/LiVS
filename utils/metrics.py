import torch 
#import torchmetrics
#from torchmetrics.functional import dice_score, precision_recall
import torch.nn as nn
#import GeodisTK
#from scipy import ndimage
from sklearn import metrics
import numpy as np
from medpy import metric

def hausdorff95(res, ref):
    return metric.binary.hd95(res, ref)

def Assd(res, ref):
    return metric.binary.assd(res, ref)

def asd(res, ref):
    return metric.binary.asd(res, ref)

def hausdorff(res, ref):
    return metric.binary.hd(res, ref)

def voe(res, ref):
    return 1.-metric.binary.jc(res, ref)

def rvd(res, ref):
    return metric.binary.ravd(res, ref)

def dice(res, ref):
    return metric.binary.dc(res, ref)

def precision(res, ref):
    return metric.binary.precision(res, ref)

def msd(res, ref):
    return np.mean((res-ref)**2)

def recall(res, ref):
    return metric.binary.recall(res, ref)

def sen(res, ref):
    return metric.binary.sensitivity(res, ref)

def spe(res, ref):
    return metric.binary.specificity(res, ref)



#a = np.zeros((1,1,128,128))
#a[0,:,100:120,50:80]=1
#a[1,:,100:120,50:80]=1
#b = np.zeros((1,1,128,128))
#b[0,:,100:110,50:80]=1
#b[1,:,100:120,50:130]=1

#print('hausdorff95=',hausdorff95(a,b))
#print('Assd=',Assd(a,b))
#print('asd=',asd(a,b))
#print('hausdorff=',hausdorff(a,b))
#print('voe=',voe(a,b))
#print('rvd=',rvd(a,b))
#print('dice=',dice(a,b))
#print('precision=',precision(a,b))
#print('msd=',msd(a,b))
#print('recall=',recall(a,b))
#print('sen=',sen(a,b))
#print('spe=',spe(a,b))
