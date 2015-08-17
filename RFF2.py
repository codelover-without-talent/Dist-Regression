from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from kerpy.Kernel import Kernel
from kerpy.BagKernel import BagKernel
from kerpy.GaussianKernel import GaussianKernel
from kerpy.GaussianBagKernel import GaussianBagKernel


rng = np.random.RandomState(0)
#########################################################
# generate sample as Zoltan's paper
#def sam_gen(l,n):
#    a1 = np.random.uniform(0,1,(2,2))
#    beta1 = np.random.uniform(0,np.pi,l)
#    
#    sample = np.zeros((l,n,2))
#    entro = np.zeros(l)
#    m0 = np.zeros(2)
#    
#    for ii in np.arange(l):
#        ri = np.array([[np.cos(beta1[ii]),-np.sin(beta1[ii])],[np.sin(beta1[ii]),np.cos(beta1[ii])]])
#        ra = np.dot(ri,a1)
#        ci = np.dot(ra,ra.T)
#        
#        entro[ii] = 0.5 * np.log(2* np.pi * np.e * ci[0,0])
#        sample[ii] = np.random.multivariate_normal(m0,ci,n)
#    return sample, entro

#sam_tr,entro_tr = sam_gen(80,500)
#sam_tt, entro_tt = sam_gen(20,500)

##########################################################
# generate 1D sample
def sam1d_gen(l,n):
    sd = np.random.uniform(0,10,l)
    
    sample = list()
    entro = list()
    m0 = 0.0
    
    for ii in np.arange(l):
        pois = np.random.normal(m0,sd[ii],n)
        en = 0.5 * np.log(2* np.pi * np.e * (sd[ii] ** 2))
        sample.append(pois)
        entro.append(en)
    
    return sample, entro

sam1d_tr, entro1d_tr = sam1d_gen(5,10)
print sam1d_tr
#print entro1d_tr
sam1d_tt, entro1d_tt = sam1d_gen(2,5)
 
##########################################################
# conduct the ridge regression
data_gamma = 1.0
bag_gamma = 1.0
data_kernel = GaussianKernel(data_gamma)
print data_kernel.kernel(sam1d_tr)

bag_kernel = GaussianBagKernel(data_kernel,bag_gamma)
#standard distribution regression - computes full kernel matrices
#coeff,ypred=bag_kernel.ridge_regress(sam1d_tr,entro1d_tr,lmbda=0.01,Xtst=sam1d_tt)
#or distribution regression with random features
#bag_kernel.rff_generate(50,60,dim=dim) #50 random features for bag_kernel, 60 for data_kernel
#coeff,ypred=bag_kernel.ridge_regress_rff(baglistX,y,Xtst=baglistXtst)

        
    
    