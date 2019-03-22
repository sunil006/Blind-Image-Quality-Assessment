from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import image
import numpy as np
import cv2
import cPickle
#import regression
from matplotlib import pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
from scipy.integrate import quad
import math
import pickle
#########################################################################################################################################
import scipy.io as si
import os
import func as f
import ggd
from tqdm import tqdm
from keras import optimizers
import hickle as hkl 
from skimage.util import view_as_windows
import pandas as pd
import glob


filenames = glob.glob("./CSIQ/dst_imgs/awgn/*.png")
filenames.sort()
Images = [cv2.imread(img) for img in filenames]

folders=["jpeg","jpeg2000","fnoise","blur","contrast"]
i=0

for folder in folders:
	filenames = glob.glob("./CSIQ/dst_imgs/"+folder+"/*.png")
	filenames.sort()
	imagesi = [cv2.imread(img) for img in filenames]
	Images=np.append(Images,imagesi,axis=0)
	print(len(Images))

Labels=pd.read_excel("./CSIQ/labels.xlsx")['Labels'].values
Labels=Labels*100
print (Images.shape)
print (Labels.shape)

print ("Stage:1 Read all the Images")

#########################################################################################################################################

print (len(Labels),Images.shape)

#########################################################################################################################################


window = signal.gaussian(49, std=7)
window=window/sum(window)
NSStrainfeatures=[]
NSStestfeatures=[]
count=0

for I in tqdm(Images):
	#print count
	count=count+1 
        #I = cv2.imread('img5.bmp',cv2.IMREAD_GRAYSCALE)
        I = I.astype(np.float32)
        u = cv2.filter2D(I,-1,window)
        u = u.astype(np.float32)
	
        diff=pow(I-u,2)
        diff = diff.astype(np.float32)
        sigmasquare=cv2.filter2D(diff,-1,window)
        sigma=pow(sigmasquare,0.5)

        Icap=(I-u)/(sigma+1)
        Icap = Icap.astype(np.float32)
        gamparam,sigma = ggd.estimateggd(Icap)
        feat=[gamparam,sigma]



        shifts = [ (0,1), (1,0) , (1,1) , (-1,1)];
        for shift in shifts:
                shifted_Icap= np.roll(Icap,shift,axis=(0,1))
                pair=Icap*shifted_Icap
                alpha,leftstd,rightstd=ggd.estimateaggd(pair)
                const=(np.sqrt(math.gamma(1/alpha))/np.sqrt(math.gamma(3/alpha)))
                meanparam=(rightstd-leftstd)*(math.gamma(2/alpha)/math.gamma(1/alpha))*const;
                feat=feat+[alpha,leftstd,rightstd,meanparam]

        feat=np.array(feat)
        NSStrainfeatures.append(feat)



#print NSStestfeatures.shape
NSStrainfeatures=np.array(NSStrainfeatures)
hkl.dump( NSStrainfeatures, 'NSStrainfeatures_CSIQ.hkl' )

hkl.dump( Labels, 'CSIQ_Labels.hkl' )
hkl.dump( Images, 'CSIQ_Images.hkl' )

