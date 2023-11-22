# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 17:45:50 2021

@author: 
"""

from scipy.ndimage import morpho
import matplotlib.pyplot as plt
import numpy as np
import os

path = os.getcwd()
img = plt.imread(path + r'\angio5.bmp')

s = img.shape
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
img_out = np.zeros((s[0], s[1]), dtype='uint8')
img_out = (0.299 * R + 0.587 * G + 0.114 * B)

plt.figure()
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_out, cmap='gray'), plt.title('Converted Image')

threshold = 100
img_bi = np.zeros(img_out.shape, dtype='uint8')
img_bi[img_out > threshold] = 255

plt.figure()
plt.imshow(img_bi, cmap='gray'), plt.title('Binary Image')

h = np.histogram(img_out, 256, range=(0,255))
h1 = h[0]/np.sum(h[0])
plt.figure()
plt.bar(np.arange(0, 256), h1), plt.xlabel('gray level'), plt.ylabel('probability function')
















