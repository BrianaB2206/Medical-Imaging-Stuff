# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

se1 = np.ones((1,5))

#afisarea continutului variabilei se1
print('se1 =', se1)

# define another structurant element: V4 si V8
V4 = np.array([[1,0,0],[0,1,0],[0,0,1]])
V8 = np.ones((3,3))

print('V4 =', V4)
print('V8 =', V8)

path1 = r'D:\Laboratoare\Laborator_5'
img = plt.imread(path1 + '/litera_j.png')

# transforma imaginea originala in imagine cu un singur plan
s = img.shape
gri = np.zeros((s[0],s[1]), dtype = 'uint8')
gri = 0.299 *img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]

# transforma in imagine binara
# binarizarea se poate face cu functia creata in laboratorul de modificare a contrastului
# sau

prag = 0.1
img_bi = np.zeros(gri.shape, dtype = 'uint8')
img_bi[gri >= prag]= 255

#afisarea imaginii binare
plt.imshow(img_bi, cmap = 'gray')
plt.title('imaginea binara')
plt.figure()
plt.subplot(1,3,1), plt.imshow(img), plt.title('orig')
plt.subplot(1,3,2), plt.imshow(gri, cmap = 'gray'),plt.title('gray_level') 
plt.subplot(1,3,3), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 

import scipy.ndimage.morphology as morpho
se1 = np.ones((1,5))
# erodarea imaginii folosind elementul structurant se1
er_se1 = morpho.binary_erosion(img_bi, se1)
plt.imshow(er_se1, cmap = 'gray') 

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er_se1, cmap = 'gray'),plt.title('erodata') 

# schimba elementul structurant

import scipy.ndimage.morphology as morpho
se2 = np.ones((10,1))
# erodarea imaginii folosind elementul structurant se1
er_se2 = morpho.binary_erosion(img_bi, se2)
plt.imshow(er_se2, cmap = 'gray') 

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er_se2, cmap = 'gray'),plt.title('erodata') 


# schimba elementul structurant cu V8

import scipy.ndimage.morphology as morpho
# erodarea imaginii folosind elementul structurant se1
er1 = morpho.binary_erosion(img_bi, V8)
plt.imshow(er1, cmap = 'gray') 

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er1, cmap = 'gray'),plt.title('erodata') 

# schimba elementul structurant cu ones(11,11)

import scipy.ndimage.morphology as morpho
# erodarea imaginii folosind elementul structurant se1
se3 = np.ones((11,11))
er2 = morpho.binary_erosion(img_bi, se3)

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er2, cmap = 'gray'),plt.title('erosion')  

import scipy.ndimage.morphology as morpho
# erodarea imaginii folosind elementul structurant se1
se4 = np.ones((20,20))
er2 = morpho.binary_erosion(img_bi, se4)

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er2, cmap = 'gray'),plt.title('erosion') 

# Part II.3: Dilation

dil = morpho.binary_dilation(img_bi,V8)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(dil, cmap = 'gray'),plt.title('dilated') 



# to fill in the circle we can apply one more time the dilation operator

dil1 = morpho.binary_dilation(dil,V8)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(dil1, cmap = 'gray'),plt.title('dilatata2') 

# Part II.3: Opening
se4 = np.ones((20,20))
opened = morpho.binary_opening(img_bi,se4)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(opened, cmap = 'gray'),plt.title('opened') 


# Part II.4: Closing
se4 = np.ones((20,20))
closed = morpho.binary_closing(img_bi,se4)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_bi, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(closed, cmap = 'gray'),plt.title('closed')