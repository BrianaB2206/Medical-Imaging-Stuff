# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 20:22:54 2023

@author: brian
"""

import numpy as np
import matplotlib.pyplot as plt









###############################################################################
############################   Lab 3   ########################################






#1
x = (1, "Medical",3.4,9,11)
print(x)
y = list(x)
y[1] = [1,2,3,4]
x1 = tuple(y)

print(x1)

#2

path  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\scintigrafia.jpg'

img = plt.imread(path)
print(type(img))

s = img.shape
print(s)

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

plt.figure()
plt.subplot(1,4,1); plt.imshow(img); plt.title('color image')
plt.subplot(1,4,2); plt.imshow(R,cmap = 'gray'); plt.title('planul de rosu')
plt.subplot(1,4,3); plt.imshow(G,cmap = 'gray'); plt.title('planul de verde')
plt.subplot(1,4,4); plt.imshow(B,cmap = 'gray'); plt.title('planul de albastru')



# scaling to the right range [0,255]
G = G.astype('float')
R = R.astype('float')
B = B.astype('float')
G1 = G + 35
R1 = R 
B1 = B

print(G1.min())
print(G1.max())

print(R1.min())
print(R1.max())

print(B1.min())
print(B1.max())

G1 = np.clip(G1, G1.min(),255).astype('uint8')
R1 = np.clip(R1, R1.min(),255).astype('uint8')
B1 = np.clip(B1, B1.min(),255).astype('uint8')

RGB = np.dstack((R1,G1,B1))

plt.figure()
plt.subplot(1,2,1), plt.imshow(img), plt.title('original img')
plt.subplot(1,2,2), plt.imshow(RGB), plt.title('modified_img')

#3
path2  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\CThead.tif'
import cv2 as cv
  
# read the image file
img = cv.imread(path2, 2)
  
ret, bw_img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
  
# converting to its binary form
bw = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
  
cv.imshow("Binary", bw_img)
cv.waitKey(0)
cv.destroyAllWindows()


#4

path3  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\gray1.png'
path4  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\gray2.png'

img3 = plt.imread(path3)
img4 = plt.imread(path4)

plt.figure()
plt.subplot(1,2,1), plt.imshow(img3, cmap = 'gray')
plt.subplot(1,2,2), plt.imshow(img4, cmap = 'gray')

h = np.histogram(img3, bins = 256, range = (0,255))

print(type(h))
print(h)
h1 = h[0]/np.sum(h[0])

sum(h1)
plt.figure()
plt.plot(np.arange(256),h1)
plt.xlabel('gray levels')
plt.ylabel('probability density function for h1')

h_new = np.histogram(img4, bins = 256, range = (0,255))

print(type(h))
print(h)
h2 = h_new[0]/np.sum(h_new[0])

sum(h2)
plt.figure()
plt.plot(np.arange(256),h2)
plt.xlabel('gray levels')
plt.ylabel('probability density function for h2')





###############################################################################






# Application I: Recap: Read an image (CTImage.tif) and display it 


path  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\CThead.tif'

img = plt.imread(path)

plt.figure()
plt.imshow(img, cmap = 'gray')

# Application II:
#Determinati histograma cu 256 de intervale (bins) a imaginii 'CThead.tif' si stocati-o in variabila h .
#Afisati continutul acestei variabile folosind functia print
#Afisati graficul histogramei

# determinarea histograma variabile img, pentru 256 de intervale
## daca imaginea poate avea maxim 256 de nivel de gri, =>
## => histograma cu 256 de intervale = histograma nivelului de gri 

h = np.histogram(img, bins = 256, range = (0,255))

print(type(h))
print(h)

# se observa ca h are 2 elemente:
## primul element contine numarul de aparitii pentru fiecare nivel de gri (interval)
## al doilea element contine limitele intervalurilor

#Display the first two elements of h
print('h[0] =', h[0])
print('h[1] =', h[1])

# histograma = distributie de probabilitate
# => este necesara conversia din numar de aparitii in probabilitati

h1 = h[0]/np.sum(h[0])

sum(h1)

#afisarea numarului de aparitii a nivelelor de gri 3,4,5,6,7,8,9,10
print('numar de aparitii a niv. de gri 3:10\n', h[0][3:11])

#afisarea probabilitatii de aparitie a nivelelor de gri 3,4,5,6,7,8,9,10
print('probabilitatea de aparitie a niv. de gri 3:10\n', h1[3:11])

# Histogram plot
plt.figure()
plt.plot(np.arange(256),h1)
plt.xlabel('gray levels')
plt.ylabel('probability density function')

# or using barplot
plt.figure()
plt.bar(np.arange(256),h1)
plt.xlabel('gray levels')
plt.ylabel('probability density function')

# se observa ca h are 2 elemente:
## primul element contine numarul de aparitii 
#pentru fiecare nivel de gri (interval)
## al doilea element contine limitele intervalurilor

# What is a Tuple?
#empty tuple
my_tuple = ()
print(my_tuple)
print(type(my_tuple))

# Tuple having integers
my_tuple = (1, 2, 3)
print(my_tuple)

#Tuple with mixed data types
my_tuple = (1, 'Hello!', 3.4)
print(my_tuple)

print(my_tuple[1])

#Tuple with nested list
# What is Python Nested List? 
# A list can contain any sort object, 
# even another list (sublist), 
# which in turn can contain sublists themselves, 
#and so on. This is known as nested list. 
#You can use them to arrange data into hierarchical 
#structures.

my_tuple = ("mouse", [8, 4, 6], (1, 2, 3))
print(my_tuple)

#Application III
#Read a color image ('Scintigrafia.jpg').
#Display the type and the dimensions of the variable

path  = r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\scintigrafia.jpg'

img = plt.imread(path)
print(type(img))

s = img.shape
print(s)

# Avand in vedere ca variabila in care s-a citit/stocat imaginea are trei dimensiuni (linii, coloane, plane) => imaginea este color.

# Application III: Split the R,G,B channels of the above considered image.

#Afisati imaginea
#Extrageti planurile de culoare din imagine
#Afisati in aceeasi figura: imaginea originala si fiecare plan in parte, una langa alta. In dreptul fiecarei imagini sa fie trecuta semnificatia acesteia (titlu)
#Indicatii: In urma extragerii planurilor, fiecare plan va fi cu nivele de gri.

R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]

plt.figure()
plt.subplot(1,4,1); plt.imshow(img); plt.title('color image')
plt.subplot(1,4,2); plt.imshow(R,cmap = 'gray'); plt.title('planul de rosu')
plt.subplot(1,4,3); plt.imshow(G,cmap = 'gray'); plt.title('planul de verde')
plt.subplot(1,4,4); plt.imshow(B,cmap = 'gray'); plt.title('planul de albastru')


#Application IV:
#having the channels splitted, add 50 value to the Green channel an combine back R,G,B channels
#to obtain the initial RGB color image

s = R.shape
img4 = np.zeros((s[0],s[1],3), dtype = 'uint8')

img4[:,:,0] = R
img4[:,:,1] = G
img4[:,:,2] = B

plt.figure()
plt.subplot(1,2,1), plt.imshow(img), plt.title('original')
plt.subplot(1,2,2), plt.imshow(img4), plt.title('recovered')

plt.imsave(r'C:\Users\brian\OneDrive\Desktop\MI\Lab3_MI\From Moodle\scinti_mod.jpg',img4)
 

# adding 50 constant value to Green channel

print(G.min())
print(G.max())

G = G + 50
print(G.min())
print(G.max())

#let us have an example 
g = np.arange(0,251,50)
print(g)
g= g + 50
print(g)
print(g.astype('uint8'))


# scaling to the right range [0,255]
G = G.astype('float')
G1 = G + 50

print(G1.min())
print(G1.max())

G1 = np.clip(G1, G1.min(),255).astype('uint8')

plt.figure()
plt.subplot(1,2,1), plt.imshow(G, cmap = 'gray'), plt.title('original Green')
plt.subplot(1,2,2), plt.imshow(G1,cmap = 'gray'), plt.title('modified_Green')


# Application 5: 
# Convert a color image Scintigrafia.jpg in a grayscale image

img_out = img - img
plt.imshow(img_out, cmap = 'gray')

s = img.shape
print(s)

img_out = np.zeros((s[0],s[1]), dtype = 'uint8')

img_out = 0.299 *img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
img_out = img_out.astype('uint8')

plt.figure()
plt.subplot(1,2,1), plt.imshow(img), plt.title('orig')
plt.subplot(1,2,2), plt.imshow(img_out, cmap = 'gray'),plt.title('gray_level') 

# App6: Write a function to check if the image is color and convert it to gray

#def rgb2gray(img_in, tip):
 #   s =img.shape
  #  if len(s) == 3 and s[2] ==3:
   #     if tip == 'png':
    #        img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
     #   elif tip == 'jpg':
      #      img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2])
       # img_out = img_out.astype('uint8')
        #return img_out
    #else:
     #   print('the image is not a color image')
        
#img_out = np.zeros((s[0],s[1]), dtype = 'uint8')
#img_out = rgb2gray(img, 'jpg')        
#plt.imshow(img_out, cmap = 'gray')









###############################################################################
############################   Lab 4   ########################################








path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\download.jpg'
img = plt.imread(path)
plt.figure()
plt.imshow(img, cmap='gray')


# App1: Write a function to check if the image is color and convert it to gray

def rgb2gray(img_in, tip):
    s =img.shape
    img_in = img_in.astype('float')
    if len(s) == 3 and s[2] ==3:
        if tip == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
        elif tip == 'jpg':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2])
        img_out = np.clip(img_out,0,255)    
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('the image is not a color image')

s = img.shape        
img_out = np.zeros((s[0],s[1]), dtype = 'uint8')
img_out = rgb2gray(img, 'jpg')        
plt.figure()
plt.imshow(img_out, cmap = 'gray')

img = img_out

# App 2: 
# Write a function which receives a grayscale image as input and display its negative

def return_negative(img_in):
    img_out = 255 - img_in
    return img_out


print(type(img))
print(img.min())
print(img.max())
plt.imshow(img.astype('uint8'), cmap = 'gray')


img_out = return_negative(img)
print(img.min())
print(img.max())
#img_out = img_out * 255
plt.figure()
plt.imshow(img_out.astype('uint8'), cmap = 'gray')


#App3:Linear Contrast Transformation

a=100
b=170
Ta=50
Tb=250

plt.figure()
plt.imshow(img, cmap = 'gray')
img_out = np.zeros((s[0],s[1]), dtype = 'uint8')     

s = img.shape


def linear_transf(img_in,Ta,a,Tb,b):
    img_in = img_in.astype('float')
    s = img.shape
    img_out = np.zeros((s[0],s[1]), dtype = 'uint8')
    for i in range(s[0]):
        for j in range(s[1]):
            if (img_in[i,j] < a):
                img_out[i,j] = (Ta/a) * img_in[i,j]
            else:
                if ((img_in[i,j] >= a) and (img_in[i,j] <=b)):
                    img_out[i,j] = Ta + (Tb-Ta)/(b-a) * (img_in[i,j] - a)
                else:
                        img_out[i,j] = Tb + ((255 - Tb)/(255 - b)) * (img_in[i,j] - b)
    img_out = np.clip(img_out,0,255).astype('uint8')                    
    return img_out         
                
s = img.shape
img_out = np.zeros((s[0],s[1]), dtype = 'uint8')                    
img_out = linear_transf(img,Ta,a,Tb,b)   
plt.figure()  
plt.imshow(img_out.astype('uint8'), cmap = 'gray')     
          

# App4: Contrast stretching

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
print(s)

out = np.zeros((s[0],s[1]), dtype = 'uint8')
Tb = 255
Ta = 0
a = 100
b = 190

def contrast_stretching(img,a, b,Ta,Tb):
    for i in range(s[0]):
        for j in range(s[1]):
            if (img[i,j] < a):
                out[i,j] = Ta
            elif img[i,j] >= a & img[i,j] < b:
                out[i,j] = (Tb - Ta)/(b-a) * (img[i,j] - a) 
            else:
                out[i,j] = Tb
    return out

out = contrast_stretching(img, a,b,Ta,Tb)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')     

# App5: Binarizare

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
# img = img * 256
s = img.shape
print(s)
out = np.zeros((s[0],s[1]), dtype = 'uint8')
prag = 160

def binara(img, prag):
    for i in range(s[0]):
        for j in range(s[1]):
            if (img[i,j] < prag):
                out[i,j] = 0
            else:
                out[i,j] = 255
    return out

out = binara(img, prag)

plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')       

# App6: Clipping

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
print(s)
Tb = 255
Ta = 80
a = 80
b = 200

out = np.zeros((s[0],s[1]), dtype = 'uint8')

def clipping(img,a, b,Ta,Tb):
    for i in range(s[0]):
        for j in range(s[1]):
            if (img[i,j] < a):
                out[i,j] = 0
            elif img[i,j] >= a & img[i,j] < b:
                out[i,j] = (Tb-Ta)/(b-a) * (img[i,j] - a)
            else:
                out[i,j] = 0
    return out

out = clipping(img, a,b,Ta,Tb)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')
       
                
#App7: Slicing                

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
print(s)

out = np.zeros((s[0],s[1]), dtype = 'uint8')

def slicing(img,a, b,Ta,Tb):
    for i in range(s[0]):
        for j in range(s[1]):
            if (img[i,j] < a):
                out[i,j] = 0
            elif img[i,j] >= a & img[i,j] < b:
                out[i,j] = Tb 
            else:
                out[i,j] = 0
    return out

out = slicing(img, a,b,Ta,Tb)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')   


# App8: Non-linear: putere

# out2 = 255*(double(a)/255).^(r);

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
img = img.astype('float')

def putere(img,r):
    out = 255 * (img/255) ** r
    return out

out = putere(img,0.2)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')   

# App 9: Non-linear: Putere punct fix

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
img = img.astype('float')

def putere_punct_fix(img, prag, r):
    for i in range (s[0]):
        for j in range(s[1]):
            if img[i,j] < prag:
                out[i,j] = prag * (img[i,j]/prag) ** r
            else: 
                out[i,j] = 255 - (255 - prag) * (255 - img[i,j]) * (255 - prag) ** r
    return out            
                
out = putere_punct_fix(img,90, 0.2)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')   


# App10: Non-linear: Logarithm

import numpy as np

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
img = img.astype('float')
img_out = np.log(img)
#img_out = np.clip(img_out, 0,255).astype('uint8')

plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(img_out, cmap = 'gray')   


# App11: Non-linear: exponential

# out1=256.^(double(a)/255)-1;

path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab4_MI\prosthesis.bmp'
img = plt.imread(path)
s = img.shape
img = img.astype('float')

def exponentiala(img):
    out = 256 ** (img/255) - 1
    return out

out = exponentiala(img)
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray')
plt.subplot(1,2,2); plt.imshow(out, cmap = 'gray')   










###############################################################################
####################  Lab 5 (alea cu morpho)   ################################









"""
Function init + test examples
"""
#images used to draw conclusions Y1.jpg, Y167.jpg, Y147.jpg, Y22.jpg, Y47.jpg,
#Y251.jpg, Y159.jpg,Y73.jpg, Y46.jpg
path  = r'C:\Users\brian\OneDrive\Desktop\MI\Proiect MI\yes\Y154.jpg'
img = plt.imread(path)

plt.figure()
plt.imshow(img, cmap = 'gray'),plt.title('Original_Test')

s = img.shape
print(s)

"""
Grayscaling

"""

def rgb2gray(img_in, tip):
    s =img.shape
    img_in = img_in.astype('float')
    if len(s) == 3 and s[2] ==3:
        if tip == 'png':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2]) * 255
        elif tip == 'jpg':
            img_out = (0.299 * img_in[:,:,0] + 0.587 * img_in[:,:,1] + 0.114 * img_in[:,:,2])
        img_out = np.clip(img_out,0,255)    
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('the image is not a color image')
        
    
"""
Test

"""    

s = img.shape        
img_out = np.zeros((s[0],s[1]), dtype = 'uint8')
img_out = rgb2gray(img, 'jpg')        
plt.figure()
plt.imshow(img_out, cmap = 'gray'),plt.title('Grayscale_Test')

img = img_out


"""
Linear transformation

"""

def linear_transform(img, a, b, Ta, Tb):
     s = img.shape
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
     
     for i in range(s[0]):
         for j in range(s[1]):
             if img[i,j] < a:
                 img_out[i,j] = (Ta/a) * img[i,j]
             elif img[i,j] >= a and img[i,j] <=b:
                 img_out[i,j] = Ta + (Tb-Ta)*(img[i,j]-a)/(b-a)
             else:
                 img_out[i,j] = Tb + (255-Tb)*(img[i,j]-b)/(255-b) 
     return img_out
 
     
"""

Test

"""     

img_linear = linear_transform(img, 100, 200, 20, 200)
plt.figure()
plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(img_linear, cmap='gray'), plt.title('Linearization_Test')

"""
Binarization

""" 

def binary_transform(img, T):
     s = img.shape
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
     
     for i in range(s[0]):
        for j in range(s[1]):
             if img[i,j] < T:
                 img_out[i,j] = 0
             else:
                 img_out[i,j] = 255
     return img_out
 
"""

Test

"""     
 
img_binar = binary_transform(img, 160)
plt.figure()
plt.subplot(121); plt.imshow(img, cmap = 'gray'); plt.title('Original image') 
plt.subplot(122); plt.imshow(img_binar, cmap = 'gray'); plt.title('Binarization_Test')


"""

Exponential Function

""" 

def exponential(img):
     s = img.shape
     img = img.astype('float')
     img_out = np.zeros((s[0], s[1]), dtype='uint8')
#     
     img_out = 256 ** (img/255) - 1
     return img_out   
 
"""

Test

"""     
 
img_exp = exponential(img) 
plt.figure()
plt.subplot(1,2,1); plt.imshow(img, cmap = 'gray'); plt.title('Original image') 
plt.subplot(1,2,2); plt.imshow(img_exp, cmap = 'gray'); plt.title('Exponential_Test')


"""

Parameters for mathematical morphology functions

""" 

se1 = np.ones((8,5))

#afisarea continutului variabilei se1
print('se1 =', se1)

# define another structurant element: V4 si V8
V4 = np.array([[1,0,0],[0,1,0],[0,0,1]])
V8 = np.ones((3,3))

print('V4 =', V4)
print('V8 =', V8)

from scipy.ndimage import morphology

"""

Erosion + Test

"""  

# erodarea imaginii folosind elementul structurant se1
er1 = morphology.binary_erosion(img_binar, se1)
plt.imshow(er1, cmap = 'gray') 

plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(er1, cmap = 'gray'),plt.title('erodata_test') 

"""

Opening + Test

"""  

se4 = np.ones((20,20))
opened = morphology.binary_opening(img_binar,se4)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(opened, cmap = 'gray'),plt.title('opened_test')


dil = morphology.binary_dilation(img_binar,V8)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(dil, cmap = 'gray'),plt.title('dilated') 


closed = morphology.binary_closing(img_binar, V8)
plt.figure()
plt.subplot(1,2,1), plt.imshow(img_binar, cmap = 'gray'),plt.title('binara') 
plt.subplot(1,2,2), plt.imshow(closed, cmap = 'gray'),plt.title('closed') 










###############################################################################
############################   Lab 6   ########################################









import matplotlib.pyplot as plt
import scipy.ndimage as sc
import numpy as np
import random
import cv2


path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\angiografie.jpg'
img = plt.imread(path)

plt.figure()
plt.imshow(img, cmap='gray'); plt.title('angiografie.jpg')

print('Image dimension: ', img.shape)

def rgb2gray(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    s = img.shape
    img_out = np.zeros((s[0], s[1]), dtype = 'uint8')
    if len(s)==3 and s[2]==3:
        img_out = (0.299*R + 0.587*G + 0.114*B)
        img_out = img_out.astype('uint8')
        return img_out
    else:
        print('Conversion can not be realized, as the image is not color')
converted = rgb2gray(img)
plt.figure()
plt.imshow(converted, cmap='gray'); plt.title('Converted image')

img_noise = converted + 0.4 * converted.std() * np.random.random(converted.shape)
plt.figure()
plt.imshow(img_noise, cmap='gray'); plt.title('Noisy image')

mediere = np.ones([3,3])/9

laplacian_mask1 = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])

laplacian_mask2 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

sobel_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

prewitt_mask = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

weighted = np.array([[1,2,1],[2,4,2],[1,2,1]])/16

print(mediere)

filter1 = sc.convolve(img_noise, mediere, mode = 'nearest')

filter2 = sc.convolve(img_noise, laplacian_mask1, mode = 'nearest')

filter3 = sc.convolve(img_noise, laplacian_mask2, mode = 'reflect')

filter4 = sc.convolve(img_noise, sobel_mask, mode = 'nearest')

filter5 = sc.convolve(img_noise, prewitt_mask, mode = 'nearest')

filter6 = sc.convolve(img_noise, weighted, mode = 'nearest')

plt.figure()
plt.imshow(filter1, cmap='gray'); plt.title('Filter 1 - Mediere')

plt.figure()
plt.imshow(converted + filter2, cmap='gray'); plt.title('Filter 2 - Laplacian_mask1')

plt.figure()
plt.imshow(filter3, cmap='gray'); plt.title('Filter 3 - Laplacian_mask2')

plt.figure()
plt.imshow(filter4, cmap='gray'); plt.title('Filter 4 - Sobel_mask')

plt.figure()
plt.imshow(filter5, cmap='gray'); plt.title('Filter 5 - Prewitt_mask')

plt.figure()
plt.imshow(filter6, cmap='gray'); plt.title('Filter 6 - Weighted')






###############################################################################








#citire imagine
path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\lena_impulsiv.jpg'
img1= plt.imread(path)
print(img1.shape)
plt.figure()
plt.imshow(img1,cmap = 'gray')
plt.show()

f_med=sc.median_filter(img1, size= [3,3])
plt.figure()
plt.imshow(f_med,cmap = 'gray')
plt.show()

#alte filtre (liniar)
#f_uniform=filter.uniform_filter(img_noise)
#f_gauss=sc.filter.gaussian_filter(img_noise,sigma=2)

# Applications: Common types of noise

# I. Write the below code lines  with common types of noise

# Different kind of imaging systems might give us different noise. 
# Here, we give an overview of three basic types of noise that are common in image processing applications:
# - Gaussian noise.
# - Random noise
# - Salt and Pepper noise (Impulse noise – only white pixels)

# Before we start with the generation of noise in images, we will give a brief method of how 
# we can generate random numbers from a Gaussian distribution or from a uniform distribution.
 
# The code below illustrates how we can obtain 2 random numbers from uniform distribution.
a = np.random.uniform(0,1)
b = np.random.uniform(0,1)

print(' a: ',a,'\n','b: ',b)


#In the similar manner we can get two random numbers from a normal distribution
a = np.random.normal(0,1)
b = np.random.normal(0,1)

print(' a: ',a,'\n','b: ',b)


# Generate Gaussian noise. 

# We may say that a Gaussian noise will be an independent identically distributed intensity level 
# drawn from a Gaussian distribution. 
# Note that here we use 1D Gaussian distribution. Commonly, it is determined with parameters μ and σ.
# The following code will generate a Gaussian noise.

# Let's first create a zero image with the same dimensions of the loaded image

image = cv2.imread(r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\mdb005.png')

gaussian_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)
cv2.imshow('All zero values',gaussian_noise)
cv2.waitKey()

# Now, we can set the pixel values as a Gaussian noise. 
# We have set a mean value to 128 and a standard deviation to 20.

cv2.randn(gaussian_noise, 128, 20)

cv2.imshow('Gaussian noise',gaussian_noise)
cv2.waitKey()
cv2.imwrite("Gaussian random noise.jpg",gaussian_noise)


# Random Uniform Noise

# In a similar way, we can create a random uniform noise. 
# In a similar manner we can create an image whose pixel values have random values drawn from an uniform distribution.

uniform_noise = np.zeros((image.shape[0], image.shape[1]),dtype=np.uint8)

cv2.randu(uniform_noise,0,255)
cv2.imshow('Uniform random noise',uniform_noise)
cv2.waitKey()
cv2.imwrite("Uniform random noise.jpg",uniform_noise)

#As we can see the uniform random noise is very similar to a Gaussian noise. 
#Here, the pixel values were set from 0 to 255.
#Next, we have to see how we can generate an impulse noise. This will be a black image with random white pixels. 
#There are many ways to implement such an image. For instance, we can actually post-process a “uniform_noise” image.  We can simply set a threshold value (binary thresholding) and convert an image into a set of black and white pixels. All pixels below a threshold (in our case 250 ) will become black (0), and those above this value will become white (255). By varying the values of a threshold we will get more or less white pixels (more or less noise).

# Generate salt and pepper noise
# And third important type of noise will be a black and pepper.
# Here we will due to a bit simplar visualization represent only a noise that has white pixels.
# One approach to do so is to let's say simply take a "uniform_noise" image.

# Set a threshold rule, where we will convert all pixels larger than a threshold to white (255) and 
# we will set the remaining to zero.

impulse_noise = uniform_noise.copy()

# Here a number 250 is defined as a threshold value.

# Obviously, if we want to increase a number of white pixels we will need to decrease it.

# Otherwise, we can increase it and in that way we will suppress the number of white pixels.

#ret,impulse_noise = cv2.threshold(uniform_noise,250,255,cv2.THRESH_BINARY)

cv2.imshow('Impuls noise',impulse_noise)
cv2.waitKey()
cv2.imwrite("Impuls noise.jpg",impulse_noise)

 
# Adding Noise to Images
# If images are just functions, then we can add two images similarly like we can add two functions. 
# Simply, every pixel value will be summed with the corresponding pixel value that has the same coordinates. 
# Of course, the images need to have the same dimensions.

image = cv2.imread(r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\mdb005.png',cv2.IMREAD_GRAYSCALE)
gaussian_noise = (gaussian_noise*0.5).astype(np.uint8)
noisy_image1 = cv2.add(image,gaussian_noise)

cv2.imshow('Noisy image - Gaussian noise',noisy_image1)
cv2.waitKey()
cv2.imwrite("Noisy image1.jpg",noisy_image1)

uniform_noise = (uniform_noise*0.5).astype(np.uint8)
noisy_image2 = cv2.add(image,uniform_noise)

cv2.imshow('Noisy image - Uniform noise',noisy_image2)
cv2.waitKey()
cv2.imwrite("Noisy image2.jpg",noisy_image2)

impulse_noise = (impulse_noise*0.5).astype(np.uint8)
noisy_image3 = cv2.add(image,impulse_noise)

cv2.imshow('Noisy image - Impuls noise',noisy_image3)
cv2.waitKey()
cv2.imwrite("Noisy image3.jpg",noisy_image3)

# Median Filter
# Next, we are going to present a median filter and basic image processing. 
# Do you know what a median operator/function does? Yes, you are right! It is that simple. 
# A median filter just scrolls across the image, and for all the elements that are overlapping with the filter, 
# position outputs the median element.

# Let’s have a look at the illustration of a 2D median filter. 
# Imagine that those are the pixel values in the image as shown in the Figure below. 
# This means that the filter is centered at the value of 90. 
#In this case, we use a 3 x 3 filter size, so all nine values we will sort in the ascending order. The median value is 27 and, that is the output value for this location of the filter. 
#In this case, a value of 90 (an extreme value in this example), will be replaced with a number 27.


#Median Filter

#Let’s have a look at the illustration of a 2D median filter. 
#Imagine that those are the pixel values in the image as shown in the Figure below. 
#This means that the filter is centered at the value of 90. In this case, we use a 3 x 3 filter size, 
#so all nine values we will sort in the ascending order. The median value is 27 and, 
#that is the output value for this location of the filter. 
#In this case, a value of 90 (an extreme value in this example), will be replaced with a number 27.

#Applying a simple median filter.

#There are, of course, as we will see, more advanced filters.
#However, not that even a simple median filter can do, rather effective job.
#This is true especially, for ? Well, you guess it.

blurred1 = cv2.medianBlur(noisy_image1, 3)
cv2.imshow('Median filter - Gaussian noise',blurred1)
cv2.waitKey()
cv2.imwrite("Median filter - Gaussian noise.jpg",blurred1)

blurred2 = cv2.medianBlur(noisy_image2, 3)
cv2.imshow('Median filter - Uniform noise',blurred2)
cv2.waitKey()
cv2.imwrite("Median filter - Uniform noise.jpg",blurred2)

blurred3 = cv2.medianBlur(noisy_image3, 3)
cv2.imshow('Median filter - Impuls noise',blurred3)
cv2.waitKey()
cv2.imwrite("Median filter - Impuls noise.jpg",blurred3)


# Part II: DICOM Standard
#!pip install pydicom

#importarea bibliotecii necesare accesarii fisierelor din Google Drive
#from google.colab import drive

#accesarea contului propriu de Google Drive
#drive.mount('/content/drive')

import pydicom as dicom
#import matplotlib.pyplot as plt
#cale = 'drive/My Drive/Colab Notebooks/imagini/dicom1/img01.dcm'
path = r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\img01.dcm'
dataset = dicom.read_file(path)
imageDataset = dataset.pixel_array
plt.figure()
plt.imshow(imageDataset, cmap = 'gray')
plt.show()

type(dataset)

#afisare tot header
print(dataset)

#afsarea unei singure informatii din header (data element)
print(dataset.PatientName)

dataset[0x0010,0x0010]

#anonimizare fisier dicom in functei de ce dorim sa ascundem
for i,row in dataset.items():
  if row.VR == "PN":
    print(row.VR)
    print(dataset[i].value)
    dataset[i].value = "anonymous"
    print(dataset[i].value)

print(dataset)

help(dicom.dcmwrite)

dicom.dcmwrite(r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\img01_anon.dcm',dataset)

path1 = r'C:\Users\brian\OneDrive\Desktop\MI\Lab6_MI\Drive\Lab 6\img01_anon.dcm'
dataset1 = dicom.read_file(path1)
imageDataset1 = dataset1.pixel_array
plt.figure()
plt.imshow(imageDataset1, cmap = 'gray')
plt.show()
print(dataset1.PatientName)

type(imageDataset1[0,0])

imageDataset.max()

imageDataset1.shape