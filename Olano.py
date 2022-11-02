### Name: Gamaliel Paul Olano
### Student Number: 2019-04251
### Section: CoE 197 M-THY

### Removing Projective Distortion on Images
### Inputs: Image file, 4 points on image
### Output: Undistorted (affine) image file

import os
import matplotlib

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import scipy
from scipy import linalg, matrix
from scipy.interpolate import  griddata

matplotlib.use("TkAgg") #For 4pts input interface

## Get Image File
cwd = os.getcwd()  #Get current directory
image_name = input("Enter filename: ")   #Input projective image
image_directory = os.path.join(cwd, image_name)
image = np.array(Image.open(image_directory))
##

## Select four points
plt.imshow(image)     
pt4 = plt.ginput(4)
plt.close()
print("Please wait while file is being processed...")
pt4.append(pt4[0]) #Make array circular
##

## Sort 4 Projective Points
pt4_proj = np.zeros((4,3))           
for ii in range(4):         
    pt4_proj[ii][0] = pt4[ii][0]; #x
    pt4_proj[ii][1] = pt4[ii][1]; #y

sortedArrx = pt4_proj[pt4_proj[:,0].argsort()] #sorted by x
left = sortedArrx[:2] #2 left vertices
right = sortedArrx[2:] #2 right vertices

sortedArry = pt4_proj[pt4_proj[:,1].argsort()] #sorted by y
upper = sortedArry[2:] #2 upper vertices
lower = sortedArry[:2] #2 lower vertices

for i in range(2): ##find upperleft
    for j in range(2):
        if (np.array_equal(left[i],upper[j])):
            pt4_proj[3] = left[i]

for i in range(2): ##find upperright
    for j in range(2):
        if (np.array_equal(right[i],upper[j])):
            pt4_proj[2] = right[i]
            
for i in range(2): ## find lowerright
    for j in range(2):
        if (np.array_equal(right[i],lower[j])):
            pt4_proj[1] = right[i]
            
for i in range(2): ## find lowerleft
    for j in range(2):
        if (np.array_equal(left[i],lower[j])):
            pt4_proj[0] = left[i]
            
pt4_proj = np.insert(pt4_proj, 4, pt4_proj[0], axis = 0)
##

## Generate affine using projective
pt4_affi = np.zeros((5,3))     
pt4_affi[0,0] = pt4_proj[0,0] #Lowerleft
pt4_affi[0,1] = pt4_proj[0,1] 
pt4_affi[1,0] = pt4_proj[1,0] #lowerright
pt4_affi[1,1] = pt4_proj[0,1] 
pt4_affi[2,0] = pt4_proj[1,0] #upperright
pt4_affi[2,1] = pt4_proj[3,1] 
pt4_affi[3,0] = pt4_proj[0,0] #upperleft
pt4_affi[3,1] = pt4_proj[3,1] 
pt4_affi[4,:] = pt4_affi[0,:] #Make array circular
##

## Calculate homography
x_1 = [pt4_proj[0][0],pt4_affi[0][0]]
y_1 = [pt4_proj[0][1],pt4_affi[0][1]]
x_2 = [pt4_proj[1][0],pt4_affi[1][0]]
y_2 = [pt4_proj[1][1],pt4_affi[1][1]]
x_3 = [pt4_proj[2][0],pt4_affi[2][0]]
y_3 = [pt4_proj[2][1],pt4_affi[2][1]]
x_4 = [pt4_proj[3][0],pt4_affi[3][0]]
y_4 = [pt4_proj[3][1],pt4_affi[3][1]]

P = np.array([
    [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]],
    [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]],
    [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0]*x_2[1], y_2[0]*x_2[1], x_2[1]],
    [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0]*y_2[1], y_2[0]*y_2[1], y_2[1]],
    [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0]*x_3[1], y_3[0]*x_3[1], x_3[1]],
    [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0]*y_3[1], y_3[0]*y_3[1], y_3[1]],
    [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0]*x_4[1], y_4[0]*x_4[1], x_4[1]],
    [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0]*y_4[1], y_4[0]*y_4[1], y_4[1]],
    ])

[U, S, Vt] = np.linalg.svd(P)
hh = Vt[-1].reshape(3, 3)
##

## Transform Image
mm,nn = image.shape[0],image.shape[0]

W = np.array([[1, nn, nn, 1 ],[1, 1, mm, mm],[ 1, 1, 1, 1]])
ws = np.dot(hh,W)
### scaling
xx = np.vstack((ws[2,:],ws[2,:],ws[2,:]))
wsX =  np.round(ws/xx)
bounds = [np.min(wsX[1,:]), np.max(wsX[1,:]),np.min(wsX[0,:]), np.max(wsX[0,:])]
    
nrows = bounds[1] - bounds[0]
ncols = bounds[3] - bounds[2]
s = max(nn,mm)/max(nrows,ncols)
scale = np.array([[s, 0, 0],[0, s, 0], [0, 0, 1]])
trasf = scale@hh
trasf_prec =  np.linalg.inv(trasf)

W = np.array([[1, nn, nn, 1 ],[1, 1, mm, mm],[ 1, 1, 1, 1]])
ws = np.dot(trasf,W)

xx = np.vstack((ws[2,:],ws[2,:],ws[2,:]))
wsX =  np.round(ws/xx)
bounds = [np.min(wsX[1,:]), np.max(wsX[1,:]),np.min(wsX[0,:]), np.max(wsX[0,:])]


nrows = (bounds[1] - bounds[0]).astype(int)
ncols = (bounds[3] - bounds[2]).astype(int)

if max(mm,nn)>1000:
    kk =6
else: kk =5
nsamples = 10**kk 

xx  = np.linspace(1, ncols, ncols)
yy  = np.linspace(1, nrows, nrows)
[xi,yi] = np.meshgrid(xx,yy) 
a0 = np.reshape(xi, -1,order ='F')+bounds[2]
a1 = np.reshape(yi,-1, order ='F')+bounds[0]
a2 = np.ones((ncols*nrows))
uv = np.vstack((a0.T,a1.T,a2.T)) 
new_trasf = np.dot(trasf_prec,uv)
val_normalization = np.vstack((new_trasf[2,:],new_trasf[2,:],new_trasf[2,:]))


newT = new_trasf/val_normalization


xi = np.reshape(newT[0,:],(nrows,ncols),order ='F') 
yi = np.reshape(newT[1,:],(nrows,ncols),order ='F')
cols = image.shape[1]
rows = image.shape[0]
xxq  = np.linspace(1, rows, rows).astype(int)
yyq  = np.linspace(1, cols, cols).astype(int)
[x,y] = np.meshgrid(yyq,xxq) 
x = (x - 1).astype(int)
y = (y - 1).astype(int) 

ix = np.random.randint(cols, size=nsamples)
iy = np.random.randint(rows, size=nsamples)
samples = image[iy,ix]

int_im = griddata((iy,ix), samples, (yi,xi))
##

## Plotting
fig = plt.figure(figsize=(8, 8))
fig.add_subplot(1, 2, 1)
plt.imshow(image)
fig.add_subplot(1, 2, 2)
plt.imshow(int_im.astype(np.uint8))
plt.show()
##