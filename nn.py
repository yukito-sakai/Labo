from sys import version
from typing import ValuesView
from numpy.lib.function_base import kaiser
from numpy.testing._private.nosetester import _numpy_tester


import random

import numpy as np
from scipy.misc import imread, imsave, imresize
img = imread('assets/cat.jpg')
print(img.dtype, img.shape) 
mg_tinted = img * [1, 0.95, 0.9]
img_tinted = imresize(img_tinted, (4, 4))
imsave('assets/cat_tinted.jpg', img_tinted)

def softmax(s):
    max = np.amax(np.array(s))
    return (np.exp(s) / max) / (np.sum(np.exp(s)) / max)


#CONV_LAYER
#4*4*1
w1 = 4
h1 = 4
d1 = 1
weight = 10
bias = 1

k = 3
f = 3
s = random.randrange(1, 3) #stride
p = random.randrange(3) #padding
w2 = (w1 - f - 2*p) / s + 1
h2 = (h1 - f - 2*p) / s + 1
d2 = k

for k in range (d1):
    for j in range(h1 + p):
        for i in range(w1 + p):
            v[i,j,k] =  np.sum(x[i*s:f+i*s,j*s:f+j*s,:] * weight) + bias
            ｚ
            





