import numpy as np
import matplotlib.pyplot as plt

# read_data (type:dict)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# load data from file data_batch_1
d = unpickle('./cifar-10-batches-py/data_batch_1')

# d is a dict. b'data' key contains the image data
firstimage = d[b'data'][0]

# Plotting the image data
# The image data is 3 x 32 x 32 = 3072 values stored in one-dimensional array,
# first 32 x 32 values for red, followed by 32 x 32 values for green, and then
# blue.
# So we first convert it to a 3 dimensional array using np.reshape.
# But to plot the image, we need the red, green, blue values for each pixel
# together, in the shape 32 x 32 x 3, so we np.transpose the data to change the
# order of values. The axes parameter describes how to reorder the data.
plt.imshow(np.transpose(np.reshape(firstimage, (3, 32, 32)), axes=(1, 2, 0)))
plt.show()


import numpy as np

# Conv
x = np.array([1, 2, 3, 4, 5])
w = np.array([1, 1, 1])
out = []

pad = 1
stride = 1

for i in range(0, len(x)+2*pad -len(w)+1, stride):
    z = 0
    for j in range(len(w)):
        if((0 <= i+j-pad) and (i+j-pad < len(x))):
            z += x[i+j-pad] * w[j]
    out.append(z)
print(out)
#for文をフィルター内のみにしたもの
z = []
for i in range(len(w)):
    
    z[:] += x[i:i+len(w)] * w[:]

print(out)

#for i in range(len(w)):
  #  z += x[:] * w[i]
   # print(z)

# MaxPooling
x = np.array(list(range(10)))  # x=[0,1,2,3,4,5,6,7,8,9]
out = []

#for i in range(0, len(x), 2):
#   out.append(np.max(x[i:][:2]))  # x[i:i+2] is OK
#=print(out)

out = np.maximum(x[0::2], x[1::2])  # 0::2 0番目の値から２の倍数をとってくる
print(out)

# Softmax
def softmax(x):
    m = np.max(x)  # To avoid the overflow of exp(x)
    return np.exp(x-m)/np.sum(np.exp(x-m))

x = np.array([1, 2, 3])
y = softmax(x)
print(y)