import numpy as np
import matplotlib.pyplot as plt

# read_data (type:dict)
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:           #rb = read binary(2進法)、with関数を使うことでopenするだけでなく、closeも勝手にやってくれる
        dict = pickle.load(fo, encoding='bytes')   #文字形式をbytesで読み込む
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
# order of values. The axe
# s parameter describes how to reorder the data.
plt.imshow(np.transpose(np.reshape(firstimage, (3, 32, 32)), axes=(1, 2, 0)))
plt.show()



# Conv
def Conv(x,w,pad,stride):
    #x = np.random.randint(-3, 3, (4,4))
    #w = np.random.randint(-1, 1, (3,3))
    #pad = 1
    #stride = 1
    '''
    out = []
    for k in range(0, len(x)+2*pad -len(w)+1, stride):
        out_row = []
        for i in range(0, len(x)+2*pad -len(w)+1, stride):
            z = 0
            for l in range(len(w)):
                for j in range(len(w)):
                    if((0 <= k+l-pad) and (k+l-pad < len(x))):
                        if((0 <= i+j-pad) and (i+j-pad < len(x))):
                            z += x[k+l-pad][i+j-pad] * w[l][j]
            out_row.append(z)
        out.append(out_row)  
        
    out = np.array(out)
    '''
    lenz = (len(x)+2*pad -len(w))//stride+1
    z = np.zeros((lenz,lenz))
    for i in range(len(w)):
        for j in range(len(w)):
            z[max(0,pad-i):min(len(z),len(z)+pad-i),max(0,pad-j):min(len(z),len(z)+pad-j)] += x[max(0,-pad+i):min(len(x),-1+pad+len(x)), max(0, -pad+j):min(len(x),-1+j+len(x))] * w[i,j]
    
    print(z)
    return(z)

#ReLu
def ReLu(x):
    return np.maximum(0,x)

# MaxPooling
def MaxPooling(out):
    out_2 = []
    #out_2 = np.maximum(conda activate tf),np.maximum(out[1::2,0::2], out[1::2,1::2]))  # 0::2 0番目の値から２の倍数をとってくる
    out_2 = np.maximum.reduce([out[0::2,0::2], out[0::2,1::2],out[1::2,0::2], out[1::2,1::2]])
    print(out_2)
    return out_2

#Fully Connected
def FC(out_2,weight,bias):
    out_2 = np.ravel(out_2)       #１次元配列にする
    #weight = np.random.rand(2,len(out_2))
    #bias = np.random.rand(2)

    fc = weight.dot(out_2) + bias
    fc = np.array(fc)
    print(fc)
    return fc
    
# Softmax
def softmax(x):
    m = np.max(x)  # To avoid the overflow of exp(x)
    return np.exp(x-m)/np.sum(np.exp(x-m))


np.random.seed(0)
x = np.random.randint(-3, 3, (4,4))
w = np.random.randint(-1, 1, (3,3))
pad = 1
stride = 1
#weight = np.random.rand(2,len(out_2))
weight = np.random.rand(2,4)
bias = np.random.rand(2)

conv = Conv(x,w,pad,stride,bias)
relu = ReLu(conv)
mp = MaxPooling(relu)
fc = FC(mp,weight,bias)
sm = softmax(fc)
print(sm)