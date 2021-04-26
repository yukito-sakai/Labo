from math import *

def sigmoid(u):
    return 1 / (1 + exp(-u))

x = ((1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))
t = (0, 1, 1, 1)
w = [[-0.5, 0.5, 0.5], [-1, 1., 1.]]
dw = [[0, 0, 0], [0, 0, 0]]
delta = 0.2

while True:
    E = 0
    dw = [[0, 0, 0], [0, 0, 0]]
    dw_sum = 0
    for a in range(4):                                                  
        for k in range(2):
            u = w[k][0] + w[k][1] * x[a][1] + w[k][2] * x[a][2]
            y = sigmoid(u) 
            for l in range(3):     
                dw[k][l] += (y - t[a]) * (1 - y) * y * x[a][l]
                dw_sum += dw[k][l]**2 
            E = E + 1/2 * (y - t[a])**2

    for k in range(2):                                                  
        for l in range(3):
            w[k][l] -=  delta * dw[k][l]
            print('w[', k, ',', l,'] = ', w[k][l])
    if dw_sum < 0.00001:
        break
    print('E = ', E) 