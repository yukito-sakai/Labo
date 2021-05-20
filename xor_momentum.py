import numpy as np

def sig(s):
    return 1 / (1 + np.exp(-s))

x = np.array([[1, 0, 0],[1, 0, 1], [1, 1, 0], [1, 1, 1]])
t = np.array([[0, 1],[1, 0], [1, 0], [0, 1]])

np.random.seed(0)
m = 5
w1 = np.random.rand(m, 3)   
w2 = np.random.rand(2, m+1)
dw1 = np.zeros(w1.shape)
dw2 = np.zeros(w2.shape)

rho = 0.3
mu = 0.9
v = 0

while True:

    u1 = w1.dot(x.T) 
    o1 = sig(u1)
    o1 = np.r_[np.ones((1,o1.shape[1])), o1]
    
    u2 = w2.dot(o1) 
    o2 = sig(u2)

    dw2 = ((o2 - t.T) * (1 - o2) * o2).dot(o1.T)
    dw1 = ((((o2 - t.T) * (1 - o2) * o2).T).dot(w2).T * ((1 - o1) * o1)).dot(x)[1:]
    
    v2 = mu * v2 - rho * dw2 # integrate velocity
    x2 += v2
    
    w2 -= rho * dw2
    #print(dw1)
    w1 -= rho * dw1
    
    
    if np.sqrt(np.sum(dw2 * dw2) + np.sum(dw1 * dw1)) < 0.0001:
        break
    E = 0.5 * np.sum((o2-t.T) ** 2)
    print(np.sum(dw1 * dw1), np.sum(dw2 * dw2), E)
print(w1)    
print(w2)