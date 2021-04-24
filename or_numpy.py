import numpy as np

def sig(s):
    return 1 / (1 + np.exp(-s))

x = np.array([[1, 0, 0],[1, 0, 1], [1, 1, 0], [1, 1, 1]])
t = np.array([[0, 1],[1, 0], [1, 0], [1, 0]])

w = np.ones((2, 3))   #3行2列の行列全ての成分0
dw = np.zeros(w.shape)

rho = 0.1

while True:

    u = w.dot(x.T) # .T : 転置行列
    o = sig(u)
    dw = ((o - t.T) * (1 - o) * o).dot(x)

    w -= rho * dw
    
    if np.sqrt(np.sum(dw * dw)) < 0.0001:
        break
    E = 0.5 * np.sum((o-t.T) ** 2)
    print(np.sum(dw * dw), E)
print(w)