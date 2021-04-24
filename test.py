import numpy as np

x = [1, 2, 3]
y = [4, 5, 6]

z = []
for i, j in zip(x, y):
    z.append(i + j)

print(z)
print(zip(x, y))                             # = <zip object at 0x000001D99A49D400>
print(list(zip(x, y)))                       # = [(1, 4), (2, 5), (3, 6)]

print(np.array([1, 2]))                      # = [1 2]
print(np.array(x))                           # = [1 2 3]
print(np.array([1, 2]) + np.array([5, 6]))   # = [6 8]

x = np.array([1, 2, 3])                      
y = np.array([4, 5, 6])

print(x + y)                                 # = [5 7 9]
print(x * y)                                 # = [ 4 10 18]
print(np.sin(x))                             # = [0.84147098 0.90929743 0.14112001]
print(-x)                                    # = [-1 -2 -3]
print(x.dot(y))                              # xとyのinner = 32

print(np.argmax(y))                          # = 2
print(y)                                     # = [4 5 6]
print(np.argmin(y))                          # = 0
print(y[np.argmin(y)])                       # = 4
print(y[np.argmax(y)])                       # = 6

print(x.size)                                # = 3
print(x.shape)                               # = (3,) (タプル)
print((3) == (3, ))                          # False
print((3) == (3, )[0])                       # True



a = np.array([[1, 2, 3],[4, 5, 6]])
print(a)                                     #[[1 2 3]
                                              #[4 5 6]]
print(a**2)                                  #[[ 1  4  9] 
                                              #[16 25 36]]
print(np.exp(a))
print(np.max(a))                             # = 6
print(np.argmax(a))                          # = 5 (2行を1行だと思うと5番目の36が一番大きい)
print(a[0])                                  # = [1 2 3]
print(a[1])                                  # = [4 5 6]

print(a.size)                                # = 6
print(a.shape)                               # = (2, 3)
print(a[1, 1])                               # = 5  (1行1列（0から数えて）)
print(a[:, 1])                               # = [2 5] (1列目)
print(a[0, ])                                # = [1 2 3]
print(a[0, :])                               # = [1 2 3]

print(a * x)                                 # = 2*3行列 * 1*3行列は成分同士のかけ算
print(a.dot(y))                              # = [32 37]　行列の積（2*3 * 3*1）
print(np.matmul(a, y))                       # = [32 37]　行列の積（2*3 * 3*1）
print(np.sum(a * x))                            
print(np.sum(a * x, axis=0))                 # =  同じ列を(行方向に）足し合わせる
print(np.sum(a * x, axis=1))                 # =  同じ行を(行方向に）足し合わせる













