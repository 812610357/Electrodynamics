import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import sparse
from scipy.sparse.linalg.dsolve.linsolve import spsolve

######## 定义求解区域 ########
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([1e-2, 1e-2])
epsilon0 = 8.85418e-12
x = np.arange(xMin, xMax+d[0], d[0])
y = np.arange(yMin, yMax+d[1], d[1])
size = np.array([len(x), len(y)], dtype='int')

######## 定义电荷分布 ########
rho = np.zeros((size[0], size[1]))

rho[(size[0]-1)//2-50, (size[1]-1)//2] = 1000
rho[(size[0]-1)//2+50, (size[1]-1)//2] = -1000

'''
rho[(size[0]-1)//2-5:(size[0]-1)//2+6, (size[1]-1)//2-5:(size[0]-1)//2+6]=np.ones((11,11))
'''
'''
rho[(size[0]-1)//2-2:(size[0]-1)//2+3, (size[1]-1)//2] = np.ones((5))
'''
######## 定义边界条件 ########
boundary = np.zeros((size[0], size[1]))
'''
######## 五点差分-松弛迭代-求解拉普拉斯方程 ########
phi = boundary
w = 1.5
error = np.Inf
step = 0
while error > 1e8:
    def sor(phi, rho, epsilon0, w):
        b = d[0]/d[1]
        B2 = b*b
        DX2 = d[0]*2
        C = w/(2*(1+B2))
        for i in range(1, size[0]-1):
            for j in range(1, size[1]-1):
                phi[i, j] = (1-w)*phi[i, j]+C*(phi[i+1, j]+phi[i-1, j] + B2*(phi[i, j+1] + phi[i, j-1])+DX2*rho[i, j]/epsilon0)
        return phi
    phi1 = sor(np.copy(phi), rho, epsilon0, w)
    error = np.max(phi1-phi)
    phi = np.copy(phi1)
    print('error=%e' % error)
    step += 1
print(step)
'''
######## 五点差分-共轭梯度-求解拉普拉斯方程 ########


def coefficientMatrix():
    #### 主对角线 ####
    C = np.ones(sizeA)*-2*(1/d[0]**2+1/d[1]**2)
    #### x方向差分 ####
    Dx = np.ones(sizeA-1)
    Dx[np.arange(1, size[0], 1, dtype='int')*size[0]-1] = 0
    #### y方向差分 ####
    Dy = np.ones(sizeA-size[0])
    A = sparse.diags([C, Dx, Dx, Dy, Dy], [0, 1, -1, size[0], -size[0]])
    return A


def conj_grad_method(A, b, x):
    r = b - sparse.dia_matrix.dot(A, x)
    d = r
    for i in range(2*A.shape[0]):
        Ad = sparse.dia_matrix.dot(A, d)
        alpha = np.dot(r.T, r) / np.dot(d.T, Ad)
        x = x + alpha*d
        beta = np.dot((r - alpha*Ad).T, r - alpha*Ad) / np.dot(r.T, r)
        r = r - alpha*Ad
        d = r + beta*d
        print('r=%e' % np.max(np.abs(r)))
        if np.max(r) < 1e-161:
            break
    return x


sizeA = size[0]*size[1]
A = coefficientMatrix()
b = np.reshape(-rho/epsilon0, (sizeA, 1))
x0 = np.zeros((sizeA, 1))
r = conj_grad_method(A, b, x0)
#r = spsolve(A, b)
phi = np.reshape(r, (size[0], size[1]))

######## 计算电场强度 ########
Ex = (phi[2:, 1:-1]-phi[:-2, 1:-1])/2/d[0]
Ey = (phi[1:-1, 2:]-phi[1:-1, :-2])/2/d[1]
Exy = -np.stack((Ex, Ey), axis=2)

######## 电场的可视化 ########


def plotElectricFieldIntensity(Exy):
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x[1:-1], y[1:-1], E.T, cmap='jet', norm=LogNorm())
    plt.colorbar()
    pass


def plotElectricFieldDirection(Exy):
    xi = np.arange(0, 20, 1, dtype='int') * (size[0]-1)//20+(size[0]-1)//40
    yi = np.arange(0, 20, 1, dtype='int') * (size[1]-1)//20+(size[1]-1)//40
    xii, yii = np.meshgrid(xi, yi)
    xa = x[xi]
    ya = y[yi]
    E = np.linalg.norm(Exy[xii-1, yii-1, :], ord=2, axis=2)
    dx = Exy[xii-1, yii-1, 0]/E
    dy = Exy[xii-1, yii-1, 1]/E
    plt.quiver(xa, ya, dx, dy)
    pass


plt.axis('equal')
plotElectricFieldIntensity(Exy)
plotElectricFieldDirection(Exy)
plt.show()
