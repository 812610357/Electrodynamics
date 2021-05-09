import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

######## 定义求解区域 ########
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([2e-3, 2e-3])
x = np.arange(xMin, xMax+d[0], d[0])
y = np.arange(yMin, yMax+d[1], d[1])
yy, xx = np.meshgrid(x, y)
size = np.array([len(x), len(y)], dtype='int')

######## 定义电场分布 ########
epsilon0 = 8.85418e-12
Ex = np.pi*np.cos(np.pi*xx)*np.sin(np.pi*yy)/epsilon0
Ey = np.pi*np.sin(np.pi*xx)*np.cos(np.pi*yy)/epsilon0
Exy = -np.stack((Ex, Ey), axis=2)

######## 三点差分-计算散度 ########

rho0 = (Exy[2:, 1:-1, 0]-Exy[:-2, 1:-1, 0])/2/d[0]
rho1 = (Exy[1:-1, 2:, 1]-Exy[1:-1, :-2, 1])/2/d[1]
rho = rho0+rho1


def plotChargeDistribution():
    plt.pcolor(x[1:-1], y[1:-1], rho.T, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Charge distribution')
    pass


plt.figure(figsize=(5, 4))
plt.axis('equal')
plotChargeDistribution()
plt.savefig('./e2q.pdf')
