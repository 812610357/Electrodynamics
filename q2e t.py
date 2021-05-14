import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

######## 设置求解参数 ########
postIterError = 1e1  # 后平滑共轭梯度迭代误差
mainIterTimes = 4    # 多重网格迭代次数，即网格粗化等级
preIterTimes = 3     # 预平滑共轭梯度迭代次数

######## 定义求解区域 ########
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([1e-3, 1e-3])  # [y,x]
x = np.arange(xMin, xMax+d[1], d[1])
y = np.arange(yMin, yMax+d[0], d[0])
xx, yy = np.meshgrid(x, y)

epsilon0 = 8.85418e-12
phi0=np.sin(np.pi*xx)*np.sin(np.pi*yy)/epsilon0
Ex = np.pi*np.cos(np.pi*xx)*np.sin(np.pi*yy)/epsilon0
Ey = np.pi*np.sin(np.pi*xx)*np.cos(np.pi*yy)/epsilon0
Exy0 = -np.stack((Ex, Ey), axis=2)[1:-1,1:-1,:]

def plotElectricPotential():
    '''
    绘制电势大小
    '''
    plt.pcolor(x, y, phi, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential distribution')
    pass


def plotEquipotentialSurface():
    '''
    绘制等势线
    '''
    plt.contour(x, y, phi, 10, colors='k',
                linestyles='dashed', linewidths=0.5)
    pass


def plotElectricFieldIntensity():
    '''
    绘制电场强度大小
    '''
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x[1:-1], y[1:-1], E, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Field distribution')
    pass


def plotElectricFieldDirection():
    '''
    绘制电场强度方向
    '''
    xi = np.arange(0, 20, 1, dtype='int') * (len(x)-1)//20+(len(x)-1)//40
    yi = np.arange(0, 20, 1, dtype='int') * (len(y)-1)//20+(len(y)-1)//40
    xii, yii = np.meshgrid(xi, yi)
    xa = x[xi]
    ya = y[yi]
    E = np.linalg.norm(Exy[yii-1, xii-1, :], ord=2, axis=2)
    dx = Exy[yii-1, xii-1, 1]/E
    dy = Exy[yii-1, xii-1, 0]/E
    plt.quiver(xa, ya, dx, dy)
    pass


plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.axis('equal')
plotElectricPotential()
plotEquipotentialSurface()
plt.subplot(1, 2, 2)
plt.axis('equal')
plotElectricFieldIntensity()
plotElectricFieldDirection()
plt.show()

