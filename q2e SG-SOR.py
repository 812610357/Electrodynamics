import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy import sparse
from scipy.sparse.linalg.dsolve import spsolve
import time

######## 定义求解区域 ########
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([1e-2, 1e-2])
epsilon0 = 1
x = np.arange(xMin, xMax+d[0], d[0])
y = np.arange(yMin, yMax+d[1], d[1])
xx, yy = np.meshgrid(x, y)
size = np.array([len(x), len(y)], dtype='int')
error=np.array([])

######## 定义电荷分布 ########
rho = np.zeros((size[0], size[1]))
rho = 2*np.pi**2*np.sin(np.pi*xx)*np.sin(np.pi*yy)
'''
rho[(size[0]-1)//2-20, (size[1]-1)//2] = 1
rho[(size[0]-1)//2+20, (size[1]-1)//2] = -1
'''
'''
rho[(size[0]-1)//2-5:(size[0]-1)//2+6, (size[1]-1)//2-5:(size[0]-1)//2+6]=np.ones((11,11))
'''
'''
rho[(size[0]-1)//2-20:(size[0]-1)//2+21, (size[1]-1)//2] = np.ones((41))
'''


######## 五点差分-松弛迭代-求解拉普拉斯方程 ########
phi = np.zeros((size[0], size[1]))
t = time.time()
w = 1.5
Error = np.Inf
step = 0
while Error > 1e-1:
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
    Error = np.max(phi1-phi)
    error=np.append(error,Error)
    phi = np.copy(phi1)
    step += 1
    print('step=%d,error=%e' % (step,Error))

print('共轭梯度', time.time()-t)

######## 计算电场强度 ########
Ex = (phi[2:, 1:-1]-phi[:-2, 1:-1])/2/d[0]
Ey = (phi[1:-1, 2:]-phi[1:-1, :-2])/2/d[1]
Exy = -np.stack((Ex, Ey), axis=2)

######## 电场的可视化 ########


def plotElectricFieldIntensity(Exy):
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x[1:-1], y[1:-1], E.T, cmap='jet')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar().set_label('|E|')
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


def writecsv(data):  # 导出线条
    dataframe = pd.DataFrame(data={'error': data})
    dataframe.to_csv("error-SG-SOR.csv",
                     index=False, mode='w', sep=',')
    pass

writecsv(error)