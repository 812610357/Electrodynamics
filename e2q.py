from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

######## 定义求解区域 ########
xMin = -0.9995
xMax = 0.9995
yMin = -0.9995
yMax = 0.9995
d = np.array([1e-3, 1e-3])  # [y,x]
x = np.arange(xMin, xMax+d[1], d[1])
y = np.arange(yMin, yMax+d[0], d[0])
xx, yy = np.meshgrid(x, y)
size = np.array([len(y), len(x)], dtype='int')

######## 定义电场分布 ########
epsilon0 = 1
'''
Ex = np.pi*np.cos(np.pi*xx)*np.sin(np.pi*yy)/epsilon0
Ey = np.pi*np.sin(np.pi*xx)*np.cos(np.pi*yy)/epsilon0
'''
Ex=10**6*xx/(4*np.pi*(xx**2+yy**2)**(3/2))
Ey=10**6*yy/(4*np.pi*(xx**2+yy**2)**(3/2))
Exy = np.stack((Ey, Ex), axis=2)

######## 三点差分-计算散度 ########
rho1 = (Exy[1:-1, 2:, 1]-Exy[1:-1, :-2, 1])/2/d[1]*epsilon0
rho2 = (Exy[2:, 1:-1, 0]-Exy[:-2, 1:-1, 0])/2/d[0]*epsilon0
rho = rho1+rho2

######## 计算误差分析 ########
rho0 = 2*np.pi**2*np.sin(np.pi*xx)*np.sin(np.pi*yy)  # 该系统的解析解
error = np.abs(rho-rho0[1:-1, 1:-1])

np.argmax
def plotChargeDistribution(rho, title, cmap='viridis'):
    plt.pcolor(x[1:-1], y[1:-1], rho, cmap=cmap,norm=LogNorm())
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    pass


plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.axis('equal')
plt.axis([-1, 1, -1, 1])
plotChargeDistribution(rho, 'Charge distribution')
plt.subplot(1, 2, 2)
plt.axis('equal')
plt.axis([-0.02, 0.02, -0.02, 0.02])
plotChargeDistribution(rho, 'Detial area')
plt.show()
