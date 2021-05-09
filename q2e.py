import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import time

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

######## 定义电荷分布 ########
epsilon0 = 8.85418e-12
rho = 2*np.pi**2*np.sin(np.pi*xx)*np.sin(np.pi*yy)

######## 五点差分-共轭梯度-求解拉普拉斯方程 ########
t = time.time()


def coefficientMatrix():
    '''
    根据五点差分法，使用稀疏矩阵，初始化系数矩阵，展平到一维空间
    '''
    #### 主对角线 ####
    C = np.ones(sizeA)*-2*(1/d[0]**2+1/d[1]**2)
    #### x方向差分 ####
    Dx = np.ones(sizeA-1)/d[0]**2
    Dx[np.arange(1, size[0], 1, dtype='int')*size[0]-1] = 0
    #### y方向差分 ####
    Dy = np.ones(sizeA-size[0])/d[1]**2
    A = sparse.diags([C, Dx, Dx, Dy, Dy], [0, 1, -1, size[0], -size[0]])
    return A


def leftBoundary(phi):
    '''
    x=xMin，左侧一类边值条件
    '''
    b[:size[0]] = phi
    A.data[0, :size[0]] = 1
    A.data[[1, 2], :size[0]] = 0
    A.data[3, :size[0]*2] = 0
    pass


def rightBoundary(phi):
    '''
    x=xMin，左侧一类边值条件
    '''
    b[-size[0]:] = phi
    A.data[0, -size[0]:] = 1
    A.data[[1, 2], -size[0]:] = 0
    A.data[4, -size[0]*2:] = 0
    pass


def bottomBoundary(phi):
    '''
    y=yMin，下侧一类边值条件
    '''
    b[np.arange(size[1])*size[0]] = phi
    A.data[0, np.arange(size[1])*size[0]] = 1
    A.data[1, np.arange(size[1])*size[0]+1] = 0
    A.data[3, np.arange(size[1])*size[0]] = 0
    A.data[4, np.arange(size[1])*size[0]] = 0
    pass


def topBoundary(phi):
    '''
    y=yMin，上侧一类边值条件
    '''
    b[np.arange(size[1])*size[0]-1] = phi
    A.data[0, np.arange(size[1])*size[0]-1] = 1
    A.data[2, np.arange(size[1])*size[0]-2] = 0
    A.data[3, np.arange(size[1])*size[0]-1] = 0
    A.data[4, np.arange(size[1])*size[0]-1] = 0
    pass


def conj_grad_method(A, b, x):
    '''
    共轭梯度迭代
    '''
    r = b - sparse.dia_matrix.dot(A, x)
    d = r
    step = 0
    while step < size[0]*size[1]:
        Ad = sparse.dia_matrix.dot(A, d)
        alpha = np.dot(r.T, r) / np.dot(d.T, Ad)
        x = x + alpha*d
        beta = np.dot((r - alpha*Ad).T, r - alpha*Ad) / np.dot(r.T, r)
        r = r - alpha*Ad
        d = r + beta*d
        step += 1
        logError = np.log10(np.linalg.norm(r, ord=np.inf))  # 后向误差
        print('step=%d,log(r)=%6f' % (step, logError))
        if logError < 0:  # 误差上限
            break
    return x


######## 组装求解矩阵 ########
sizeA = size[0]*size[1]
A = coefficientMatrix()
b = np.reshape(-rho/epsilon0, (sizeA, 1))

######## 定义边界条件 ########
leftBoundary(0)
rightBoundary(0)
bottomBoundary(0)
topBoundary(0)

######## 解线性方程组 ########
phi = np.zeros((sizeA, 1))  # 初始预测解
phi = conj_grad_method(A, b, phi)
phi = np.reshape(phi, (size[0], size[1]))  # 重整回二维空间

######## 三点差分-计算梯度 ########
Ex = (phi[2:, 1:-1]-phi[:-2, 1:-1])/2/d[0]
Ey = (phi[1:-1, 2:]-phi[1:-1, :-2])/2/d[1]
Exy = -np.stack((Ex, Ey), axis=2)

######## 电场的可视化 ########
print('耗时%6fs' % (time.time()-t))


def plotElectricPotential():
    plt.pcolor(x, y, phi.T, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential distribution')
    pass


def plotEquipotentialSurface():
    plt.contour(x, y, phi.T, 10, colors='k',
                linestyles='dashed', linewidths=0.5)
    pass


def plotElectricFieldIntensity():
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x[1:-1], y[1:-1], E.T, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Field distribution')
    pass


def plotElectricFieldDirection():
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


plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.axis('equal')
plotElectricPotential()
plotEquipotentialSurface()
plt.subplot(1, 2, 2)
plt.axis('equal')
plotElectricFieldIntensity()
plotElectricFieldDirection()
plt.savefig('./q2e.pdf')
