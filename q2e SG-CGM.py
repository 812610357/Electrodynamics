import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import pandas as pd
from scipy.sparse import linalg
import time

######## 定义求解区域 ########
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([1e-3, 1e-3])  # [y,x]
x = np.arange(xMin, xMax+d[1], d[1])
y = np.arange(yMin, yMax+d[0], d[0])
xx, yy = np.meshgrid(x, y)
size = np.array([len(y), len(x)], dtype='int')
error=np.array([])

######## 定义电荷分布 ########
epsilon0 = 1
rho = 2*np.pi**2*np.sin(np.pi*xx)*np.sin(np.pi*yy)

######## 五点差分-共轭梯度-求解拉普拉斯方程 ########
t = time.time()


def coefficientMatrix():
    '''
    根据五点差分法，使用稀疏矩阵，初始化系数矩阵，展平到一维空间
    '''
    #### 主对角线 ####
    C = np.ones(sizeA)*-2*(1/d[1]**2+1/d[0]**2)
    #### x方向差分 ####
    Dx = np.ones(sizeA-1)/d[1]**2
    Dx[np.arange(1, size[0], 1, dtype='int')*size[1]-1] = 0
    #### y方向差分 ####
    Dy = np.ones(sizeA-size[1])/d[0]**2
    A = sparse.diags([C, Dx, Dx, Dy, Dy], [0, 1, -1, size[1], -size[1]])
    return A


def leftBoundary(phi):
    '''
    x=xMin，左侧一类边值条件
    '''
    b[:size[1]] = phi
    A.data[0, :size[1]] = 1
    A.data[[1, 2], :size[1]] = 0
    A.data[3, :size[1]*2] = 0
    pass


def rightBoundary(phi):
    '''
    x=xMin，左侧一类边值条件
    '''
    b[-size[1]:] = phi
    A.data[0, -size[1]:] = 1
    A.data[[1, 2], -size[1]:] = 0
    A.data[4, -size[1]*2:] = 0
    pass


def bottomBoundary(phi):
    '''
    y=yMin，下侧一类边值条件
    '''
    b[np.arange(size[0])*size[1]] = phi
    A.data[0, np.arange(size[0])*size[1]] = 1
    A.data[1, np.arange(size[0])*size[1]+1] = 0
    A.data[3, np.arange(size[0])*size[1]] = 0
    A.data[4, np.arange(size[0])*size[1]] = 0
    pass


def topBoundary(phi):
    '''
    y=yMin，上侧一类边值条件
    '''
    b[np.arange(size[0])*size[1]-1] = phi
    A.data[0, np.arange(size[0])*size[1]-1] = 1
    A.data[2, np.arange(size[0])*size[1]-2] = 0
    A.data[3, np.arange(size[0])*size[1]-1] = 0
    A.data[4, np.arange(size[0])*size[1]-1] = 0
    pass


def conj_grad_method(A, b, x):
    '''
    预条件共轭梯度迭代
    '''
    global error
    Di=sparse.diags(np.power(A.data[0],-1),format='dia')
    U=sparse.triu(A,format='dia')
    #U.data[0]=0
    L=sparse.tril(A,format='dia')
    #L.data[0]=0
    M=sparse.dia_matrix.dot(sparse.dia_matrix.dot(L, Di),U)
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
        Error = np.linalg.norm(r, ord=np.inf)  # 后向误差
        print('step=%d,log(r)=%6f' % (step, np.log10(Error)))
        error=np.append(error,Error)
        if Error < 1e-10:  # 误差上限
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
print('耗时%6fs' % (time.time()-t))

######## 三点差分-计算梯度 ########
Ex = (phi[1:-1, 2:]-phi[1:-1, :-2])/2/d[1]
Ey = (phi[2:, 1:-1]-phi[:-2, 1:-1])/2/d[0]
Exy = -np.stack((Ey, Ex), axis=2)

######## 电场的可视化 ########


def plotElectricPotential():
    plt.pcolor(x, y, phi, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Potential distribution')
    pass


def plotEquipotentialSurface():
    plt.contour(x, y, phi, 10, colors='k',
                linestyles='dashed', linewidths=0.5)
    pass


def plotElectricFieldIntensity():
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x[1:-1], y[1:-1], E, cmap='jet')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Field distribution')
    pass


def plotElectricFieldDirection():
    xi = np.arange(0, 20, 1, dtype='int') * (size[1]-1)//20+(size[1]-1)//40
    yi = np.arange(0, 20, 1, dtype='int') * (size[0]-1)//20+(size[0]-1)//40
    xii, yii = np.meshgrid(xi, yi)
    xa = x[xi]
    ya = y[yi]
    E = np.linalg.norm(Exy[yii-1, xii-1, :], ord=2, axis=2)
    dx = Exy[yii-1, xii-1, 1]/E
    dy = Exy[yii-1, xii-1, 0]/E
    plt.quiver(xa, ya, dx, dy)
    pass

'''
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
'''
def writecsv(data):  # 导出线条
    dataframe = pd.DataFrame(data={'error': data})
    dataframe.to_csv("error-SG-CGM.csv",
                     index=False, mode='w', sep=',')
    pass

writecsv(error)
