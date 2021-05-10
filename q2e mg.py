import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import time

######## 定义求解区域 ########
postIterError = 1
mainIterTimes = 4
preIterTimes = 10
xMin = -1
xMax = 1
yMin = -1
yMax = 1
d = np.array([1e-3, 1e-3])  # [y,x]
x = np.arange(xMin, xMax+d[1], d[1])
y = np.arange(yMin, yMax+d[0], d[0])
xx, yy = np.meshgrid(x, y)
size = np.array([len(y), len(x)], dtype='int')

######## 定义电荷分布 ########
epsilon0 = 8.85418e-12
rho = 2*np.pi**2*np.sin(np.pi*xx)*np.sin(np.pi*yy)

######## 多重网格-五点差分-共轭梯度-求解拉普拉斯方程 ########
t = time.time()


def smoothing(phi, f, d,pre=False):
    '''
    平滑算子，共轭梯度迭代，返回迭代结果和剩余误差
    '''
    def coefficientMatrix():
        '''
        展平到一维空间，根据五点差分法，使用稀疏矩阵，初始化系数矩阵
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

    def conjGradMethod(A, b, x):
        '''
        共轭梯度迭代求解
        '''
        r = b - sparse.dia_matrix.dot(A, x)
        d = r
        CGMstep = 0
        while CGMstep < size[0]*size[1]:
            Ad = sparse.dia_matrix.dot(A, d)
            alpha = np.dot(r.T, r) / np.dot(d.T, Ad)
            x = x + alpha*d
            beta = np.dot((r - alpha*Ad).T, r - alpha*Ad) / np.dot(r.T, r)
            r = r - alpha*Ad
            d = r + beta*d
            CGMstep += 1
            Error = np.linalg.norm(r, ord=np.inf)  # 后向误差
            print('VMGstep=%d,CGMstep=%d,log(r)=%6f' %
                  (VMGstep, CGMstep, np.log10(Error)))
            if pre and CGMstep>preIterTimes: # 预迭代次数上限
                break     
            if Error < postIterError:  # 误差上限
                break
        return x

    ######## 组装求解矩阵 ########
    sizeA = size[0]*size[1]
    A = coefficientMatrix()
    b = np.reshape(f, (sizeA, 1))

    ######## 定义边界条件 ########
    leftBoundary(0)
    rightBoundary(0)
    bottomBoundary(0)
    topBoundary(0)

    ######## 解线性方程组 ########
    phi = np.reshape(phi, (sizeA, 1))  # 展平到一维空间
    phi = conjGradMethod(A, b, phi)

    ######## 计算剩余误差 ########
    res = b-sparse.dia_matrix.dot(A, phi)  # 计算剩余误差
    phi = np.reshape(phi, (size[0], size[1]))  # 重整回二维空间
    res = np.reshape(res, (size[0], size[1]))

    return phi, res


def restriction(res):
    '''
    限制算子，将网格间距为h的剩余误差投影到2h
    '''
    xi = np.arange(0, res.shape[1]+1, 2, dtype='int')
    yi = np.arange(0, res.shape[0]+1, 2, dtype='int')
    xic = xi[1:-1]
    yic = yi[1:-1]
    xii, yii = np.meshgrid(xi, yi)
    xiic = xii[1:-1, 1:-1]
    yiic = yii[1:-1, 1:-1]
    rhs = np.zeros(((res.shape[0]-1)//2+1, (res.shape[1]-1)//2+1))
    rhs[1:-1, 1:-1] = (4*res[xiic, yiic]+2*(res[xiic-1, yiic]+res[xiic+1, yiic]+res[xiic, yiic-1]+res[xiic, yiic+1]) +
                       res[xiic-1, yiic-1]+res[xiic-1, yiic+1] + res[xiic+1, yiic+1]+res[xiic+1, yiic-1])/16  # 非边界
    rhs[1:-1, 0] = (2*res[yic, 0]+res[yic-1, 0]+res[yic+1, 0])/4  # 左边界
    rhs[1:-1, -1] = (2*res[yic, -1]+res[yic-1, -1]+res[yic+1, -1])/4  # 右边界
    rhs[0, 1:-1] = (2*res[0, xic]+res[0, xic-1]+res[0, xic+1])/4  # 下边界
    rhs[-1, 1:-1] = (2*res[-1, xic]+res[-1, xic-1]+res[-1, xic+1])/4  # 上边界
    rhs[[0, 0, -1, -1], [0, -1, 0, -1]] = res[[0, 0, -1, -1], [0, -1, 0, -1]]  # 四个角
    return rhs


def VCycleMultiGrid(phi, f, d):
    '''
    V循环多重网格法主程序
    '''
    phi, res = smoothing(phi, f, d)  # 预平滑
    rhs = restriction(res)
    eps = np.zeros_like(rhs)
    if VMGstep < mainIterTimes:
        VCycleMultiGrid(eps, rhs, 2*d)
    else:
        eps = smoothing(eps, rhs, 2*d)[0] # 到达最大主迭代次数

    phi = smoothing(phi, f, d)
    return phi


VMGstep = 1
phi = np.zeros((size[0], size[1]))  # 等号左侧自变量的预测
f = -rho/epsilon0                   # 等号右侧的函数表达式
phi = VCycleMultiGrid(phi, f, d)

######## 三点差分-计算梯度 ########
Ex = (phi[1:-1, 2:]-phi[1:-1, :-2])/2/d[1]
Ey = (phi[2:, 1:-1]-phi[:-2, 1:-1])/2/d[0]
Exy = -np.stack((Ey, Ex), axis=2)

######## 电场的可视化 ########
print('耗时%6fs' % (time.time()-t))


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
