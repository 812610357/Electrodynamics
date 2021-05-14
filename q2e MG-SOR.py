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

mainIterTimes=3

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
t=time.time()
def smoothing(phi,f,d,pre=False):
    global error
    size=np.array([phi.shape[0],phi.shape[1]])
    phi = np.zeros((size[0], size[1]))
    t = time.time()
    w = 1.5
    Error = np.Inf
    step = 0
    while Error > 1e-1:
        def sor(phi, f, w):
            b = d[0]/d[1]
            B2 = b*b
            DX2 = d[0]*2
            C = w/(2*(1+B2))
            for i in range(1, size[0]-1):
                for j in range(1, size[1]-1):
                    phi[i, j] = (1-w)*phi[i, j]+C*(phi[i+1, j]+phi[i-1, j] + B2*(phi[i, j+1] + phi[i, j-1])+DX2*f[i, j])
            return phi
        phi1 = sor(np.copy(phi), f ,w)
        Error = np.max(phi1-phi)
        error=np.append(error,Error)
        phi = np.copy(phi1)
        step += 1
        print('step=%d,error=%e' % (step,Error))
        res = np.zeros((size[0], size[1]))
        if pre and step>=3:
            for i in range(1, size[0]-1):
                for j in range(1, size[1]-1):
                    res[i,j]=f[i,j]-((phi[i-1,j]+phi[i+1,j])/d[0]/d[0]+(phi[i,j-1]+phi[i,j+1])/d[1]/d[1]-2*phi[i,j]*(1/d[0]/d[0]+1/d[1]/d[1]))
            break
    return phi,res

def restriction(res):
    '''
    限制算子，将网格间距为h的剩余误差投影到间距为2h的网格
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


def prolongation(eps):
    '''
    延拓算子，将网格间距为2h的低精度解插值到间距为h的网格
    '''
    xi = np.arange(0, eps.shape[1], 1, dtype='int')
    yi = np.arange(0, eps.shape[0], 1, dtype='int')
    xic = xi[:-1]
    yic = yi[:-1]
    xii, yii = np.meshgrid(xi, yi)
    xiic = xii[:-1, :-1]
    yiic = yii[:-1, :-1]
    phi = np.zeros(((eps.shape[0]-1)*2+1, (eps.shape[1]-1)*2+1))
    phi[xii*2, yii*2] = eps[xii, yii]  # 非插值点
    phi[xiic*2+1, yiic*2+1] = (eps[xiic, yiic]+eps[xiic+1, yiic] +
                               eps[xiic+1, yiic+1]+eps[xiic+1, yiic+1])/4  # 中心点
    phi[xiic*2+1, yiic*2] = (eps[xiic, yiic]+eps[xiic+1, yiic])/2  # 左右点
    phi[yic*2+1, -1] = (eps[yic, -1]+eps[yic+1, -1])/2
    phi[xiic*2, yiic*2+1] = (eps[xiic, yiic]+eps[xiic, yiic+1])/2  # 上下点
    phi[-1, xic*2+1] = (eps[-1, xic]+eps[-1, xic+1])/2
    return phi


def VCycleMultiGrid(phi, f, d):
    '''
    V循环多重网格法主程序
    '''
    global VMGstep
    phi, res = smoothing(phi, f, d, pre=True)  # 预平滑
    rhs = restriction(res)
    eps = np.zeros_like(rhs)
    VMGstep += 1
    if VMGstep < mainIterTimes:
        VCycleMultiGrid(eps, rhs, 2*d)
    else:
        eps = smoothing(eps, rhs, 2*d, pre=True)[0]  # 到达最大主迭代次数
    VMGstep -= 1
    phi += prolongation(eps)
    phi = smoothing(phi, f, d)[0]  # 后平滑
    return phi


VMGstep = 1
phi = np.zeros((len(y), len(x)))  # 等号左侧自变量的预测
f = -rho/epsilon0                 # 等号右侧的函数表达式
phi = VCycleMultiGrid(phi, f, d)

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

'''
plt.axis('equal')
plotElectricFieldIntensity(Exy)
plotElectricFieldDirection(Exy)
plt.show()
'''

def writecsv(data):  # 导出线条
    dataframe = pd.DataFrame(data={'error': data})
    dataframe.to_csv("error-MG-SOR.csv",
                     index=False, mode='w', sep=',')
    pass

writecsv(error)