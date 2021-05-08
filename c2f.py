import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

domain = np.array([-4, 4, -4, 4])  # [xmin, xmax, ymin, ymax]
size = 1001
epsilon0 = 8.85418e-12
x = np.linspace(domain[0], domain[1], size)
y = np.linspace(domain[2], domain[3], size)
xx = np.pad(np.array([x]), ((0, size-1), (0, 0)), 'edge')
yy = np.pad(np.array([y]), ((0, size-1), (0, 0)), 'edge').T

asize = (size-1)//50+1
ax = np.linspace(domain[0], domain[1], asize)[1:-1]
ay = np.linspace(domain[2], domain[3], asize)[1:-1]
axx = np.pad(np.array([ax]), ((0, asize-3), (0, 0)), 'edge')
ayy = np.pad(np.array([ay]), ((0, asize-3), (0, 0)), 'edge').T


def readCharges(path='./charges.csv'):
    return np.array(pd.read_csv(path, index_col=False, header=0))


def pointChargeElectricField(x, y, charge, axis=0):
    x0 = charge[0]
    y0 = charge[1]
    Q = charge[2]
    r = np.sqrt(np.power(x-x0, 2)+np.power(y-y0, 2))
    Ex = Q*(x-x0)/(4*np.pi*epsilon0*np.power(r, 3))
    Ey = Q*(y-y0)/(4*np.pi*epsilon0*np.power(r, 3))
    return np.stack((Ex, Ey), axis=axis)


def plotElectricFieldIntensity(charges):
    Exy = np.empty((size, size, 2))
    for charge in charges:
        Exy += pointChargeElectricField(xx, yy, charge, axis=2)
    E = np.linalg.norm(Exy, ord=2, axis=2)
    plt.pcolor(x, y, E, norm=LogNorm(vmin=1e7, vmax=1e14), cmap='jet')
    pass


def plotElectricFieldDirection(charges):
    Exy = np.empty((asize-2, asize-2, 2))
    for charge in charges:
        Exy += pointChargeElectricField(axx, ayy, charge, axis=2)
    E = np.linalg.norm(Exy, ord=2, axis=2)
    dx = Exy[:, :, 0]/E
    dy = Exy[:, :, 1]/E
    plt.quiver(ax, ay, dx, dy)
    pass


def plotElectricFieldLine(charges):
    for charge0 in charges:
        Q = abs(charge0[2])
        for i in range(1, round(32*Q), 2):
            p = np.array([[charge0[0]+3e-3*np.cos(i*np.pi/round(16*Q)),
                           charge0[1]+3e-3*np.sin(i*np.pi/round(16*Q))]])
            judge = True
            while judge:
                Exy = np.empty(2)
                for charge in charges:
                    Exy += pointChargeElectricField(p[-1, 0], p[-1, 1], charge)
                E = np.linalg.norm(Exy, ord=2)
                dp = 1e-3*Exy/E
                p = np.row_stack((p, p[-1]+np.sign(charge0[2])*dp))
                for charge in charges:
                    if np.linalg.norm(p[-1]-charge[0:2], ord=2) < 2e-3 or not(domain[0] < p[-1, 0] < domain[1] and domain[2] < p[-1, 1] < domain[3]):
                        plt.plot(p[:-1, 0], p[:-1, 1], '-k', linewidth=0.5)
                        judge = False
                        break
    pass


def plotEquipotentialSurface(charges):
    phi = 0
    for charge in charges:
        x0 = charge[0]
        y0 = charge[1]
        Q = charge[2]
        r = np.sqrt(np.power(xx-x0, 2)+np.power(yy-y0, 2))
        phi += Q/(4*np.pi*epsilon0*r)
    plt.contour(x, y, np.log10(np.abs(phi)), 10, colors='k',
                linestyles='dashed', linewidths=0.5)
    pass


charges = readCharges()
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 2)
plt.axis('equal')
plotElectricFieldIntensity(charges)
plotElectricFieldDirection(charges)
plt.subplot(1, 2, 1)
plt.axis('equal')
plotElectricFieldLine(charges)
plotEquipotentialSurface(charges)
plt.savefig('./2.pdf')
plt.show()
