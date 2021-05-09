import numpy as np
import matplotlib.pyplot as plt

E = np.array([0., 0., 1.])
B = np.array([0., 0., 1.])
Q = 1
m = 1
r0 = np.array([0., 0., 0.])
v0 = np.array([0., 1., 0.])
dt = 1e-3
T = 20
tt = np.arange(0, T+dt, dt)


def F(v):
    FE = Q*E
    FB = Q*np.cross(v, B)
    return FE+FB


r = np.array([r0])
v = np.array([v0])
for t in tt:
    a = F(v[-1])/m
    dr = v[-1]*dt+(a*dt**2)/2
    r = np.row_stack((r, r[-1]+dr))
    dv = a*dt
    v = np.row_stack((v, v[-1]+dv))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(r[:, 0], r[:, 1], r[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
