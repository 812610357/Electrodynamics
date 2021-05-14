import numpy as np
import matplotlib.pyplot as plt

E = np.array([0., 0., 0.])  # 电场强度
B = np.array([0., 0., 1.])  # 磁感应强度
Q = 1  # 电荷量
m = 1  # 质量
r0 = np.array([-1., 0., 0.])  # 初始位移
v0 = np.array([0., 1., 0.])  # 初始速度
dt = 1e-3  # 时间精度
T = 20*np.pi  # 求解时长
tt = np.arange(0, T+dt, dt)

r = np.array([r0])
v = np.array([v0])
for t in tt[1:]:
    a = (Q*E+Q*np.cross(v, B))/m
    dr = v[-1]*dt+(a*dt**2)/2
    dv = a*dt
    r = np.row_stack((r, r[-1]+dr))
    v = np.row_stack((v, v[-1]+dv))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(r[:, 0], r[:, 1], r[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Path of particle')
plt.show()
