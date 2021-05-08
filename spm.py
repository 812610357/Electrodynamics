import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def readField(path='./field.csv'):
    return np.array(pd.read_csv(path, index_col=0, header=0))


def readParticle(path='./particle.csv'):
    return np.array(pd.read_csv(path, index_col=False, header=0))


field = readField()
particle = readParticle()[0]
E = field[0, :]
B = field[1, :]
Q = particle[0]
m = particle[1]
r0 = particle[2:5]
v0 = particle[5:8]
dt = particle[9]
T=particle[10]
tt=np.arange(0,T+dt,dt)

def F(v):
    FE = Q*E
    FB = Q*np.cross(v, B)
    return FE+FB

r=np.array([r0])
v=np.array([v0])
for t in tt:
    dr=v[-1]*dt+(F(v[-1])/m*dt**2)/2
    r=np.row_stack(r,r[-1]+dr)
    dv=F(v[-1])/m*dt
    v=np.row_stack(v,v[-1]+dv)

plt

