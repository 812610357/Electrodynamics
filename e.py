import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(np.pi/6+0.05,np.pi/2,1000)
y=np.sin(x)
plt.plot(x,y)
plt.plot([0.9,0.9,1.1,1.1,0.9,1.1,1.1],[0.5,np.sin(0.9),np.sin(1.1),np.sin(0.9),np.sin(0.9),np.sin(0.9),0.5])

plt.annotate('',(0.9,0.55),xytext=(1.1,0.55),arrowprops={'arrowstyle':'<->'})
plt.text(0.96,0.56,'d$x$',fontsize=20)

plt.annotate('',(1.12,0.5),xytext=(1.18,0.5),arrowprops={'arrowstyle':'-'})
plt.annotate('',(1.12,np.sin(0.9)),xytext=(1.18,np.sin(0.9)),arrowprops={'arrowstyle':'-'})
plt.annotate('',(1.12,np.sin(1.1)),xytext=(1.18,np.sin(1.1)),arrowprops={'arrowstyle':'-'})
plt.annotate('',(1.15,0.5),xytext=(1.15,np.sin(0.9)),arrowprops={'arrowstyle':'<->'})
plt.annotate('',(1.15,np.sin(0.9)),xytext=(1.15,np.sin(1.1)),arrowprops={'arrowstyle':'<->'})
plt.text(1.18,0.63,'$f(x_i)$',fontsize=20)
plt.text(1.18,0.83,'$f\'(x_i)$',fontsize=20)

plt.text(0.87,0.46,'$x_i$',fontsize=20)
plt.text(1.04,0.46,'$x_{i+1}$',fontsize=20)
plt.text(1.7,0.46,'$x$',fontsize=16)

plt.plot([0.4,1.7,1.65,1.7,1.65],[0.5,0.5,0.52,0.5,0.48],'k')
plt.axis('off')
plt.show()