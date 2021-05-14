import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data1 = pd.read_csv(f"error-MG-CGM.csv", index_col=False, header=0)
data2 = pd.read_csv(f"error-SG-CGM.csv", index_col=False, header=0)

plt.figure(figsize=(10, 4))
plt.axes(yscale='log')
plt.plot(data1,label='MG-CGM')
plt.plot(data2,label='SG-CGM')
plt.title('residual errors')
plt.legend()
plt.savefig('error.pdf')
