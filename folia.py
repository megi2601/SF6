import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

#folia

data = np.loadtxt("dane/folia.asc")
X = data[:, 0]
Y = data[:, 1]

fig = plt.gcf()
fig.clf()
plt.figure(figsize=(10, 5))
plt.plot(X, Y)
plt.grid()
plt.gca().invert_xaxis()

p, _ = signal.find_peaks(-Y, prominence=20, height=-70)
p = p[1:]
print(X[p])
for peak in p:
    plt.plot(X[peak],  Y[peak], '*', color="red")
x = Y[X==1328]
print(x)
for peak in [(1328, x)]:                        #ocena ręczna
    plt.plot(peak[0],  peak[1], '*', color="red")
plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
plt.ylabel("Transmitancja [%]")
plt.savefig("folia", dpi=300, bbox_inches='tight')
plt.clf()

p_exp = [int(i) for i in np.hstack((X[p], [1328, 1282]))]
p_exp = sorted(p_exp, reverse=True)
p_theoretical = [3027, 2924, 2850, 1944, 1871, 1801, 1601, 1583, 1495, 1454, 1353, 1332, 1282, 1181, 1154, 1069, 1028, 906, 842, 752, 698]

df = pd.DataFrame({"exp":p_exp, "th":p_theoretical})
df["exp"] = df["exp"].abs()
df["delta"] = (df["exp"] - df["th"]).abs()


#print(df.to_latex(index=False))

