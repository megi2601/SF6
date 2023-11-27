import scipy.signal as signal
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd


import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

filenames = ["dane\PLB_k1_chloroform.asc","dane\PLB_k2_chloroform.asc","dane\PLB_k28_chloroform.asc","dane\PLB_k5_chloroform.asc","dane\PLB_k02_chloroform.asc","dane\PLB_k04_chloroform.asc","dane\PLB_k06_chloroform.asc",]
filenames_k = [s[:-15]+".asc" for s in filenames]

znamionowe = [float("0."+s.split("_")[1][1:]) for s in filenames]


def periodic(x, T, a, b, c):
    return a*np.cos(2*np.pi/T*x+c) + b


absorbancja = []
absorbancja_zla=[]


for n, f in enumerate(filenames):
    data = np.loadtxt(f)
    X = np.array([int(i) for i in data[:, 0]])
    Y = data[:, 1]
    fig = plt.gcf()
    fig.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(X, Y)
    plt.grid()
    plt.gca().invert_xaxis()
    plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
    plt.ylabel("Transmitancja [%]")
    plt.savefig(f"chloroform{n}", dpi=400, bbox_inches='tight')
    absorbancja.append(-np.log(Y[X==929]/100))
    absorbancja_zla.append(-np.log(Y[X==3018]/100))



print(znamionowe)

init = [5/float("0."+s.split("_")[1][1:]) for s in filenames]

conds = [(1800, 2200), (2000, 2100), (2200, 2300), (2200, 2250), (1400, 3000), (1900, 2500), (1800, 2250)]
a = [5, 1.5, 0.5, 0.6, 5, 3, 3]
b= [83, 89.5, 77.5, 79, 73, 76.5, 82.5 ]
init[-3] = 1/2/0.003
dl_kuwet = []


plt.figure(figsize=(8, 5))
for n, f in enumerate(filenames_k):
    data = np.loadtxt(f)
    X = np.array([int(i) for i in data[:, 0]])
    Y = data[:, 1]
    Y=Y[(X>conds[n][0]) & (X<conds[n][1])]
    X=X[(X>conds[n][0]) & (X<conds[n][1])]
    pars, cov = curve_fit(periodic, X, Y, p0=[init[n], a[n], b[n], 0])
    #l [mm]
    l=1/2/pars[0]*10
    dl_kuwet.append(l)
    print()
    print(l)
    dl = 1/2/pars[0]/pars[0]*np.sqrt(cov[0, 0])*10
    print(dl)
    plt.clf()
    plt.plot(X, Y, label="Pomiar")
    plt.plot(X, periodic(X, *pars), linestyle='dashed', alpha=1, label="Dopasowanie")
    plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
    plt.ylabel("Transmitancja [%]")
    plt.legend()
    plt.grid()
    plt.gca().invert_xaxis()
    plt.savefig("dopasowanie"+str(znamionowe[n])+".png", dpi=200)


print(absorbancja)
print(absorbancja_zla)

plt.clf()
z = np.polyfit(dl_kuwet, absorbancja, 1)
x=np.linspace(min(dl_kuwet), max(dl_kuwet))
plt.plot(x, z[0]*x+z[1], linestyle="dashed", color='black', alpha=0.4, label="Dopasowana prosta")
plt.scatter(dl_kuwet, absorbancja, marker="o", color = "darkorange", label="Wyniki obliczeń")
plt.legend()
plt.grid()
plt.ylabel("Absorbancja")
plt.xlabel("Długość kuwety [mm]")
plt.savefig("absorbancja_dobra929", dpi=200)

plt.clf()
plt.scatter(dl_kuwet, absorbancja_zla, marker="o", color = "darkorange")
plt.grid()
plt.ylabel("Absorbancja")
plt.xlabel("Długość kuwety [mm]")
plt.savefig("absorbancja_zla3018", dpi=200)
