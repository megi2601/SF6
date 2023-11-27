import scipy.signal as signal
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd




import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def line(x, a, b):
    return a*x+b

#HCl

data = np.loadtxt("dane\HCl.asc")
X = np.array([int(i) for i in data[:, 0]])
Y = data[:, 1]

# fig = plt.gcf()
# fig.clf()
plt.figure(figsize=(8, 5))
plt.plot(X, Y)
plt.grid()
plt.gca().invert_xaxis()
p, _ = signal.find_peaks(-Y, height=-102 )
#print(X[p])
for peak in p:
    xs = 0.02*(plt.xlim()[1]-plt.xlim()[0])
    ys = 0.04*(plt.ylim()[1]-plt.ylim()[0])
    plt.annotate(f"{X[peak]:.0f}", (X[peak]-xs, Y[peak]-ys))
plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
plt.ylabel("Transmitancja [%]")
plt.savefig("HCl", dpi=300, bbox_inches='tight')
plt.clf()

xp = np.delete(X[p], 4) #usunięcie jedynego podwójnego

#print(xp)
R = xp[:10]
P = xp[10:]
R = sorted(R)

df = pd.DataFrame(columns=["r", "p", "deltab1", "deltab0"])

deltab1 = R[1:] - P
deltab0 = R[:-2] - P[1:]

df["p"] = np.hstack((pd.NA, P))
df["r"] = np.hstack((R  ))
df["deltab1"] = np.hstack((pd.NA, deltab1))
df["deltab0"] = np.hstack(( [pd.NA, pd.NA], deltab0,))

#print(df.to_latex(na_rep="-"))

par, cov = curve_fit(line, range(1, 10), list(df["deltab1"][1:]), sigma=np.ones(9))
a1 = par[0]
d_a1 = np.sqrt(cov[0, 0])
par, cov = curve_fit(line, range(2, 10), list(df["deltab0"][2:]), sigma=np.ones(8))
a0 = par[0]
d_a0 = np.sqrt(cov[0, 0])

#stałe
h = 6.626*10**(-34)
mh=1.66*10**(-27)
mcl = 35*mh
c = 3*10**10

def sqrt_hcl():
    return np.sqrt(h*(mh+mcl)/3.14/3.14/mh/mcl/c)

print("R HCl:")
print(sqrt_hcl()/np.sqrt((3*a0-a1)))
print(np.sqrt((d_a0*sqrt_hcl()*1.5/(3*a0-a1)**1.5)**2+(d_a1*sqrt_hcl()/2/(3*a0-a1)**1.5)**2))


#CH4

data = np.loadtxt("dane\metan.asc")
X = np.array([int(i) for i in data[:, 0]])
Y = data[:, 1]

# fig = plt.gcf()
# fig.clf()
plt.figure(figsize=(9, 5))
plt.plot(X, Y)
plt.grid()
plt.gca().invert_xaxis()
p, _ = signal.find_peaks(-Y, height=-95, prominence=2)
#print(X[p])
for peak in p:
    xs = 0.01*(plt.xlim()[1]-plt.xlim()[0])
    ys = 0.03*(plt.ylim()[1]-plt.ylim()[0])
    if X[peak] == 3067:
        ys*=1.6
        xs/=1.1
    plt.annotate(f"{X[peak]:.0f}", (X[peak]-xs, Y[peak]-ys), size=7)
plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
plt.ylabel("Transmitancja [%]")
plt.savefig("metan", dpi=300, bbox_inches='tight')
plt.clf()

xp = X[p]
#print(xp)
R = xp[:15]
P = xp[16:] #13
R = sorted(R)


df = pd.DataFrame(columns=["r", "p", "deltab1", "deltab0"])
deltab1 = R[2:] - P
deltab0 = R[:-2] - P

df["p"] = np.hstack((pd.NA, pd.NA, P))  # z analizy kształtu - najbardziej obsadzony poziom wniosek że jeden pik  z gałęzi P jest wewnątrz Q
df["r"] = np.hstack((R))

df["deltab1"] = np.hstack(([pd.NA, pd.NA], deltab1))
df["deltab0"] = np.hstack(( deltab0, [pd.NA, pd.NA]))

#print(df.to_latex(na_rep="-"))


par, cov = curve_fit(line, range(2, 15), list(df["deltab1"][2:]), sigma=np.ones(13))

a1 = par[0]
d_a1 = np.sqrt(cov[0, 0])

par, cov = curve_fit(line, range(13), list(df["deltab0"][:-2]), sigma=np.ones(13))


a0 = par[0]
d_a0 = np.sqrt(cov[0, 0])



def sqrt_met():
    return np.sqrt(3*h/8/np.pi/np.pi/mh/c)

print("R metan:")
print(sqrt_met()/np.sqrt((3*a0-a1)))
print(np.sqrt((d_a0*sqrt_met()*1.5/(3*a0-a1)**1.5)**2+(d_a1*sqrt_met()/2/(3*a0-a1)**1.5)**2))


## druga wersja

xp = X[p]
#print(xp)
R = xp[:15]
P = xp[16:] #13
R = sorted(R)


df = pd.DataFrame(columns=["r", "p", "deltab1", "deltab0"])
deltab1 = R[:-2] - P
deltab0 = R[:-4] - P[2:]

df["p"] = np.hstack((pd.NA, P,  [pd.NA, pd.NA]))  # z analizy kształtu - najbardziej obsadzony poziom wniosek że jeden pik  z gałęzi P jest wewnątrz Q
df["r"] = np.hstack((pd.NA, R))

df["deltab1"] = np.hstack(([pd.NA], deltab1, [pd.NA, pd.NA]))
df["deltab0"] = np.hstack(( [pd.NA], deltab0, [pd.NA, pd.NA, pd.NA,  pd.NA]))

#print(df.to_latex(na_rep="-"))


par, cov = curve_fit(line, range(1, 14), list(df["deltab1"][1:-2]), sigma=np.ones(13))

a1 = par[0]
d_a1 = np.sqrt(cov[0, 0])

par, cov = curve_fit(line, range(1, 12), list(df["deltab0"][1:-4]), sigma=np.ones(11))


a0 = par[0]
d_a0 = np.sqrt(cov[0, 0])



def sqrt_met():
    return np.sqrt(3*h/8/np.pi/np.pi/mh/c)

print("R metan:")
print(sqrt_met()/np.sqrt((3*a0-a1)))
print(np.sqrt((d_a0*sqrt_met()*1.5/(3*a0-a1)**1.5)**2+(d_a1*sqrt_met()/2/(3*a0-a1)**1.5)**2))