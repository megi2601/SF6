import scipy.signal as signal
from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd

# import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# #folia

# data = np.loadtxt("dane/chloroform.asc")
# X1 = data[:, 0]
# Y1 = data[:, 1]

# data2 = np.loadtxt("dane/D_chloroform.asc")

# X2 = data2[:, 0]
# Y2 = data2[:, 1]

# p1, _ = signal.find_peaks(-Y1, height=-30 )
# p2, _ = signal.find_peaks(-Y2, height=-30 )
# p1 = [p1[0]]
# p2 = [p2[0]]



# # fig = plt.gcf()
# # fig.clf()
# plt.figure(figsize=(10, 5))
# plt.plot(X1, Y1, label = "chloroform")
# plt.plot(X2, Y2, label = "D-chloroform")


# for peak in p1:
#     xs = -0.02*(plt.xlim()[1]-plt.xlim()[0])
#     ys = 0.05*(plt.ylim()[1]-plt.ylim()[0])
#     plt.annotate(f"{X1[peak]:.0f}", (X1[peak]-xs, Y1[peak]-ys))

# for peak in p2:
#     xs = -0.02*(plt.xlim()[1]-plt.xlim()[0])
#     ys = 0.05*(plt.ylim()[1]-plt.ylim()[0])
#     plt.annotate(f"{X2[peak]:.0f}", (X2[peak]-xs, Y2[peak]-ys))


# plt.grid()
# plt.gca().invert_xaxis()
# plt.xlabel("Liczba falowa [cm $ ^{-1} $]")
# plt.ylabel("Transmitancja [%]")
# plt.legend()
# plt.savefig("chloroform", dpi=400, bbox_inches='tight')
# plt.clf()

mh = mh=1.66*10**(-27)
mc = 12*mh

mi = mh*mc/(mh+mc)

print(3018*3018*4*np.pi*np.pi*9*10**20*mi)
print(8*np.pi*3018*9*10**20*mi)