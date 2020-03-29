# Python 2 required
# pip install git+https://github.com/aaren/sparse_dmd

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from pydmd import DMD
import sparse_dmd
from sklearn import preprocessing

#fname = "data/Data/HuGaDB_v1_walking_14_04.txt"
fname = "data/Data/HuGaDB_v1_bicycling_01_01.txt"
df = pd.read_csv(fname, index_col = 1, skiprows = 1, skipfooter = 3)

# for bicycle
datarange = range(101, 401)

plt.subplot(221)
plt.plot(df.loc[datarange, df.columns.str.startswith("acc_l")])
plt.title("Acc_left")
plt.subplot(222)
plt.plot(df.loc[datarange, df.columns.str.startswith("acc_r")])
plt.title("Acc_right")
plt.subplot(223)
plt.plot(df.loc[datarange, df.columns.str.startswith("gyro_l")])
plt.title("Gyro_left")
plt.subplot(224)
plt.plot(df.loc[datarange, df.columns.str.startswith("gyro_r")])
plt.title("Gyro_right")
plt.show()

#X = df.loc[datarange, :].values
X = df.loc[datarange, df.columns.str.startswith("acc")].values

dmd = sparse_dmd.DMD(X.T, axis = -1)
dmd.compute()
# eigvals
plt.scatter(dmd.ritz_values.real, dmd.ritz_values.imag)
plt.show()
dmd.amplitudes

spdmd = sparse_dmd.SparseDMD(dmd = dmd)
gamma = np.logspace(-2, 10, 300)
spdmd.compute_sparse(gamma)
opt_amps = spdmd.sparse.xpol
spdmd.sparse.Nz

plotter = sparse_dmd.SparsePlots(spdmd)
fig = plotter.performance_loss_gamma()
fig, ax = plt.subplots()
plotter.nonzero_gamma(ax)
plt.show()


reconstruction = spdmd.reconstruction(Ni=3)
plt.plot(reconstruction.rdata)
plt.show()



#X = df.loc[datarange, df.columns.str.startswith("g")].values
#dmd = DMD(svd_rank = 2)
#dmd.fit(X.T)
#
#   for eig in dmd.eigs:
#       print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))
#
#   dmd.plot_eigs(show_axes=True, show_unit_circle=True)
#
#   xmax = np.vstack([X, dmd.reconstructed_data.T.real]).max()
#   xmin = np.vstack([X, dmd.reconstructed_data.T.real]).min()
#   fig = plt.figure(figsize=(17,12))
#   plt.subplot(231)
#   plt.pcolor(X, vmin = xmin, vmax = xmax)
#   plt.subplot(232)
#   plt.pcolor(dmd.reconstructed_data.T.real, vmin = xmin, vmax = xmax)
#   plt.subplot(233)
#   plt.pcolor(np.abs((X - dmd.reconstructed_data.T).real), vmin = xmin, vmax = xmax)
#   fig = plt.colorbar()
#   plt.subplot(234)
#   plt.plot(X)
#   plt.ylim((xmin, xmax))
#   plt.subplot(235)
#   plt.plot(dmd.reconstructed_data.T.real)
#   plt.ylim((xmin, xmax))
#   plt.subplot(236)
#   plt.plot((X - dmd.reconstructed_data.T).real)
#   plt.ylim((xmin, xmax))
#   plt.show()
