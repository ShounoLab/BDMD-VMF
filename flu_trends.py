import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydmd import DMD

fname = "data/google_flu_trends_us.csv"
df = pd.read_csv(fname, index_col = 0)
df = df.iloc[:, 1:52]

plt.subplot(121)
plt.plot(df.iloc[300:, :])
plt.subplot(122)
plt.pcolor(df.iloc[300:, :])
plt.colorbar()
plt.show()

X = df.iloc[300:, :].values
dmd = DMD(svd_rank = 50)
dmd.fit(X.T)

for eig in dmd.eigs:
    print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag**2+eig.real**2 - 1)))

dmd.plot_eigs(show_axes=True, show_unit_circle=True)

xmax = np.vstack([X, dmd.reconstructed_data.T.real]).max()
xmin = np.vstack([X, dmd.reconstructed_data.T.real]).min()
fig = plt.figure(figsize=(17,12))
plt.subplot(231)
plt.pcolor(X, vmin = xmin, vmax = xmax)
plt.subplot(232)
plt.pcolor(dmd.reconstructed_data.T.real, vmin = xmin, vmax = xmax)
plt.subplot(233)
plt.pcolor(np.abs((X - dmd.reconstructed_data.T).real), vmin = xmin, vmax = xmax)
fig = plt.colorbar()
plt.subplot(234)
plt.plot(X)
plt.ylim((xmin, xmax))
plt.subplot(235)
plt.plot(dmd.reconstructed_data.T.real)
plt.ylim((xmin, xmax))
plt.subplot(236)
plt.plot((X - dmd.reconstructed_data.T).real)
plt.ylim((xmin, xmax))
plt.show()
