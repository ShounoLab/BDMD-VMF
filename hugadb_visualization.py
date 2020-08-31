import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df_original = pd.read_csv("output/hugadb_original.csv", header = None)
df_missing = pd.read_csv("output/hugadb_missing.csv", na_values = "missing", header = None)


fname = "data/Data/HuGaDB_v1_bicycling_01_01.txt"
df = pd.read_csv(fname, header = 3, delimiter = '\t')
part_names = df.columns[df.columns.str.startswith("gyro_")]
part_names = list(map(lambda x: x.replace('gyro_', ''), part_names))


f, (ax1, ax2, axcb) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.08]}, figsize = (18, 8))
ax1.get_shared_y_axes().join(ax2)
g1 = sns.heatmap(df_original, cmap = plt.cm.inferno, cbar_ax = axcb, ax = ax1, rasterized = True)
ax1.set_yticks(list(map(lambda x: x + 0.5, range(19))))
ax1.set_yticklabels(part_names, rotation = 0, fontsize = 30)
ax1.set_xticks([0, 50, 100, 150])
ax1.set_xticklabels([0, 50, 100, 150], fontsize = 21, rotation = 0)
ax1.invert_yaxis()
ax1.set_title("(a) Original", y = -0.2, fontsize = 48, **{"fontname": "Times New Roman"})
g2 = sns.heatmap(df_missing, cmap = plt.cm.inferno, cbar = False, ax = ax2, rasterized = True)
g2.set_yticks([])
ax2.set_xticks([0, 50, 100, 150])
ax2.set_xticklabels([0, 50, 100, 150], fontsize = 21, rotation = 0)
ax2.invert_yaxis()
ax2.set_title("(b) Masked", y = -0.2, fontsize = 48, **{"fontname": "Times New Roman"})
ax1.set_ylim([0, 18])
ax2.set_ylim([0, 18])
f.tight_layout()
ax1.spines["bottom"].set_visible(True)
ax2.spines["bottom"].set_visible(True)
ax1.spines["left"].set_visible(True)
ax2.spines["left"].set_visible(True)
axcb.tick_params(labelsize = 18, rotation = 60)
f.savefig("output/hugadb_data.pdf")
