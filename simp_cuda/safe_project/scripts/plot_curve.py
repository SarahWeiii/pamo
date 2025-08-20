import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import seaborn as sns

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ' \\usepackage{libertine} '
mpl.rcParams['ps.usedistiller'] = 'xpdf'
mpl.rcParams['font.size'] = 14

root_dir = "output/stage3_0515_curve"

cd_path = os.path.join(root_dir, "cd_curve.csv")
hd_path = os.path.join(root_dir, "hd_curve.csv")

# load csv, ignore first row
cd_df = pd.read_csv(cd_path, skiprows=1, header=None).to_numpy()[:, 2:]
hd_df = pd.read_csv(hd_path, skiprows=1, header=None).to_numpy()[:, 2:]

x_axis = np.arange(0, cd_df.shape[1], 1)

cd_mean = np.mean(cd_df, axis=0)
cd_std = np.std(cd_df, axis=0)
hd_mean = np.mean(hd_df, axis=0)
hd_std = np.std(hd_df, axis=0)

fig = plt.figure(figsize=(4.8, 2.5))
# draw CD and HD on the same plot but with different y-axis
# ax1 = fig.add_subplot(111)
# ax1.plot(x_axis, cd_mean, label="CD", color="blue")
# # ax1.fill_between(x_axis, cd_mean - cd_std, cd_mean + cd_std, color="blue", alpha=0.2)
# ax1.set_ylabel("CD")
# ax1.set_xlabel("Iteration")
# ax1.legend(loc="upper left")

# ax2 = ax1.twinx()
# ax2.plot(x_axis, hd_mean, label="HD", color="green")
# # ax2.fill_between(x_axis, hd_mean - hd_std, hd_mean + hd_std, color="green", alpha=0.2)
# ax2.set_ylabel("HD")
# ax2.legend(loc="upper right")

# color1 = "tab:blue"
# color2 = "tab:orange"

# color1 = (248, 113, 103)
# color2 = (143, 209, 236)

color2 = "#7FC1EC"
color1 = "#F87167"

# color1 = "#4A81AE"
# color2 = "#D78F4F"

ax1 = fig.add_subplot(111)
ax1.set_ylabel("CD ($10^{-5}$)")
ax1.set_xlabel("Iteration $t$")
ax1.set_axisbelow(True)

ax1.tick_params(axis='x', which='both', length=0)

sns.lineplot(x=x_axis, y=cd_mean * 1e5, label="CD", color=color1)

# set ax1 ticks
ax1.set_xticks(np.arange(0, 51, 10))

ax2 = plt.twinx()
ax2.set_ylabel("HD ($10^{-2}$)")
ax2.set_axisbelow(True)

sns.lineplot(x=x_axis, y=hd_mean * 1e2, label="HD", linestyle="--", color=color2)

# ax1.grid(color='white', alpha=1, axis='y')
ax1.grid(color='white', alpha=1, axis='x')
# ax2.grid(color='white', alpha=1, axis='y', linestyle="--")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
ax1.legend().remove()

# ax1.grid(color='tab:blue', alpha=0.2, axis='y')
# ax2.grid(color='tab:orange', alpha=0.2, axis='y')
# ax1.grid(color='tab:gray', alpha=0.2, axis='x')

ax1.set_facecolor((0.92, 0.92, 0.92))
sns.despine(left=True, bottom=True, right=True)

fig.tight_layout()
fig.savefig(os.path.join(root_dir, "stage3_convergence.png"), dpi=300)
plt.show()