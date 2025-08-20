import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse
import seaborn as sns
import re


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = ' \\usepackage{libertine} '
mpl.rcParams['ps.usedistiller'] = 'xpdf'
mpl.rcParams['font.size'] = 14

color2 = "#7FC1EC"
color1 = "#F87167"

root_dir = "output/undo"
s2_runtime_path = "data/runtime.txt"
s3_root_dir = "output/full_stage3_0516_0517_todo"
s3_eval_path = os.path.join(s3_root_dir, "stage3_eval.csv")
s2_s3_data_path = os.path.join(s3_root_dir, "stage3_eval_with_s2_time.csv")
load_s2_s3_data = False

if not load_s2_s3_data:
    s2_re = re.compile("(\d+)\.[^\d].*Time : ([\d\.]+)")
    s2_id_time = []
    with open(s2_runtime_path) as f:
        s2_lines = f.readlines()
        for l in s2_lines:
            m = s2_re.search(l)
            if m:
                id = m.group(1)
                time = float(m.group(2))
                s2_id_time.append((id, time))
    s2_id_time = sorted(s2_id_time, key=lambda x: x[0])
    s2_id, s2_time = zip(*s2_id_time)

    s3_data = pd.read_csv(s3_eval_path)
    # drop the first column 
    s3_data = s3_data.drop(s3_data.columns[0], axis=1)
    # drop the last 2 rows
    s3_data = s3_data.drop(s3_data.index[-2:])
    # print(s3_data)
    # add one column (stage 2 time) and match according to ID
    # convert ID to str
    s3_data["ID"] = s3_data["ID"].astype(str)
    s3_data["Stage 2 Time"] = np.nan
    for id, time in s2_id_time:
        s3_data.loc[s3_data["ID"] == id, "Stage 2 Time"] = time
        
    # Add a column for sum of s2 and s3 time
    s3_data["Total Time"] = s3_data["Time(s)"] + s3_data["Stage 2 Time"]

    # save the data
    s3_data.to_csv(os.path.join(s3_root_dir, "stage3_eval_with_s2_time.csv"))
else:
    s3_data = pd.read_csv(s2_s3_data_path)
    # print(s3_data)

# Plot a histogram of the s2 time 
fig = plt.figure(figsize=(8, 3.5))
ax1 = fig.add_subplot(121)

ax1.grid(color='white', alpha=1, axis='y', linestyle="-", linewidth=1.5)
ax1.set_facecolor((0.94, 0.94, 0.94))
ax1.tick_params(axis='x', which='both', length=0)
ax1.set_axisbelow(True)

ax1.set_ylabel("Frequency")
ax1.set_xlabel("Time (s)")

s2_time = s3_data["Stage 2 Time"].to_numpy()
s2_max = s2_time.max()
bin_thres = [0, 0.5, 1, 2, 3, 5, 10]
if s2_max >= bin_thres[-1]:
    bin_thres.append(int(np.ceil(s2_max)))

s2_bin_cnts = np.histogram(s2_time, bins=bin_thres)[0]

colors = sns.color_palette("Blues_r", len(s2_bin_cnts))

sns.barplot(x=list(range(len(s2_bin_cnts))), y=s2_bin_cnts, ax=ax1, color=color1)
ax1.bar_label(ax1.containers[0], fmt='%d', label_type='edge')
ax1.set_ybound(0, 10500)
ax1.set_xticks(np.arange(len(bin_thres) - 1))
ax1.set_xticklabels([f"[{bin_thres[i]},{bin_thres[i + 1]})" for i in range(len(bin_thres) - 1)], rotation=30, fontsize=12)
ax1.set_yticks(np.arange(0, 12000, 2000))
ax1.set_yticklabels(["{}k".format(i) for i in range(0, 12, 2)])

ax1.set_title("w/o Safe Projection")

# =============================================================================

ax2 = fig.add_subplot(122)

ax2.set_ylabel("Frequency")
ax2.set_xlabel("Time (s)")

ax2.grid(color='white', alpha=1, axis='y', linestyle="-", linewidth=1.5)
ax2.set_facecolor((0.94, 0.94, 0.94))
ax2.tick_params(axis='x', which='both', length=0)
ax2.set_axisbelow(True)

total_time = s3_data["Total Time"].to_numpy()
total_max = total_time.max()
bin_thres = [0, 0.5, 1, 2, 3, 5, 10]
if total_max >= bin_thres[-1]:
    bin_thres.append(int(np.ceil(total_max)))

total_bin_cnts = np.histogram(total_time, bins=bin_thres)[0]

sns.barplot(x=list(range(len(total_bin_cnts))), y=total_bin_cnts, ax=ax2, color=color2)
ax2.bar_label(ax2.containers[0], fmt='%d', label_type='edge')
ax2.set_ybound(0, 10500)
ax2.set_xticks(np.arange(len(bin_thres) - 1))
ax2.set_xticklabels([f"[{bin_thres[i]},{bin_thres[i + 1]})" for i in range(len(bin_thres) - 1)], rotation=30, fontsize=12)
ax2.set_yticks(np.arange(0, 12000, 2000))
ax2.set_yticklabels(["{}k".format(i) for i in range(0, 12, 2)])

ax2.set_title("w/ Safe Projection")

sns.despine(left=True, bottom=True, right=True)

fig.tight_layout()
plt.show()

fig.savefig(os.path.join(s3_root_dir, "stage2_stage3_time_hist.png"), dpi=300)

print(f"Total time under 2s ratio: {np.sum(total_time < 2)}/{len(total_time)}={np.sum(total_time < 2) / len(total_time)}")
