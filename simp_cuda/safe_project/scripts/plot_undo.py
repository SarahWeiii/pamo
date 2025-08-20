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

color3 = "#B260FF"
color2 = "#60B6F0"
color1 = "#F87167"

root_dir = "output/undo"

# w_path = os.path.join(root_dir, "w_512.txt")
# wo_path = os.path.join(root_dir, "wo.txt")
w_path = os.path.join(root_dir, "ours_DMC.txt")
wo_path = os.path.join(root_dir, "wo_remesh.txt")
mc_path = os.path.join(root_dir, "MC.txt")

undo_re = re.compile("h_n_edges_undo (\d+)")
n_edges_re = re.compile("n_edges (\d+)")

with open(w_path, "r") as f:
    w_lines = f.readlines()
    w_undo_iters = np.array([int(undo_re.search(l).group(1)) for l in w_lines if undo_re.search(l)])
    w_n_edges = np.array([int(n_edges_re.search(l).group(1)) for l in w_lines if n_edges_re.search(l)])
    
with open(wo_path, "r") as f:
    wo_lines = f.readlines()
    wo_undo_iters = np.array([int(undo_re.search(l).group(1)) for l in wo_lines if undo_re.search(l)])
    wo_n_edges = np.array([int(n_edges_re.search(l).group(1)) for l in wo_lines if n_edges_re.search(l)])
    
with open(mc_path, "r") as f:
    mc_lines = f.readlines()
    mc_undo_iters = np.array([int(undo_re.search(l).group(1)) for l in mc_lines if undo_re.search(l)])
    mc_n_edges = np.array([int(n_edges_re.search(l).group(1)) for l in mc_lines if n_edges_re.search(l)])

# assert len(w_undo_iters) == len(wo_undo_iters), f"Different number of iterations: {len(w_undo_iters)} vs {len(wo_undo_iters)}"
# n_iters = max(len(w_undo_iters), len(wo_undo_iters))
n_iters = 200
w_undo_iters = w_undo_iters[:n_iters]
wo_undo_iters = wo_undo_iters[:n_iters]
mc_undo_iters = mc_undo_iters[:n_iters]
w_n_edges = w_n_edges[:n_iters]
wo_n_edges = wo_n_edges[:n_iters]
mc_n_edges = mc_n_edges[:n_iters]

# x_axis = np.arange(0, n_iters, 1)
    
fig = plt.figure(figsize=(7, 3))
ax1 = fig.add_subplot(121)

ax1.set_ylabel("\# of Undone Edges")
ax1.set_xlabel("Iteration $t$")
ax1.set_axisbelow(True)
ax1.tick_params(axis='both', which='both', length=0)

ax1.grid(color='white', alpha=1, axis='y', linestyle="-", linewidth=1.5)
ax1.set_facecolor((0.94, 0.94, 0.94))

sns.lineplot(x=np.arange(len(wo_undo_iters)), y=wo_undo_iters, color=color1)
sns.lineplot(x=np.arange(len(w_undo_iters)), y=w_undo_iters, color=color2)
sns.lineplot(x=np.arange(len(mc_undo_iters)), y=mc_undo_iters, color=color3)

ax1.text(0.61, 0.50, "w/o remesh", transform=ax1.transAxes, color=color1)
ax1.text(0.69, 0.31, "w/ MC", transform=ax1.transAxes, color=color3)
ax1.text(0.51, 0.07, "w/ DualMC", transform=ax1.transAxes, color=color2)


ax2 = fig.add_subplot(122)
ax2.set_ylabel("\# of Remaining Edges")
ax2.set_xlabel("Iteration $t$")
ax2.set_axisbelow(True)
ax2.tick_params(axis='both', which='both', length=0)

ax2.set_yscale("log")

ax2.grid(color='white', alpha=1, axis='y', linestyle="-", linewidth=1.5)
ax2.set_facecolor((0.94, 0.94, 0.94))

sns.lineplot(x=np.arange(len(wo_n_edges)), y=wo_n_edges, color=color1)
sns.lineplot(x=np.arange(len(w_n_edges)), y=w_n_edges, color=color2)
sns.lineplot(x=np.arange(len(mc_n_edges)), y=mc_n_edges, color=color3)

# plot horizontal line
ax2.axhline(y=682, color='orange', linestyle='--')

ax2.text(0.61, 0.70, "w/o remesh", transform=ax2.transAxes, color=color1)
ax2.text(0.67, 0.52, "w/ MC", transform=ax2.transAxes, color=color3)
ax2.text(0.5, 0.23, "w/ DualMC", transform=ax2.transAxes, color=color2)
ax2.text(0.2, 0.07, "target", transform=ax2.transAxes, color='orange')

sns.despine(left=True, bottom=True, right=True)

fig.tight_layout()
fig.savefig(os.path.join(root_dir, "undo_remesh_add_MC.png"), dpi=300)
plt.show()


