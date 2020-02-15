import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import pdb

in_dir = "losses"
out_dir = "losses/pics"

# ours_fname = "RTE-ours-4.pt"
# std_fname = "RTE-std-4.pt"
# baseline_fname = "RTE-baseline-5.pt"
ours_fname = "RTE-ours-4-0.021.pt" # loss: 0.014
std_fname = "RTE-std-5-0.021.pt"
baseline_fname = "RTE-baseline-5-0.021.pt" #

ours_dict = torch.load(os.path.join(in_dir, ours_fname))
std_dict = torch.load(os.path.join(in_dir, std_fname))
base_dict = torch.load(os.path.join(in_dir, baseline_fname))

labels = ["Baseline", "Std-Adv", "Ours"]

for key in ours_dict.keys():

    plt.figure()
    max_val = max(ours_dict[key]+ std_dict[key]+ base_dict[key])
    bins = np.linspace(0, max_val*1.05, 200)
    plt.hist(base_dict[key], bins, alpha=0.5, label=labels[0])
    plt.hist(std_dict[key], bins, alpha=0.5, label=labels[1])
    plt.hist(ours_dict[key], bins, alpha=0.5, label=labels[2])
    plt.xlabel("Loss Value")
    plt.ylabel("Frequency")
    plt.legend(loc='upper right')

    out_fname = "Sample_{}_{}_{}_{}.pdf".format(key, baseline_fname, std_fname, ours_fname)
    plt.savefig(os.path.join(out_dir, out_fname))
