import torch
import os
import numpy as np
import matplotlib.pyplot as plt

loss_path = 'losses'
# loss_names = ["RTE-maxeps0.2-res-dict-anchor.pt", "RTE-maxeps0.2-res-dict-0.pt", "RTE-maxeps0.2-res-dict-1.pt"]
# loss_names = ["RTE-maxeps0.3-res-dict-anchor.pt", "RTE-maxeps0.3-res-dict-0.pt", "RTE-maxeps0.3-res-dict-1.pt"]
# loss_names = ["RTE-maxeps0.3-res-dict-0-StdAdvAsAnchor.pt", "RTE-maxeps0.3-res-dict-anchor-StdAdvAsAnchor.pt", "RTE-maxeps0.3-res-dict-1-StdAdvAsAnchor.pt"]
# loss_names = ["CoLA-maxeps0.25-res-dict-anchor.pt", "CoLA-maxeps0.25-res-dict-0.pt", "CoLA-maxeps0.25-res-dict-1.pt"]
# loss_names = ["CoLA-maxeps0.25-res-dict-0-StdAdvAsAnchor.pt", "CoLA-maxeps0.25-res-dict-anchor-StdAdvAsAnchor.pt", "CoLA-maxeps0.25-res-dict-1-StdAdvAsAnchor.pt"]
loss_names = ["MRPC-maxeps0.45-res-dict-anchor.pt", "MRPC-maxeps0.45-res-dict-0.pt", "MRPC-maxeps0.45-res-dict-1.pt"]
# loss_names = ["MRPC-maxeps0.45-res-dict-0-StdAdvAsAnchor.pt", "MRPC-maxeps0.45-res-dict-anchor-StdAdvAsAnchor.pt", "MRPC-maxeps0.45-res-dict-1-StdAdvAsAnchor.pt"]
legend_names = ["Ours", "PGD", "Baseline"]
out_dir = os.path.join(loss_path, "pics")

max_val = -1
max_loss_list_list, init_loss_list_list = [], []
for legend_name, lname in zip(legend_names[::-1], loss_names[::-1]):
    loss_dict = torch.load(os.path.join(loss_path, lname))
    max_loss_list = list(loss_dict['max_loss_dict'].values())
    init_loss_list = list(loss_dict['init_loss_dict'].values())

    loss_inc = np.array(max_loss_list) - np.array(init_loss_list)

    max_loss_list_list.append(max_loss_list)
    init_loss_list_list.append(init_loss_list)
    mean_max_loss = np.mean(max_loss_list)
    var_max_loss = np.std(max_loss_list)
    mean_init_loss = np.mean(init_loss_list)
    var_init_loss = np.std(init_loss_list)
    median_max_loss = np.median(max_loss_list)
    median_init_loss = np.median(init_loss_list)

    mean_inc_loss = np.mean(loss_inc)
    median_inc_loss = np.median(loss_inc)
    std_inc_loss = np.std(loss_inc)

    if max(max_loss_list) > max_val:
        max_val = max(max_loss_list)
    # print("{}, {}, median(var)/mean max loss: {:e}({:e}) /{:e}, median(var) / mean init loss: {:e}({:e})/{:e}".format(
    #         legend_name, lname, median_max_loss, var_max_loss, mean_max_loss, median_init_loss, var_init_loss, mean_init_loss))
    print("Num samples: {}".format(len(max_loss_list)))
    print("{}, {}, median(var)/mean max loss: {:e}({:e}) /{:e}, median(var) / mean init loss: {:e}({:e})/{:e}".format(
        legend_name, lname, median_inc_loss, std_inc_loss, mean_inc_loss, median_init_loss, var_init_loss,
        mean_init_loss))

# bins = np.linspace(0, max_val*1.01, 1000)
bins = np.linspace(0, 0.01, 200)
for n, (lname, max_loss_list, init_loss_list) in enumerate(zip(legend_names, max_loss_list_list, init_loss_list_list)):
    plt.hist(max_loss_list, bins, alpha=0.5, label=lname + " (Max)")
    # plt.hist(init_loss_list, bins, alpha=0.5, label=lname + " (Init)")

# plt.xscale("log")
plt.xlabel("Loss Value")
plt.ylabel("Frequency")
plt.legend(loc='upper right')
out_fname = "{}.pdf".format('-'.join(loss_names[0].split('-')[:2]))
plt.savefig(os.path.join(out_dir, out_fname))