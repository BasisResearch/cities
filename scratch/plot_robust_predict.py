import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dill


# I ran all of these component experiments separately, so there is some variability in the ground truth
# MAP optimization, which I'm just averaging over manually here. These are the ground truth MAP effect sizes.
ground_truth = torch.tensor([
    1.9721251726150513,
    1.9721604585647583,
    1.9723777770996094,
    1.9721548557281494,
    1.9718914031982422,
    1.9718817472457886,
]).mean()

# Load robust_predict_results_408.pt and 816 and 1632 with dill
with open("robust_predict_results_408_1000.pt", "rb") as f:
    plugins_and_corrections_408_1000 = dill.load(f)

with open("robust_predict_results_816_1000.pt", "rb") as f:
    plugins_and_corrections_816_1000 = dill.load(f)

with open("robust_predict_results_1632_1000.pt", "rb") as f:
    plugins_and_corrections_1632_1000 = dill.load(f)

with open("robust_predict_results_408_5000.pt", "rb") as f:
    plugins_and_corrections_408_5000 = dill.load(f)

with open("robust_predict_results_816_5000.pt", "rb") as f:
    plugins_and_corrections_816_5000 = dill.load(f)

with open("robust_predict_results_1632_5000.pt", "rb") as f:
    plugins_and_corrections_1632_5000 = dill.load(f)

# Print the lens
print(f"Length of 408: {len(plugins_and_corrections_408_1000)}")
print(f"Length of 816: {len(plugins_and_corrections_816_1000)}")
print(f"Length of 1632: {len(plugins_and_corrections_1632_1000)}")
print(f"Length of 408_5000: {len(plugins_and_corrections_408_5000)}")
print(f"Length of 816_5000: {len(plugins_and_corrections_816_5000)}")
print(f"Length of 1632_5000: {len(plugins_and_corrections_1632_5000)}")


# Do the above but in subplots with shared x axis, also, make a function we can reuse for each of the plugin and correc
def plot_plugin_corrected(plugins_and_corrections, dataset_size, ground_truth, ax):
    ax.set_title(f"Number of Census Tracts: {dataset_size}")
    sns.kdeplot([x.plugin.item() for x in plugins_and_corrections], label=f"Plugin Estimate", color='blue', ax=ax)
    sns.kdeplot([x.corrected.item() for x in plugins_and_corrections], label=f"Corrected Estimate", linestyle='--', color='orange', ax=ax)
    plugin_mean = np.mean([x.plugin.item() for x in plugins_and_corrections])
    ax.axvline(plugin_mean, color='blue', label="Plugin Mean")
    corrected_mean = np.mean([x.corrected.item() for x in plugins_and_corrections])
    ax.axvline(corrected_mean, color='orange', linestyle='--', label="Corrected Mean")

    # Plot bars on the top of the frame mapping to standard error (stdev / root n)
    plugin_std = np.std([x.plugin.item() for x in plugins_and_corrections])
    corrected_std = np.std([x.corrected.item() for x in plugins_and_corrections])
    plugin_se = plugin_std / np.sqrt(len(plugins_and_corrections))
    corrected_se = corrected_std / np.sqrt(len(plugins_and_corrections))

    ax.errorbar(plugin_mean, 2.8, xerr=plugin_se, fmt='|', color='blue', linewidth=6.0)
    ax.errorbar(corrected_mean, 3.0, xerr=corrected_se, fmt='|', color='orange', linewidth=6.0)
    ax.errorbar(plugin_mean, 2.8, xerr=2 * plugin_se, fmt='|', color='blue', linewidth=3.0)
    ax.errorbar(corrected_mean, 3.0, xerr=2 * corrected_se, fmt='|', color='orange', linewidth=3.0)
    ax.errorbar(plugin_mean, 2.8, xerr=3 * plugin_se, fmt='|', color='blue', linewidth=1.0)
    ax.errorbar(corrected_mean, 3.0, xerr=3 * corrected_se, fmt='|', color='orange', linewidth=1.0)

    # remove y axis ticks and labels
    ax.yaxis.set_visible(False)

    ax.axvline(ground_truth.item(), color='black', label="Ground Truth", linestyle=':')


fig, axs = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 10))

[ax.set_xlim((0.5, 3.5)) for ax in axs]

plot_plugin_corrected(plugins_and_corrections_408_1000, 408, ground_truth, axs[0])
plot_plugin_corrected(plugins_and_corrections_816_1000, 816, ground_truth, axs[1])
plot_plugin_corrected(plugins_and_corrections_1632_1000, 1632, ground_truth, axs[2])
EFIM = 1000

# plot_plugin_corrected(plugins_and_corrections_408_5000, 408, ground_truth, axs[0])
# plot_plugin_corrected(plugins_and_corrections_816_5000, 816, ground_truth, axs[1])
# plot_plugin_corrected(plugins_and_corrections_1632_5000, 1632, ground_truth, axs[2])
# EFIM = 5000

plt.suptitle(f"Plugin and Corrected Estimates over 100 Samples (Empirical FIM Estimated w/ {EFIM} Samples)")
plt.xlabel("Average Estimated Effect of Fully Removing Parking Limits on (Standardized) Permit Applications")
plt.ylabel("Density")

# Grab a subset of legend elements (plugin line, correction line, and ground truth line).
handles, labels = axs[2].get_legend_handles_labels()
handles = [handles[0], handles[1], handles[-1]]
labels = [labels[0], labels[1], labels[-1]]
axs[0].legend(handles, labels)

plt.tight_layout()
plt.show()
