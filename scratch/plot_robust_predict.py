import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import dill


# HACK save and retrieve this:
ground_truth = torch.tensor(1.972286581993103)


# Load "robust_predict_results.pt" with dill
with open("robust_predict_results.pt", "rb") as f:
    plugins_and_corrections = dill.load(f)


# HACK save and retrieve this, then slice into it according to the length of plugins_and_corrections.\
# This lets us run when the other script has only partially run.
sizes = [int(100 * 1.1 ** i) for i in range(len(plugins_and_corrections))]


# 1. A plot with the groundtruth vline, plugin, and corrected estimate.
plt.figure()
plt.axhline(ground_truth.item(), color='red', label="Ground Truth")
plt.plot(sizes, [x.plugin.item() for x in plugins_and_corrections], label="Plugin Estimate")
plt.plot(sizes, [x.plugin_actual.item() for x in plugins_and_corrections], label="Plugin*")
plt.plot(sizes, [x.corrected.item() for x in plugins_and_corrections], label="Corrected", linestyle='--')
plt.plot(sizes, [x.corrected_actual.item() for x in plugins_and_corrections], label="Corrected*", linestyle='--')
plt.xscale('log')

plt.xlabel("Dataset Size")
plt.ylabel("Housing Units Delta")
plt.legend()

# 2. A plot of the absolute bias of the plugin and the corrected estimate.
plt.figure()
plt.plot(sizes, [abs(x.plugin.item() - ground_truth.item()) for x in plugins_and_corrections], label="Plugin")
plt.plot(sizes, [abs(x.plugin_actual.item() - ground_truth.item()) for x in plugins_and_corrections], label="Plugin*")
plt.plot(sizes, [abs(x.corrected.item() - ground_truth.item()) for x in plugins_and_corrections], label="Corrected", linestyle='--')
plt.plot(sizes, [abs(x.corrected_actual.item() - ground_truth.item()) for x in plugins_and_corrections], label="Corrected*", linestyle='--')
plt.xscale('log')

plt.xlabel("Dataset Size")
plt.ylabel("Absolute Error")
plt.legend()

# 3. A plot of the correction term and what the correction shouuld be (diff between plugin and ground truth).
plt.figure()
plt.plot(sizes, [x.corrected.item() - x.plugin.item() for x in plugins_and_corrections], label="Correction")
plt.plot(sizes, [ground_truth.item() - x.plugin.item() for x in plugins_and_corrections], label="Perfect Correction")
plt.plot(sizes, [ground_truth.item() - x.plugin_actual.item() for x in plugins_and_corrections], label="Perfect Correction*")
plt.xscale('log')

plt.xlabel("Dataset Size")
plt.ylabel("Correction")
plt.legend()

plt.show()
