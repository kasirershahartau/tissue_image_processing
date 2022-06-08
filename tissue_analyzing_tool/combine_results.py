import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Change before running
folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-12-p0_utricle_ablation\\position4-analysis"

# Contact length with SC
# HC_file = "SC_HC_contact_length_compare_data"
# SC_file = "SC_SC_contact_length_compare_data"
# title = "Average contact length P0"
# y_label = "Contact length (#pixels)"
# HC_data_name = "HC contact length"
# SC_data_name = "SC contact length"

# Area
# HC_file = "HC_area_compare_data"
# SC_file = "SC_area_compare_data"
# title = "Average area P0"
# y_label = "Area (#pixels^2)"
# HC_data_name = "area"
# SC_data_name = "area"

# Roundness
# HC_file = "HC_roundness_compare_data"
# SC_file = "SC_roundness_compare_data"
# title = "Average roundness P0"
# y_label = "Roundness"
# HC_data_name = "roundness"
# SC_data_name = "roundness"

# Neighbors
HC_file = "SC_HC_neighbors_compare_data"
SC_file = "SC_SC_neighbors_compare_data"
title = "Average # of neighbors for SC P0"
y_label = "#neighbors"
HC_data_name = "HC neighbors"
SC_data_name = "SC neighbors"

# Combining and plotting
labels = ["Initial", "24 hours", "48 hours"]
HC_data = pd.read_pickle(os.path.join(folder, HC_file))
SC_data = pd.read_pickle(os.path.join(folder, SC_file))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, HC_data[HC_data_name + " average"], width, label='HC', yerr=HC_data[HC_data_name + " se"], alpha=0.5, ecolor='blue', capsize=10)
rects2 = ax.bar(x + width/2, SC_data[SC_data_name + " average"], width, label='SC', yerr=SC_data[SC_data_name + " se"], alpha=0.5, ecolor='orange', capsize=10)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel(y_label)
ax.set_title(title)
ax.set_xticks(x, labels)
ax.legend()

ax.bar_label(rects1, labels=HC_data["N"], padding=3)
ax.bar_label(rects2, labels=SC_data["N"], padding=3)

fig.tight_layout()

plt.show()