import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Change before running
folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-12-p0_utricle_ablation\\position4-analysis"

def combine_results(folder):
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

from scipy.optimize import curve_fit

ellipse_folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-24_E17.5_circular_ablation"
def fit_circular_ablation_results(folder, initial_radius):
    major_axis_data =  pd.read_pickle(os.path.join(folder,"inner_elipse_major_axis_data"))
    minor_axis_data = pd.read_pickle(os.path.join(folder, "inner_elipse_minor_axis_data"))
    eccentricity_data = pd.read_pickle(os.path.join(folder, "inner_elipse_eccentricity_data"))
    time = np.array([0,15,30,45,60, 75])
    eccentricity = eccentricity_data["inner ellipse:eccentricity average"]
    eccentricity_err = eccentricity_data["inner ellipse:eccentricity se"]
    major = major_axis_data["inner ellipse:semi-major average"]*0.1
    major_err = major_axis_data["inner ellipse:semi-major se"]*0.1
    minor = minor_axis_data["inner ellipse:semi-minor average"]*0.1
    minor_err = minor_axis_data["inner ellipse:semi-minor se"]*0.1

    minor = np.hstack([initial_radius, minor])
    major = np.hstack([initial_radius, major])
    eccentricity = np.hstack([0, eccentricity])
    minor_err = np.hstack([0.1, minor_err])
    major_err = np.hstack([0.1, major_err])
    eccentricity_err = np.hstack([0.001, eccentricity_err])

    popt_major, pcov_major = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a, time[:-1], major[:-1], p0=[50 , 10],
                                       sigma=major_err[:-1])
    popt_minor, pcov_minor = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a, time[:-1], minor[:-1], p0=[50, 0.5],
                                       sigma=minor_err[:-1])
    popt_eccentricity, pcov_eccentricity = curve_fit(lambda t, a, b:  a * (1 - np.exp(-b * t)), time[:-1], eccentricity[:-1], p0=[0.075, 0.5],
                                       sigma=eccentricity_err[:-1])
    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1, 3)
    ax[0].errorbar(time, major, yerr=major_err, fmt="*", markersize=16, label="Data")
    ax[0].plot(time, (initial_radius - popt_major[0])*np.exp(-popt_major[1]*time) + popt_major[0], label="Fit")
    ax[0].set_xlabel("Time (minutes)", fontsize=16)
    ax[0].set_ylabel("Major axis (microns)", fontsize=16)
    ax[0].legend(loc="upper left")
    print("Major axis results: constant=%f+-%f exponent=%f+-%f" % (popt_major[0], np.sqrt(pcov_major[0,0]),
                                                                   popt_major[1], np.sqrt(pcov_major[1,1])))

    ax[1].errorbar(time, minor, yerr=minor_err, fmt="*", markersize=16, label="Data")
    ax[1].plot(time, (initial_radius - popt_minor[0]) * np.exp(-popt_minor[1] * time) + popt_minor[0], label="Fit")
    ax[1].set_xlabel("Time (minutes)", fontsize=16)
    ax[1].set_ylabel("Minor axis (microns)", fontsize=16)
    print("Minor axis results: constant=%f+-%f exponent=%f+-%f" % (popt_minor[0], np.sqrt(pcov_minor[0,0]),
                                                                   popt_minor[1], np.sqrt(pcov_minor[1,1])))

    ax[2].errorbar(time, eccentricity, yerr=eccentricity_err, fmt="*", markersize=16, label="Data")
    ax[2].plot(time, popt_eccentricity[0] * (1 - np.exp(-popt_eccentricity[1] * time)), label="Fit")
    ax[2].set_xlabel("Time (minutes)", fontsize=16)
    ax[2].set_ylabel("Eccentricity", fontsize=16)
    print("Eccentricity results: constant=%f+-%f exponent=%f+-%f" % (popt_eccentricity[0], np.sqrt(pcov_eccentricity[0,0]),
                                                                     popt_eccentricity[1], np.sqrt(pcov_eccentricity[1,1])))
    plt.subplots_adjust(wspace=0.4)
    plt.show()





