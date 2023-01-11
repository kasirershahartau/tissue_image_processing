import shutil
import pathlib
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os, sys
import subprocess


# Change before running
folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\analysis after epyseg\\"




def combine_frame_compare_results(folder):
    # Contact length with SC
    # HC_file = "SC_HC_contact_length_compare_data"
    # SC_file = "SC_SC_contact_length_compare_data"
    # title = "Average contact length P0"
    # y_label = "Contact length (#pixels)"
    # HC_data_name = "HC contact length"
    # SC_data_name = "SC contact length"

    # Area
    HC_file = "HC area over time data"
    SC_file = "SC area over time data"
    title = "Average area P0"
    y_label = "Area (#pixels^2)"
    HC_data_name = "area"
    SC_data_name = "area"

    # Roundness
    # HC_file = "HC roundness over time data"
    # SC_file = "SC roundness over time data"
    # title = "Average roundness E17.5"
    # y_label = "Roundness"
    # HC_data_name = "roundness"
    # SC_data_name = "roundness"

    # Neighbors
    # HC_file = "SC_HC_neighbors_compare_data"
    # SC_file = "SC_SC_neighbors_compare_data"
    # title = "Average # of neighbors for SC P0"
    # y_label = "#neighbors"
    # HC_data_name = "HC neighbors"
    # SC_data_name = "SC neighbors"

    # Combining and plotting
    labels = ["Initial", "12 hours", "24 hours", "36 hours", "48 hours"]
    HC_data = pd.read_pickle(os.path.join(folder, HC_file))
    SC_data = pd.read_pickle(os.path.join(folder, SC_file))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars


    font_size = 25
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, HC_data[HC_data_name + " average"], width, label='HC', yerr=HC_data[HC_data_name + " se"], alpha=0.5, ecolor='blue', capsize=10)
    rects2 = ax.bar(x + width/2, SC_data[SC_data_name + " average"], width, label='SC', yerr=SC_data[SC_data_name + " se"], alpha=0.5, ecolor='orange', capsize=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.set_xticks(x, labels, fontsize=font_size)
    ax.legend(loc="lower left")

    ax.bar_label(rects1, labels=HC_data["N"], padding=3, fontsize=font_size)
    ax.bar_label(rects2, labels=SC_data["N"], padding=3,fontsize=font_size)

    fig.tight_layout()

    plt.show()

from scipy.optimize import curve_fit

ellipse_folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-24_E17.5_circular_ablation"
def fit_circular_ablation_results(folder, initial_radius):
    font_size = 35
    major_axis_data = pd.read_pickle(os.path.join(folder,"inner_elipse_major_axis_data"))
    minor_axis_data = pd.read_pickle(os.path.join(folder, "inner_elipse_minor_axis_data"))
    eccentricity_data = pd.read_pickle(os.path.join(folder, "inner_elipse_eccentricity_data"))
    time = np.array([0, 5, 10, 15, 20, 25])
    time_fit = np.linspace(0,25,300)
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

    popt_major, pcov_major = curve_fit(lambda t, a: a, time[:-1], major[:-1], p0=[60],
                                       sigma=major_err[:-1])
    popt_minor, pcov_minor = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a, time[:-1], minor[:-1], p0=[50, 0.5],
                                       sigma=minor_err[:-1])
    popt_eccentricity, pcov_eccentricity = curve_fit(lambda t, a, b:  a * (1 - np.exp(-b * t)), time[:-1], eccentricity[:-1], p0=[0.075, 0.5],
                                       sigma=eccentricity_err[:-1])
    plt.rcParams.update({'font.size': font_size})
    fig1, ax1 = plt.subplots(figsize=(10,10))

    ax1.errorbar(time, major, yerr=major_err, fmt="*", markersize=16, label="Data")
    ax1.plot(time, popt_major[0]*np.ones(time.shape), label="Fit")
    ax1.set_xlabel("Time (minutes)", fontsize=font_size)
    ax1.set_ylabel("Major axis (microns)", fontsize=font_size)
    ax1.legend(loc="upper left")
    print("Major axis results: constant=%f+-%f" % (popt_major[0], np.sqrt(pcov_major[0,0])))
    plt.show()
    fig2, ax2 = plt.subplots(figsize=(10,10))
    ax2.errorbar(time, minor, yerr=minor_err, fmt="*", markersize=16, label="Data")
    ax2.plot(time_fit, (initial_radius - popt_minor[0]) * np.exp(-popt_minor[1] * time_fit) + popt_minor[0], label="Fit")
    ax2.set_xlabel("Time (minutes)", fontsize=font_size)
    ax2.set_ylabel("Minor axis (microns)", fontsize=font_size)
    print("Minor axis results: constant=%f+-%f exponent=%f+-%f" % (popt_minor[0], np.sqrt(pcov_minor[0,0]),
                                                          popt_minor[1], np.sqrt(pcov_minor[1,1])))
    plt.show()
    fig3, ax3 = plt.subplots(figsize=(10,10))
    ax3.errorbar(time, eccentricity, yerr=eccentricity_err, fmt="*", markersize=16, label="Data")
    ax3.plot(time_fit, popt_eccentricity[0] * (1 - np.exp(-popt_eccentricity[1] * time_fit)), label="Fit")
    ax3.set_xlabel("Time (minutes)", fontsize=font_size)
    ax3.set_ylabel("Eccentricity", fontsize=font_size)
    print("Eccentricity results: constant=%f+-%f exponent=%f+-%f" % (popt_eccentricity[0], np.sqrt(pcov_eccentricity[0,0]),
                                                                     popt_eccentricity[1], np.sqrt(pcov_eccentricity[1,1])))
    plt.show()

def combine_single_cell_results(folder, initial_time=-1, final_time=-1, differentiation_time=-1):
    font_size = 22
    plt.rcParams.update({'font.size': font_size})
    roundness_file = "cell_895_roundness_data"
    atoh_level_file = "cell_895_atoh_level_data"
    y1_label = "Roundness"
    y2_label = "Mean intensity (a.u.)"

    roundness_data = pd.read_pickle(os.path.join(folder, roundness_file))
    atoh_level_data = pd.read_pickle(os.path.join(folder, atoh_level_file))
    time = roundness_data["Time"].to_numpy()
    atoh_level = atoh_level_data["Mean atoh intensity"].to_numpy()
    roundness = roundness_data["roundness"].to_numpy()

    if initial_time > 0:
        atoh_level = atoh_level[time >= initial_time]
        roundness = roundness[time >= initial_time]
        time = time[time >= initial_time]
    if final_time > initial_time:
        atoh_level = atoh_level[time <= final_time]
        roundness = roundness[time <= final_time]
        time = time[time <= final_time]
    if differentiation_time > 0:
        time -= differentiation_time

    fig, ax = plt.subplots()
    # make a plot
    graph1 = ax.plot(time/60, roundness, "ro", markersize=16)[0]
    ax.set_xlabel("Time (hours)", fontsize=font_size)
    ax.set_ylabel(y1_label, color="red", fontsize=font_size)
    ax.tick_params(axis='y', color='red', labelcolor='red')
    ax2 = ax.twinx()
    graph2 = ax2.plot(time/60, atoh_level, "b*", markersize=16)[0]
    ax2.set_ylabel(y2_label, color="blue", fontsize=font_size)
    ax2.tick_params(axis='y', color='blue', labelcolor='blue')
    if differentiation_time > 0:
        ymin, ymax = ax.get_ylim()
        ax.plot([0, 0],[ymin, ymax], "m--")
        ax.set_ylim((ymin, ymax))
    plt.tight_layout()
    # make a movie
    out_path = os.path.join(folder, "diff_plot.gif")
    plot_animation((time/60, time/60), (roundness, atoh_level), (graph1, graph2), fig, out_path)



def plot_animation(x, y, graphs, fig, out_path):
    movies_dir = pathlib.Path(tempfile.mkdtemp())
    def animate(i):
        for plot_idx in range(len(graphs)):
            graphs[plot_idx].set_data(x[plot_idx][:i+1], y[plot_idx][:i+1])
        fig.savefig(movies_dir / f"movie_{i:04d}.png")

        return fig

    for i in range(len(x[0])):
        animate(i)
    proc = subprocess.run(
        ["C:\Program Files\ImageMagick-7.1.0-Q16-HDRI\convert", (movies_dir / "movie_*.png").as_posix(), out_path]
    )
    shutil.rmtree(movies_dir)

if __name__ == "__main__":
    # fit_circular_ablation_results(ellipse_folder, 60)
    # combine_frame_compare_results(folder)
    combine_single_cell_results(folder, 1000, 2800, 1900)


