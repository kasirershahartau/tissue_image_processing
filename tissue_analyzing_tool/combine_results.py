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
E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\position3-analysis\\Event statistics\\",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-23_E17.5_utricle_atoh_zo\\position4-analysis"]
P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\",
              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position2-analysis\\"]
Rho_inhibition_E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-12-01_E17.5_utricle_rho_inhibition\\position2_event_statistics\\"]
Rho_inhibition_P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2023-06-25_P0_atoh_zo_rock_inhibitor\\position3_event_statistics\\"]

def combine_frame_compare_results():
    # Contact length with SC
    # HC_file = "SC_HC_contact_length_compare_data"
    # SC_file = "SC_SC_contact_length_compare_data"
    # title = "Average contact length P0"
    # y_label = "Contact length (#pixels)"
    # HC_data_name = "HC contact length"
    # SC_data_name = "SC contact length"

    # Area
    # HC_file = "HC area over time data"
    # SC_file = "SC area over time data"
    # title = "Average area P0"
    # y_label = "Area (#pixels^2)"
    # HC_data_name = "area"
    # SC_data_name = "area"

    # Roundness
    HC_file = "HC_roundness_initial_data"
    SC_file = "SC_roundness_initial_data"
    y_label = "Roundness"
    HC_data_name = "roundness"
    SC_data_name = "roundness"

    # Neighbors
    # HC_file = "SC_HC_neighbors_compare_data"
    # SC_file = "SC_SC_neighbors_compare_data"
    # title = "Average # of neighbors for SC P0"
    # y_label = "#neighbors"
    # HC_data_name = "HC neighbors"
    # SC_data_name = "SC neighbors"

    # Combining and plotting
    labels = ["E17.5 SCs", "E17.5 HCs", "P0 SCs", "P0 HCs"]
    E17_HC_data = pd.read_pickle(os.path.join(E17_folder, HC_file))
    E17_SC_data = pd.read_pickle(os.path.join(E17_folder, SC_file))
    P0_HC_data = pd.read_pickle(os.path.join(E17_folder, HC_file))
    P0_SC_data = pd.read_pickle(os.path.join(E17_folder, SC_file))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    compare_event_statistics([E17_folder, P0_folder], [SC_file, HC_file], [SC_file, HC_file], labels, [(0,1), (2,3)],
                             [1],["roundness"], ["Roundness"], continues=True, color=["red", "green", "red", "green"],
                             edge_color=["red", "green", "red", "green"])

from scipy.optimize import curve_fit

ellipse_folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-24_E17.5_circular_ablation"
def fit_circular_ablation_results(folder, initial_radius):
    font_size = 40
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

    ax1.errorbar(time, major, yerr=major_err, fmt="*", markersize=30, label="Data", linewidth=6)
    ax1.plot(time, popt_major[0]*np.ones(time.shape), label="Fit", linewidth=6)
    ax1.set_xlabel("Time (minutes)", fontsize=font_size)
    ax1.set_ylabel("Major axis (microns)", fontsize=font_size)
    ax1.legend(loc="upper left")
    print("Major axis results: constant=%f+-%f" % (popt_major[0], np.sqrt(pcov_major[0,0])))
    plt.tight_layout()
    plt.show()
    fig2, ax2 = plt.subplots(figsize=(10,10))
    ax2.errorbar(time, minor, yerr=minor_err, fmt="*", markersize=30, label="Data", linewidth=6)
    ax2.plot(time_fit, (initial_radius - popt_minor[0]) * np.exp(-popt_minor[1] * time_fit) + popt_minor[0], label="Fit", linewidth=6)
    ax2.set_xlabel("Time (minutes)", fontsize=font_size)
    ax2.set_ylabel("Minor axis (microns)", fontsize=font_size)
    print("Minor axis results: constant=%f+-%f exponent=%f+-%f" % (popt_minor[0], np.sqrt(pcov_minor[0,0]),
                                                          popt_minor[1], np.sqrt(pcov_minor[1,1])))
    plt.tight_layout()
    plt.show()
    fig3, ax3 = plt.subplots(figsize=(10,10))
    ax3.errorbar(time, eccentricity, yerr=eccentricity_err, fmt="*", markersize=30, label="Data", linewidth=6)
    ax3.plot(time_fit, popt_eccentricity[0] * (1 - np.exp(-popt_eccentricity[1] * time_fit)), label="Fit", linewidth=6)
    ax3.set_xlabel("Time (minutes)", fontsize=font_size)
    ax3.set_ylabel("Eccentricity", fontsize=font_size)
    print("Eccentricity results: constant=%f+-%f exponent=%f+-%f" % (popt_eccentricity[0], np.sqrt(pcov_eccentricity[0,0]),
                                                                     popt_eccentricity[1], np.sqrt(pcov_eccentricity[1,1])))
    plt.tight_layout()
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


def compare_event_statistics(folder_list, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues, color='white', edge_color='grey'):
    font = {'family': 'sans',
            'size': 25}
    import matplotlib
    matplotlib.rc('font', **font)
    if isinstance(folder_list, tuple):
        folder = folder_list[0]
        ref_folder = folder_list[1]
    else:
        folder = folder_list
        ref_folder = folder

    from statistical_analysis import compare_and_plot_samples
    for data_label, normalization, y_label in zip(data_labels, normalization_list, y_labels):
        if isinstance(folder, list):
            samples = [np.empty(0) for i in range(np.max([len(data_files_list[j]) for j in range(len(data_files_list))]) +
                                         np.max([len(reference_files_list[j]) for j in range(len(reference_files_list))]))]
            for f, rf, d, r in zip(folder, ref_folder, data_files_list, reference_files_list):
                data_list = [pd.read_pickle(os.path.join(f, data_file)) for data_file in d]
                ref_list = [pd.read_pickle(os.path.join(rf, ref_file)) for ref_file in r]
                s = [data[data_label].to_numpy() / normalization for data in data_list]
                s += [ref_data[data_label].to_numpy() / normalization for ref_data in ref_list]
                s = [x[~np.isnan(x)] for x in s]
                for i in range(len(s)):
                    samples[i] = np.hstack([samples[i], s[i]])

        else:
            data_list = [pd.read_pickle(os.path.join(folder, data_file)) for data_file in data_files_list]
            ref_list = [pd.read_pickle(os.path.join(ref_folder, ref_file)) for ref_file in reference_files_list]
            samples = [data[data_label].to_numpy()/normalization for data in data_list]
            samples += [ref_data[data_label].to_numpy()/normalization for ref_data in ref_list]
            samples = [sample[~np.isnan(sample)] for sample in samples]
        style = "violin" if continues else "bar"
        fig, ax, res = compare_and_plot_samples(samples, x_labels, pairs_to_compare, continues=continues,
                                                plot_style=style, color=color, edge_color=edge_color)
        ax.set_ylabel(y_label)
        print(res)
    # ax.set_xticklabels([])
    # ax.set_ylim([0, 3.1])
    # axes[1].set_ylim([0, 3.3])
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

def plot_E17_HC_density_and_fraction():
    folder = E17_folders

    data_files_list = [["HC_fraction_and_density_delamination_data", "HC_fraction_and_density_division_data",
                       "HC_fraction_and_density_differentiation_rad_200_pixels_data"],
                       ["HC_density_and_fraction_delamination_data", "HC_density_and_fraction_division_data",
                       "HC_density_and_fraction_differentiation_data"]]
    reference_files_list = [["HC_fraction_and_density_overall_SC_frame1_data",
                            "HC_fraction_and_density_overall_SC_frame96_data",
                            "HC_fraction_and_density_overall_SC_frame191_data"],
                            ["HC_density_and_fraction_reference_SC_frame1_data",
                            "HC_density_and_fraction_reference_SC_frame97_data"]]
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations", "reference SC initial",
                "reference SC 24h", "reference SC 48h"]
    normalization_list = [0.01, 1]
    pairs_to_compare = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4), (1, 3), (3, 5), (1, 4), (2, 5),
                        (0, 4), (1, 5), (0, 5)]
    y_labels = ["HC density (#HC/micron)", "HC type fraction"]
    data_labels = ["HC density", "HC type_fraction"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True)


def plot_E17_neighbors_by_type():
    folder = E17_folders
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations",
                "reference SC 24h"]
    data_files_list = [["neighbors_by_type_delamination_data", "neighbors_by_type_division_data",
                       "neighbors_by_type_differentiation_data"],
                       ["neighbors_by_type_delamination_data", "neighbors_by_type_division_data",
                        "neighbors_by_type_differentiation_data"]]
    reference_files_list = [["neighbors_by_type_reference_SC_frame96_data"],
                            ["neighbors_by_type_reference_SC_frame97_data"]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1, 2), (1,3), (2,3), (0,3)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)

def plot_E17_rho_inhibition_neighbors_by_type():
    folder = Rho_inhibition_E17_folders
    x_labels = ["Near differentiations",
                "reference SC 24h"]
    data_files_list = [["neighbors_by_type_differentiation_data"]]
    reference_files_list = [["neighbors_by_type_reference_SC_frame91_data"]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)

def plot_E17_second_neighbors_by_type():
    folder = E17_folders
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations",
                "reference SC initial", "reference SC 24h", "reference SC 48h"]
    data_files_list = [["second_neighbors_by_type_delamination_data", "second_neighbors_by_type_division_data",
                       "second_neighbors_by_type_differentiation_data"],
                       ["second_neighbors_by_type_delamination_data", "second_neighbors_by_type_division_data",
                        "second_neighbors_by_type_differentiation_data"]]
    reference_files_list = [["second_neighbors_by_type_reference_SC_frame1_data",
                             "second_neighbors_by_type_reference_SC_frame96_data",
                             "second_neighbors_by_type_reference_SC_frame191_data"],
                            ["second_neighbors_by_type_reference_SC_frame1_data",
                             "second_neighbors_by_type_reference_SC_frame96_data",
                             "second_neighbors_by_type_reference_SC_frame191_data"]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1, 2), (1, 3), (2, 3), (0, 3)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC second neighbors", "HC second neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)


def plot_E17_contact_length_by_type():
    folder = E17_folders
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations",
                "reference SC 24h"]
    data_files_list = [["contact_length_by_type_delamination_data", "contact_length_by_type_division_data",
                       "contact_length_by_type_differentiation_data"],
                       ["contach_length_delamination_data", "contach_length_division_data",
                        "contach_length_differentiation_data"]]
    reference_files_list = [["contact_length_by_type_reference_SC_frame95_data"],
                            ["contach_length_reference_SC_frame97_data"]]
    normalization_list = [10, 10]
    pairs_to_compare = [(0, 1), (1, 2), (1,3), (2,3), (0,3)]
    y_labels = ["contact length with SC neighbors", "contact length with neighbors"]
    data_labels = ["SC contact length", "HC contact length"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True)

def plot_E17_number_of_neighbors():
    folder = E17_folders
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations",
                "reference SC 24h"]
    data_files_list = [["number_of_neighbors_delamination_data", "number_of_neighbors_division_data",
                       "number_of_neighbors_differentiation_data"],
                       ["number_of_neighbors_delamination_data", "number_of_neighbors_division_data",
                        "number_of_neighbors_differentiation_data"]]
    reference_files_list = [["number_of_neighbors_reference_SC_frame96_data"],
                            ["number_of_neighbors_reference_SC_frame97_data"]]
    normalization_list = [1]
    pairs_to_compare = [(2, 3), (1, 3), (0, 3)]
    y_labels = ["# neighbors"]
    data_labels = ["n_neighbors"]
    color = ["red", "blue", "green", "grey"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=color)

def plot_E17_area_and_roundness():
    folder = E17_folders
    x_labels = ["Near delaminations", "Near divisions", "Near differentiations",
                "reference SC 24h"]
    data_files_list = [["area_and_roundness_delamination_data", "area_and_roundness_division_data",
                       "area_and_roundness_differentiation_data"],
                       ["area_and_roundness_delamination_data", "area_and_roundness_division_data",
                        "area_and_roundness_differentiation_data"]]
    reference_files_list = [["area_and_roundness_reference_SC_frame96_data"],
                            ["area_and_roundness_reference_SC_frame97_data"]]
    normalization_list = [100, 1]
    pairs_to_compare = [(2, 3), (1, 3), (0, 3)]
    y_labels = ["Area (um^2)", "Roundness"]
    data_labels = ["area", "roundness"]
    color = ["red", "blue", "green", "grey"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color=color, edge_color=color)

def compare_E17_P0_neighbors_by_type():
    E17_folder = E17_folders
    P0_folder = P0_folders
    # x_labels = ["Differentiating\ncells",  "All\nSCs", "Differentiating\ncells",
    #             "All\nSCs"]
    x_labels = [""]*4
    E17_data_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                           ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame97_data"]]
    P0_reference_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                               ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (2,3), (0,2)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    color = ["blue", "white", "red", "white"]
    edge_color = ["blue", "blue", "red", "red"]

    compare_event_statistics((E17_folder,P0_folder), E17_data_files_list, P0_reference_files_list , x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color)

def compare_E17_P0_rho_inhibition_neighbors_by_type():
    E17_folder = Rho_inhibition_E17_folders
    P0_folder = Rho_inhibition_P0_folders
    # x_labels = ["Differentiating\ncells",  "All\nSCs", "Differentiating\ncells",
    #             "All\nSCs"]
    x_labels = [""] * 4
    E17_data_files_list = [
        ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame91_data"]]
    P0_reference_files_list = [
        ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame93_data"]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (2, 3), (0, 2)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    color = ["blue", "white", "red", "white"]
    edge_color = ["blue", "blue", "red", "red"]

    compare_event_statistics((E17_folder, P0_folder), E17_data_files_list, P0_reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color)

def compare_E17_P0_density_and_fraction():
    E17_folder = E17_folders
    P0_folder = P0_folders
    E17_data_files_list = [["HC_fraction_and_density_overall_SC_frame1_data",
                            "HC_fraction_and_density_overall_SC_frame96_data",
                            "HC_fraction_and_density_overall_SC_frame191_data"],
                           ["HC_density_and_fraction_reference_SC_frame1_data",
                            "HC_density_and_fraction_reference_SC_frame97_data",
                            "HC_density_and_fraction_reference_SC_frame199_data"]]
    P0_reference_files_list = [["HC_density_and_fraction_reference_SC_frame1_data",
                                "HC_density_and_fraction_reference_SC_frame96_data",
                                "HC_density_and_fraction_reference_SC_frame165_data"],
                               ["HC_density_and_fraction_reference_SC_frame1_data",
                                "HC_density_and_fraction_reference_SC_frame96_data"]]
    x_labels = ["E17.5", "+24h", "+48h",# "Diff.\ncells",
                 "P0", "+24h", "+48h"] #"Diff.\ncells"]
    normalization_list = [0.01, 1]
    pairs_to_compare = [(0, 1), (1,2), (2,3), (3,4), (4,5)]
    y_labels = ["", "HC type fraction"]
    data_labels = ["HC density", "HC type_fraction"]
    color = ["blue", "blue", "blue", "red", "red", "red"]
    edge_color = ["blue", "blue", "blue", "red", "red", "red"]
    compare_event_statistics((E17_folder, P0_folder), E17_data_files_list, P0_reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True,  color=color, edge_color=edge_color)

def compare_E17_P0_number_of_neighbors():
    E17_folder = E17_folders
    P0_folder = P0_folders
    E17_data_files_list = [["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame96_data"],
                           ["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame97_data"]]
    P0_reference_files_list = [["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame96_data"],
                               ["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame96_data"]]
    x_labels = ["Differentiating cells E17.5", "All SCs E17.5", "Differentiating cells P0",
                "All SCs P0"]
    normalization_list = [1]
    pairs_to_compare = [(0, 1), (2,3), (0,2)]
    y_labels = ["# neighbors"]
    data_labels = ["n_neighbors"]
    compare_event_statistics((E17_folder, P0_folder), E17_data_files_list, P0_reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)

def compare_E17_P0_area_and_roundness():
    E17_folder = E17_folders
    P0_folder = P0_folders
    E17_data_files_list = [["area_and_roundness_differentiation_data", "area_and_roundness_reference_SC_frame96_data"],
                           ["area_and_roundness_differentiation_data", "area_and_roundness_reference_SC_frame97_data"]]
    P0_reference_files_list = [["area_and_roundness_differentiation_data", "area_and_roundness_reference_SC_frame96_data"],
                               ["area_and_roundness_differentiation_data","area_and_roundness_reference_SC_frame96_data"]]
    x_labels = ["Differentiating cells E17.5", "All SCs E17.5", "Differentiating cells P0",
                "All SCs P0"]
    normalization_list = [100, 1]
    pairs_to_compare = [(0, 1), (2,3), (0,2)]
    y_labels = ["Area (um^2)", "Roundness"]
    data_labels = ["area", "roundness"]
    color = ["blue", "blue", "red", "red"]
    edge_color = ["blue", "blue", "red", "red"]
    compare_event_statistics((E17_folder, P0_folder), E17_data_files_list, P0_reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color=color, edge_color=edge_color)

def compare_E17_P0_contact_length():
    E17_folder = E17_folders
    P0_folder = P0_folders
    E17_data_files_list = [["contact_length_by_type_differentiation_data",
                            "contact_length_by_type_reference_SC_frame95_data"],
                           ["contach_length_differentiation_data",
                            "contach_length_reference_SC_frame97_data"]]
    P0_reference_files_list = [["contact_length_by_type_differentiation_data",
                            "contact_length_by_type_reference_SC_frame96_data"],
                               ["contact_length_by_type_differentiation_data",
                                "contact_length_by_type_reference_SC_frame96_data"]]
    x_labels = ["Differentiating\ncells", "All\nSCs",
                "Differentiating\ncells", "All\nSCs"]
    normalization_list = [10, 10]
    pairs_to_compare = [(0, 1), (2, 3), (0,2)]
    color = ["blue", "blue", "red", "red"]
    edge_color = ["blue", "blue", "red", "red"]
    y_labels = ["contact length with\nSC neighbors (microns)", "contact length with\nHC neighbors (microns)"]
    data_labels = ["SC contact length", "HC contact length"]
    compare_event_statistics((E17_folder, P0_folder), E17_data_files_list, P0_reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color=color, edge_color=edge_color)


def plot_P0_HC_density_and_fraction():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"

    data_files_list = ["HC_density_and_fraction_differentiation_data"]
    reference_files_list = ["HC_density_and_fraction_reference_SC_frame1_data",
                            "HC_density_and_fraction_reference_SC_frame96_data",
                            "HC_density_and_fraction_reference_SC_frame165_data"]
    x_labels = ["Near differentiations", "reference SC initial",
                "reference SC 24h", "reference SC 48h"]
    normalization_list = [0.01, 1]
    pairs_to_compare = [(0, 1), (1, 2), (2, 3), (1, 3)]
    y_labels = ["HC density (#HC/micron)", "HC type fraction"]
    data_labels = ["HC density", "HC type_fraction"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True)


def plot_P0_neighbors_by_type():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"
    x_labels = ["Near differentiations",
                "reference SC initial","reference SC 24h","reference SC 48h"]
    data_files_list = ["neighbors_by_type_differentiation_data"]
    reference_files_list = ["neighbors_by_type_reference_SC_frame1_data","neighbors_by_type_reference_SC_frame96_data",
                            "neighbors_by_type_reference_SC_frame165_data"]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1,2), (2,3)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)


def plot_P0_contact_length_by_type():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"
    x_labels = ["Near differentiations",
                "reference SC initial", "reference SC 24h", "reference SC 48h"]
    data_files_list = ["contact_length_by_type_differentiation_data"]
    reference_files_list = ["contact_length_by_type_reference_SC_frame1_data","contact_length_by_type_reference_SC_frame96_data",
                            "contact_length_by_type_reference_SC_frame165_data"]
    normalization_list = [10, 10]
    pairs_to_compare = [(0, 1), (1,2), (2,3)]
    y_labels = ["contact length with SC neighbors (microns)", "contact length with HC neighbors (microns)"]
    data_labels = ["SC contact length", "HC contact length"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True)

def plot_P0_number_of_neighbors():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"
    x_labels = ["Near differentiations",
                "reference SC initial", "reference SC 24h", "reference SC 48h"]
    data_files_list = ["number_of_neighbors_differentiation_data"]
    reference_files_list = ["number_of_neighbors_reference_SC_frame1_data","number_of_neighbors_reference_SC_frame96_data",
                            "number_of_neighbors_reference_SC_frame165_data"]
    normalization_list = [1]
    pairs_to_compare = [(0, 2), (1,2), (2,3)]
    y_labels = ["# neighbors"]
    data_labels = ["n_neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)

def plot_P0_area_and_roundness():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"
    x_labels = ["Near differentiations",
                "reference SC initial", "reference SC 24h", "reference SC 48h"]
    data_files_list = ["area_and_roundness_differentiation_data"]
    reference_files_list = ["area_and_roundness_reference_SC_frame1_data","area_and_roundness_reference_SC_frame96_data",
                            "area_and_roundness_reference_SC_frame165_data"]
    normalization_list = [100, 1]
    pairs_to_compare = [(0, 1), (1,2), (2,3)]
    y_labels = ["Area (um^2)", "Roundness"]
    data_labels = ["area", "roundness"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True)

def plot_P0_second_neighbors_by_type():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\Events statistics\\"
    x_labels = ["Near differentiations",
                "reference SC initial", "reference SC 24h", "reference SC 48h"]
    data_files_list = ["second_neighbors_by_type_differentiation_data"]
    reference_files_list = ["second_neighbors_by_type_reference_SC_frame1_data",
                            "second_neighbors_by_type_reference_SC_frame96_data",
                            "second_neighbors_by_type_reference_SC_frame165_data" ]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1, 2), (1, 3), (2, 3), (0, 3)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC second neighbors", "HC second neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False)

def compare_distance_from_ablation():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\"
    data_files_list = ["distance_from_ablation_differentiation_data"]
    reference_files_list = ["distance_from_ablation_reference_SC_frame1_data"]
    x_labels = ["Differentiating SCs", "All SCs"]
    normalization_list = [10]
    pairs_to_compare = [(0, 1)]
    y_labels = ["Distance from nearest\n ablation (microns)"]
    data_labels = ["Distance from ablation"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color="blue", edge_color="blue")

def compare_normal_and_promoted_differentiation():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\"
    x_labels = ["Promoted differentiations", "Normal differentiations"]
    pairs_to_compare = [(0, 1)]

    data_files_list = ["HC_density_and_fraction_normal_differentiation_data"]
    reference_files_list = ["HC_density_and_fraction_promoted_differentiation_data"]
    normalization_list = [0.01, 1]
    y_labels = ["HC density (#HCs/micron^2)", "HC fraction (#HCs/#cells)"]
    data_labels = ["HC density", "HC type_fraction"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color=["red", "purple"], edge_color=["red", "purple"])

def compare_deformability():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\"
    hc_before = pd.read_pickle(os.path.join(folder, "hc_before"))
    hc_before.at[24, "label"] = 1269.
    hc_after = pd.read_pickle(os.path.join(folder, "hc_after"))
    sc_before = pd.read_pickle(os.path.join(folder, "sc_before"))
    sc_after = pd.read_pickle(os.path.join(folder, "sc_after"))
    hc_before.sort_values("label", axis=0, ascending=True, inplace=True)
    hc_after.sort_values("label", axis=0, ascending=True, inplace=True)
    sc_before.sort_values("label", axis=0, ascending=True, inplace=True)
    hc_after.sort_values("label", axis=0, ascending=True, inplace=True)
    sc_area_before = sc_before.area.to_numpy()
    sc_perimeter_before = sc_before.perimeter.to_numpy()
    hc_area_before = hc_before.area.to_numpy()
    hc_perimeter_before = hc_before.perimeter.to_numpy()
    sc_area_after = sc_after.area.to_numpy()
    sc_perimeter_after = sc_after.perimeter.to_numpy()
    hc_area_after = hc_after.area.to_numpy()
    hc_perimeter_after = hc_after.perimeter.to_numpy()


    hc_area_diff = hc_area_after/hc_area_before
    hc_perimeter_diff = hc_perimeter_after/hc_perimeter_before
    sc_area_diff = sc_area_after/sc_area_before
    sc_perimeter_diff = sc_perimeter_after /sc_perimeter_before

    font = {'family': 'sans',
            'size': 40}
    import matplotlib
    matplotlib.rc('font', **font)
    from statistical_analysis import compare_and_plot_samples
    samples = [hc_area_diff, sc_area_diff]
    style = "violin"
    x_labels = ["HC area", "SC area"]
    pairs_to_compare = [(0,1)]
    color = ["red", "green"]
    fig, ax, res = compare_and_plot_samples(samples, x_labels, pairs_to_compare, continues=True,
                                            plot_style=style, color=color, edge_color=color)
    ax.set_ylabel("Fold change")
    print(res)
    plt.show()

def plot_number_of_events():
    data = {"E17.5": {"division": 24, "delamination":30, "differentiation":32, "frames":173, "area":5661947},
            "P0": {"division": 0, "delamination":0, "differentiation":14, "frames":121, "area":2826552},
            "P0 with ablation": {"division": 2, "delamination":20, "differentiation":25, "frames":168, "area":5942121}}

    x = np.arange(3)  # the label locations
    width = 0.35  # the width of the bars

    font_size = 25
    plt.rcParams.update({'font.size': font_size})
    fig, ax = plt.subplots()
    colors = ["red", "blue", "green"]
    w = width / 2
    color_index = 0
    rects = []
    for key, value in data.items():
        y = np.array([value[type] for type in ["division", "delamination", "differentiation"]])/(value["area"]*4)
        rects.append(ax.bar(x - w, y, width/3, label=key, alpha=0.5, ecolor=colors[color_index]))
        w -= width/3
        color_index += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Events per micron^2 per hour", fontsize=font_size)
    ax.set_xticks(x, ["division", "delamination", "differentiation"], fontsize=font_size)
    ax.legend(loc="upper left")


    fig.tight_layout()

    plt.show()



if __name__ == "__main__":
    # fit_circular_ablation_results(ellipse_folder, 60)
    # combine_frame_compare_results()
    # combine_single_cell_results(folder, 1000, 2800, 1900)
    # plot_E17_HC_density_and_fraction()
    # plot_E17_neighbors_by_type()
    # plot_E17_number_of_neighbors()
    # plot_E17_area_and_roundness()
    # plot_E17_contact_length_by_type()
    # plot_P0_HC_density_and_fraction()
    # plot_P0_neighbors_by_type()
    # plot_P0_number_of_neighbors()
    # plot_P0_area_and_roundness()
    # plot_P0_contact_length_by_type()
    # compare_E17_P0_density_and_fraction()
    # compare_E17_P0_neighbors_by_type()
    # compare_E17_P0_number_of_neighbors()
    # compare_E17_P0_area_and_roundness()
    # compare_E17_P0_contact_length()

    # compare_distance_from_ablation()
    # compare_normal_and_promoted_differentiation()
    # plot_number_of_events()
    # compare_deformability()
    # plot_E17_second_neighbors_by_type()
    # plot_P0_second_neighbors_by_type()
    # plot_E17_rho_inhibition_neighbors_by_type()
    compare_E17_P0_rho_inhibition_neighbors_by_type()