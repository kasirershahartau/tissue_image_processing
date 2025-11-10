import shutil
import pathlib
import tempfile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os, sys
import subprocess
import seaborn as sns
from tensorflow.python.ops.numpy_ops import append

from statistical_analysis import compare_and_plot_samples

# Change before running
E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\position3-analysis\\",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-23_E17.5_utricle_atoh_zo\\position4-analysis",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\position4-analysis\\"]
P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\",
              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position2-analysis\\",
              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2024-08-04_P0 Utricle_Zo1GFP_atoh1 mCherry\\position2-analysis\\"]
Rho_inhibition_E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-12-01_E17.5_utricle_rho_inhibition\\position2_event_statistics\\"]
Rho_inhibition_P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2023-06-25_P0_atoh_zo_rock_inhibitor\\position3_event_statistics\\"]
E17_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-20-E17.5_ablation\\position3-analysis\\"]
P0_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics"]
E17_DAPT_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-06-E17.5_DAPT\\position2-results"]
P0_DAPT_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-20_P0_DAPT"]
E19_folders = ["D:\\Kasirer\\experimental_results\\fixed\Vestibule\\2025-03-30_E19.5_utricle\\position1-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\Vestibule\\2025-03-30_E19.5_utricle\\position2-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\Vestibule\\2025-04-06_E19.5_utricle\\position1-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\Vestibule\\2025-04-06_E19.5_utricle\\position2-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\Vestibule\\2025-04-06_E19.5_utricle\\position3-analysis"]
E17_circular_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle3",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-11_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-11_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-11_E17.5_utricle_circular_ablation\\utricle3"]
P0_circular_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-07-27_P0_circular_ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-07-27_P0_circular_ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_circular_ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_circular_ablation\\utricle2",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_circular_ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-14_P0_utricle_circular_ablation\\utricle1"]
class DataCollector:
    def __init__(self, name, folders=[], file_names=[], data_labels=[], normalization=1, sample=None):
        self.name = name
        if sample is not None:
            self.sample = sample/normalization
        else:
            self.files = [os.path.join(folder, file_name) for folder, file_name in zip(folders, file_names)]
            self.labels = data_labels
            self.normalization = normalization
            self.initial_batch_indices = None
            self.sample = self.collect()

    def collect(self):
        s = np.empty(0)
        self.initial_batch_indices = []
        for f, l in zip(self.files, self.labels):
            all_data = pd.read_pickle(f)
            relevant_data = all_data[l]
            self.initial_batch_indices.append(s.size)
            s = np.hstack((s, relevant_data.to_numpy() / self.normalization))
        self.initial_batch_indices = np.array(self.initial_batch_indices)
        return s[~np.isnan(s)]

    def get_name(self):
        return self.name

    def get_sample(self):
        return self.sample

    def get_partial_sample(self, file_index):
        start = self.initial_batch_indices[file_index]
        end = self.sample.size if file_index + 1 >= self.initial_batch_indices.size  else self.initial_batch_indices[file_index + 1]
        return self.sample[start:end]

    def get_number_of_data_points(self):
        return self.sample.size

    def get_average(self):
        return np.average(self.sample)

    def get_number_of_biological_repeats(self):
        return len(self.files)

    def get_std(self):
        return np.std(self.sample)

    def get_se(self):
        return self.get_std() / np.sqrt(self.get_number_of_data_points())

    def get_max(self):
        return np.max(self.sample)

    def get_min(self):
        return np.min(self.sample)



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
    E17_HC_data = pd.read_pickle(os.path.join(E17_folders, HC_file))
    E17_SC_data = pd.read_pickle(os.path.join(E17_folders, SC_file))
    P0_HC_data = pd.read_pickle(os.path.join(E17_folders, HC_file))
    P0_SC_data = pd.read_pickle(os.path.join(E17_folders, SC_file))

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    compare_event_statistics([E17_folders, P0_folders], [SC_file, HC_file], [SC_file, HC_file], labels, [(0,1), (2,3)],
                             [1],["roundness"], ["Roundness"], continues=True, color=["red", "green", "red", "green"],
                             edge_color=["red", "green", "red", "green"])

from scipy.optimize import curve_fit

ellipse_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-24_E17.5_circular_ablation",
                   "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-20_P0_circular_ablation\\60um"]
# ellipse_folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-06-P0_circular_ablation\\50um"
# ellipse_folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-05-13_P0_circular_ablation"
def fit_circular_ablation_results_to_ellipse(folder, initial_radius):
    font_size = 40
    major_axis_data = pd.read_pickle(os.path.join(folder,"inner_ellipse_major_axis_data"))
    minor_axis_data = pd.read_pickle(os.path.join(folder, "inner_ellipse_minor_axis_data"))
    eccentricity_data = pd.read_pickle(os.path.join(folder, "inner_ellipse_eccentricity_data"))

    eccentricity = eccentricity_data["inner ellipse:eccentricity average"]
    eccentricity_err = eccentricity_data["inner ellipse:eccentricity se"]
    major = major_axis_data["inner ellipse:semi-major average"]*0.1
    major_err = major_axis_data["inner ellipse:semi-major se"]*0.1
    minor = minor_axis_data["inner ellipse:semi-minor average"]*0.1
    minor_err = minor_axis_data["inner ellipse:semi-minor se"]*0.1
    time = np.arange(start=0, stop=5*(eccentricity.size+1), step=5)
    time_fit = np.linspace(0, 5*(eccentricity.size+1), 300)
    minor = np.hstack([initial_radius, minor])
    major = np.hstack([initial_radius, major])
    eccentricity = np.hstack([0, eccentricity])
    minor_err = np.hstack([0.1, minor_err])
    major_err = np.hstack([0.1, major_err])
    eccentricity_err = np.hstack([0.001, eccentricity_err])

    popt_major, pcov_major = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a, time, major, p0=[45, 0],
                                       sigma=major_err)
    popt_minor, pcov_minor = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a, time, minor, p0=[45, 0],
                                       sigma=minor_err)
    popt_eccentricity, pcov_eccentricity = curve_fit(lambda t, a, b:  a * (1 - np.exp(-b * t)) , time, eccentricity, p0=[0.075, 0],
                                       sigma=eccentricity_err)
    plt.rcParams.update({'font.size': font_size})
    fig1, ax1 = plt.subplots(figsize=(10,10))

    ax1.errorbar(time, major, yerr=major_err, fmt="*", markersize=30, label="Data", linewidth=6)
    ax1.plot(time_fit, (initial_radius - popt_major[0]) * np.exp(-popt_major[1] * time_fit) + popt_major[0], label="Fit", linewidth=6)
    ax1.set_xlabel("Time (minutes)", fontsize=font_size)
    ax1.set_ylabel("Major axis (microns)", fontsize=font_size)
    ax1.legend(loc="upper left")
    print("Major axis results: constant=%f+-%f exponent=%f+-%f" % (popt_major[0], np.sqrt(pcov_major[0,0]),
                                                                   popt_major[1], np.sqrt(pcov_major[1,1])))
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

def fit_circular_ablation_results_to_circle(E17_folders, P0_folders, initial_radius):
    font_size = 40
    plt.rcParams.update({'font.size': font_size})
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    colors = ["blue", "red"]
    labels = ["E17.5", "P0"]
    radii_avg = []
    radii_se = []
    stresses = []
    stresses_err = []
    times = []
    fit_times = []
    for (list_index, folder_list) in enumerate([E17_folders, P0_folders]):
        if not folder_list:
            continue
        radii_data = [pd.read_pickle(os.path.join(folder,"inner_circle_radius_data")) for folder in folder_list]
        radii = [radius_data["inner circle:radius average"].values*0.1983 for radius_data in radii_data]
        radii_err = [radius_data["inner circle:radius se"].values*0.1983 for radius_data in radii_data]
        max_size = np.max([radius.size for radius in radii])
        time = np.arange(start=1, stop=max_size+1, step=1)
        time = np.hstack([0, time])
        times.append(time)
        time_fit = np.linspace(0, max_size+1, 300)
        fit_times.append(time_fit)
        radii = [np.hstack([initial_radius, radius]) for radius in radii]
        radii_err = [np.hstack([0.1, radius_err]) for radius_err in radii_err]
        stresses.append([])
        stresses_err.append([])
        # Shorten long lists
        min_len = min(len(sublist) for sublist in radii)
        truncated_radii = np.array([sublist[:min_len] for sublist in radii])
        truncated_radii_err = np.array([sublist[:min_len] for sublist in radii_err])
        for radius, radius_err in zip(truncated_radii, truncated_radii_err):
            popt_radius, pcov_radius = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a,
                                                 time[:radius.size], radius,
                                                 p0=[initial_radius * 0.8, 0], sigma=radius_err)
            final_radius = popt_radius[0]
            final_radius_err = np.sqrt(pcov_radius[0, 0])
            young_over_visc = popt_radius[1]
            young_over_visc_err = np.sqrt(pcov_radius[1, 1])
            print("Final_radius=%f+-%f young's_modulus_over_viscosity=%f+-%f" % (final_radius, final_radius_err,
                                                                                 young_over_visc, young_over_visc_err))
            stress = (initial_radius / final_radius - 1) * 4 * young_over_visc
            stress_error = np.sqrt(
                ((-initial_radius / (final_radius ** 2)) * 4 * young_over_visc * final_radius_err) ** 2 +
                ((initial_radius / final_radius - 1) * 4 * young_over_visc_err) ** 2)
            ax1.plot(time_fit, (initial_radius - final_radius) * np.exp(-young_over_visc * time_fit) + final_radius,
                     color=colors[list_index], linewidth=3)
            stresses[list_index].append(stress)
            stresses_err[list_index].append(stress_error)
            print("Bulk_stress_over_viscosity=%f+-%f" % (stress, stress_error))

        radii_avg.append(np.average(truncated_radii, axis=0))
        radius_std = np.std(truncated_radii, axis=0)
        radius_std[0] = 0.1
        if truncated_radii.shape[0] > 1:
            radii_se.append(radius_std/np.sqrt(truncated_radii.shape[0]))
        else:
            radii_se.append(radii_err[0])
        ax1.errorbar(time[:radii_avg[list_index].size], radii_avg[list_index], yerr=radii_se[list_index], fmt="*", markersize=30,
                     label="%s Data" %labels[list_index], linewidth=2, color=colors[list_index])
    # Calculating bulk stress according to 10.7554/eLife.57964
    # calculated stress/viscosity in units of 1/time-unit
    index = 0
    for radius, radius_err in zip(radii_avg, radii_se):
        popt_radius, pcov_radius = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a,
                                             times[index][:radius.size], radius,
                                             p0=[initial_radius * 0.8, 0], sigma=radius_err)
        final_radius = popt_radius[0]
        young_over_visc = popt_radius[1]
        ax1.plot(time_fit, (initial_radius - final_radius) * np.exp(-young_over_visc * fit_times[index]) + final_radius,
                 label="%s Fit" % labels[index], linewidth=6, color=colors[index])
        index += 1
    ax1.set_xlabel("Time (minutes)", fontsize=font_size)
    ax1.set_ylabel("Radius (microns)", fontsize=font_size)
    ax1.legend(loc="upper right")
    edge_color = ["blue", "red"]
    E17_stress = DataCollector("E17 Stress", sample=np.array(stresses[0]))
    P0_stress = DataCollector("P0 Stress", sample=np.array(stresses[1]))
    fig2, ax2, res = compare_and_plot_samples([E17_stress, P0_stress], [(0, 1)], continues=True,
                                            plot_style="bar", color=edge_color, edge_color=edge_color,
                                            show_statistics=True, show_N=True)

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

def load_data(folder_list, data_files_list, reference_files_list, data_labels, normalization_list):
    if isinstance(folder_list, tuple):
        folder = folder_list[0]
        ref_folder = folder_list[1]
    else:
        folder = folder_list
        ref_folder = folder

    for data_label, normalization in zip(data_labels, normalization_list):
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
        yield samples

def compare_event_statistics(folder_list, data_files_list, reference_files_list, x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues, color='white', edge_color='grey',
                             show_statistics=False, show_N=False):
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
    for samples, y_label in zip(load_data(folder_list, data_files_list,
                                          reference_files_list, data_labels, normalization_list),
                                y_labels):

        style = "violin" if continues else "bar"
        fig, ax, res = compare_and_plot_samples(samples, x_labels, pairs_to_compare, continues=continues,
                                                plot_style=style, color=color, edge_color=edge_color,
                                                show_statistics=show_statistics, show_N=show_N)
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
                       "HC_density_and_fraction_differentiation_data"],
                       ["HC_density_and_fraction_delamination_data", "HC_density_and_fraction_division_data",
                        "HC_density_and_fraction_differentiation_data"]]
    reference_files_list = [["HC_fraction_and_density_overall_SC_frame1_data",
                            "HC_fraction_and_density_overall_SC_frame96_data",
                            "HC_fraction_and_density_overall_SC_frame191_data"],
                            ["HC_density_and_fraction_reference_SC_frame1_data",
                            "HC_density_and_fraction_reference_SC_frame97_data"],
                            ["HC_density_and_fraction_reference_SC_frame1_data",
                             "HC_density_and_fraction_reference_SC_frame96_data"]
                            ]
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
                        "neighbors_by_type_differentiation_data"],
                       ["neighbors_by_type_delamination_data", "neighbors_by_type_division_data",
                        "neighbors_by_type_differentiation_data"]
                       ]
    reference_files_list = [["neighbors_by_type_reference_SC_frame96_data"],
                            ["neighbors_by_type_reference_SC_frame97_data"],
                            ["neighbors_by_type_reference_SC_frame96_data"]]
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
                        "second_neighbors_by_type_differentiation_data"],
                       ["second_neighbors_by_type_delamination_data", "second_neighbors_by_type_division_data",
                        "second_neighbors_by_type_differentiation_data"]
                       ]
    reference_files_list = [["second_neighbors_by_type_reference_SC_frame1_data",
                             "second_neighbors_by_type_reference_SC_frame96_data",
                             "second_neighbors_by_type_reference_SC_frame191_data"],
                            ["second_neighbors_by_type_reference_SC_frame1_data",
                             "second_neighbors_by_type_reference_SC_frame96_data",
                             "second_neighbors_by_type_reference_SC_frame191_data"],
                            ["second_neighbors_by_type_reference_SC_frame1_data",
                             "second_neighbors_by_type_reference_SC_frame96_data"]
                            ]
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
                        "contach_length_differentiation_data"],
                       ["contach_length_delamination_data", "contach_length_division_data",
                        "contach_length_differentiation_data"]
                       ]
    reference_files_list = [["contact_length_by_type_reference_SC_frame95_data"],
                            ["contach_length_reference_SC_frame97_data"],
                            ["contach_length_reference_SC_frame96_data"]]
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
                        "number_of_neighbors_differentiation_data"],
                       ["number_of_neighbors_delamination_data", "number_of_neighbors_division_data",
                        "number_of_neighbors_differentiation_data"]
                       ]
    reference_files_list = [["number_of_neighbors_reference_SC_frame96_data"],
                            ["number_of_neighbors_reference_SC_frame97_data"],
                            ["number_of_neighbors_reference_SC_frame96_data"]]
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
                        "area_and_roundness_differentiation_data"],
                       ["area_and_roundness_delamination_data", "area_and_roundness_division_data",
                        "area_and_roundness_differentiation_data"]
                       ]
    reference_files_list = [["area_and_roundness_reference_SC_frame96_data"],
                            ["area_and_roundness_reference_SC_frame97_data"],
                            ["area_and_roundness_reference_SC_frame96_data"]]
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
    x_labels = ["E17.5", "E17.5", "+24h", "+24h", "+48h","+48h",  # "Diff.\ncells",
                "P0", "P0", "+24h", "+24h", "+48h", "+48h"]  # "Diff.\ncells"]
    x_labels = [""]*12
    E17_data_files_list = [["neighbors_by_type_reference_SC_frame1_data",
                            "neighbors_by_type_reference_HC_frame1_data",
                            "neighbors_by_type_reference_SC_frame96_data",
                            "neighbors_by_type_reference_HC_frame96_data",
                            "neighbors_by_type_reference_SC_frame191_data",
                            "neighbors_by_type_reference_HC_frame191_data",
                            ],
                           ["neighbors_by_type_reference_SC_frame1_data",
                            "neighbors_by_type_reference_HC_frame1_data",
                            "neighbors_by_type_reference_SC_frame97_data",
                            "neighbors_by_type_reference_HC_frame97_data",
                            "neighbors_by_type_reference_SC_frame199_data",
                            "neighbors_by_type_reference_HC_frame199_data",
                            ],
                           ["neighbors_by_type_reference_SC_frame1_data",
                            "neighbors_by_type_reference_HC_frame1_data",
                            "neighbors_by_type_reference_SC_frame96_data",
                            "neighbors_by_type_reference_HC_frame96_data",
                            ]
                           ]
    P0_files_list = [["neighbors_by_type_reference_SC_frame1_data",
                      "neighbors_by_type_reference_HC_frame1_data",
                      "neighbors_by_type_reference_SC_frame96_data",
                      "neighbors_by_type_reference_HC_frame96_data",
                      "neighbors_by_type_reference_SC_frame165_data",
                      "neighbors_by_type_reference_HC_frame165_data",
                      ],
                      ["neighbors_by_type_reference_SC_frame1_data",
                       "neighbors_by_type_reference_HC_frame1_data",
                       "neighbors_by_type_reference_SC_frame96_data",
                       "neighbors_by_type_reference_HC_frame96_data",
                       ]]
    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (2,3), (4,5), (6,7), (8,9), (10,11), (0,2), (2, 4), (4,6), (6,8), (8,10),
                        (1,3), (3,5), (5,7), (7,9), (9,11)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    color = ["mediumpurple", "white"] * 3 + ["lightpink", "white"] * 3
    edge_color = ["mediumpurple", "mediumpurple"] * 3+ ["lightpink", "lightpink"] * 3

    compare_event_statistics((E17_folder,P0_folder), E17_data_files_list, P0_files_list , x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color)

    SC_with_Zero_HC_neighbors = np.zeros((5,2))
    for experiment_idx in range(2):
        for samples in load_data((E17_folder[experiment_idx],P0_folder[experiment_idx]),
                                 E17_data_files_list[experiment_idx],P0_files_list[experiment_idx],
                                 data_labels, normalization_list):
            for i in range(SC_with_Zero_HC_neighbors.shape[0]):
                SC_with_Zero_HC_neighbors[i, experiment_idx] = 100*np.sum((samples[2*i] == 0).astype(int))/samples[2*i].size
    fig, ax = plt.subplots()

    color = ["mediumpurple"] * 3 + ["lightpink"] * 2
    edge_color = ["mediumpurple"] * 3 + ["lightpink"] * 2
    compare_and_plot_samples(list(SC_with_Zero_HC_neighbors), [""]*5, [], continues=True, plot_style="box", color=color,
    edge_color=edge_color, fig=fig, ax=ax, show_statistics=False, show_N=False)
    plt.show()

def compare_E17_P0_HC_neighbors_for_differentiation_and_trans_differentiation():
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["neighbors_by_type_differentiation_data"]*3,
                             ["HC neighbors"]*3)
    E17_trans_diff = DataCollector("E17.5 trans-differentiating cells",
                            E17_ablation_folders,
                             ["promoted_differentiation_neighbors_by_type_data"],
                             ["HC neighbors"])
    E17_trans_ref_SC = DataCollector("E17.5 trans-differentiating reference_SC",
                                   E17_ablation_folders,
                                   ["reference_SC_frame_96_data"],
                                   ["HC neighbors"])
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", E17_folders,
                               ["neighbors_by_type_reference_SC_frame96_data",
                                 "neighbors_by_type_reference_SC_frame97_data",
                                "neighbors_by_type_reference_SC_frame96_data"],
                             ["HC neighbors"] * 3)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                             ["neighbors_by_type_differentiation_data"] * 3,
                             ["HC neighbors"] * 3)
    P0_trans_diff = DataCollector("P0 trans-differentiating cells",
    P0_ablation_folders,
                             ["neighbors_by_type_promoted_differentiation_data"],
                             ["HC neighbors"])
    P0_trans_ref_SC = DataCollector("P0 trans-differentiating reference SC",
                                  P0_ablation_folders,
                                  ["neighbors_by_type_reference_SC_frame96_data"],
                                  ["HC neighbors"])
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["neighbors_by_type_reference_SC_frame96_data"]*3,
                             ["HC neighbors"] * 3)
    samples_list = [E17_diff, E17_trans_diff, E17_ref_SC, P0_diff, P0_trans_diff, P0_ref_SC]
    pairs_to_compare = [(0,3),(1,4), (0,1), (3,4), (0,2), (3,5)]
    # samples_list = [E17_diff, E17_ref_SC,P0_diff, P0_ref_SC]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="histogram", color= ["cyan"] * 3 + ["pink"] *3, edge_color=["blue"] * 3 + ["red"] * 3,
                                            show_statistics=True, show_N=True, hirarchical=True)
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=["cyan"] * 3 + ["pink"] * 3,
                                                      edge_color=["blue"] * 3 + ["red"] * 3,
                                                      show_statistics=False, show_N=False)
    plt.show()

    E17_rev_hist = [None]*3
    P0_rev_hist = [None]*3
    for i in range(3):
        E17_diff_hist, E17_diff_bin_edges = np.histogram(E17_diff.get_partial_sample(i), bins=(np.arange(np.max(E17_diff.get_partial_sample(i)) + 2) - 0.5))
        E17_ref_hist, E17_ref_bin_edges = np.histogram(E17_ref_SC.get_partial_sample(i),
                                                    bins=(np.arange(np.max(E17_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        E17_rev_hist[i] = E17_diff_hist/E17_ref_hist[:E17_diff_hist.size]
    for i in range(3):
        P0_diff_hist, P0_diff_bin_edges = np.histogram(P0_diff.get_partial_sample(i),
                                                     bins=(np.arange(np.max(P0_diff.get_partial_sample(i)) + 2) - 0.5))
        P0_ref_hist, P0_ref_bin_edges = np.histogram(P0_ref_SC.get_partial_sample(i),
                                                   bins=(np.arange(np.max(P0_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        P0_rev_hist[i] = P0_diff_hist / P0_ref_hist[:P0_diff_hist.size]
    from itertools import zip_longest
    E17_rev_hist = np.array(list(zip_longest(*E17_rev_hist, fillvalue=0))).T
    P0_rev_hist = np.array(list(zip_longest(*P0_rev_hist, fillvalue=0))).T
    E17_rev_averages = np.average(E17_rev_hist, axis=0)
    P0_rev_averages = np.average(P0_rev_hist, axis=0)
    plt.bar(np.arange(E17_rev_averages.size)-0.125, 100*E17_rev_averages, width=0.25, color="cyan", edgecolor="blue")
    plt.bar(np.arange(P0_rev_averages.size)+0.125, 100*P0_rev_averages, width=0.25, color="pink", edgecolor="red")
    plt.legend(["E17.5", "P0"], prop={'size': 20})
    for i in range(E17_rev_hist.shape[0]):
        plt.scatter(np.arange(E17_rev_averages.size) - 0.125, 100 * E17_rev_hist[i], marker="*", color="black")
    for i in range(P0_rev_hist.shape[0]):
        plt.scatter(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_hist[i], marker="*", color="black")
    from statistical_analysis import TwoSampleCompare
    for n_neighbors in range(len(E17_rev_hist[0])):
        sample1 = np.array([E17_rev_hist[i][n_neighbors] for i in range(3)])
        sample2 = np.array([P0_rev_hist[i][n_neighbors] for i in range(3)])
        comparer = TwoSampleCompare(sample1, sample2, sample1_label='E17.5', sample2_label='P0', continues=True)
        pval = comparer.compare_samples()
        print(sample1)
        print(sample2)
        print("pval for %d HC neighbors = %f" %(n_neighbors, pval))
    plt.xlabel("#HC neighbors", fontsize=20)
    plt.ylabel("% differentiating SCs", fontsize=20)
    plt.ylim([0, 30])
    plt.xticks(np.arange(P0_rev_hist.size), labels=np.arange(P0_rev_hist.size), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([-0.5, 2.5])
    plt.tight_layout()

    plt.figure()
    E17_trans_diff_hist, E17_trans_diff_bin_edges = np.histogram(E17_trans_diff.get_sample(),
                                                     bins=(np.arange(np.max(E17_trans_diff.get_sample()) + 2) - 0.5))
    E17_trans_ref_hist, E17_trans_ref_bin_edges = np.histogram(E17_trans_ref_SC.get_sample(),
                                                   bins=(np.arange(np.max(E17_trans_ref_SC.get_sample()) + 2) - 0.5))
    E17_trans_rev_hist = E17_trans_diff_hist / E17_trans_ref_hist[:E17_trans_diff_hist.size]
    P0_trans_diff_hist, P0_trans_diff_bin_edges = np.histogram(P0_trans_diff.get_sample(),
                                                   bins=(np.arange(np.max(P0_trans_diff.get_sample()) + 2) - 0.5))
    P0_trans_ref_hist, P0_trans_ref_bin_edges = np.histogram(P0_trans_ref_SC.get_sample(),
                                                 bins=(np.arange(np.max(P0_trans_ref_SC.get_sample()) + 2) - 0.5))
    P0_trans_rev_hist = P0_trans_diff_hist / P0_trans_ref_hist[:P0_trans_diff_hist.size]
    plt.bar(np.arange(E17_trans_rev_hist.size)-0.125, 100*E17_trans_rev_hist, width=0.25, color="cyan", edgecolor="blue")
    plt.bar(np.arange(P0_trans_rev_hist.size) + 0.125, 100*P0_trans_rev_hist, width=0.25, color="pink", edgecolor="red")
    plt.legend(["E17.5", "P0"], prop={'size': 20})
    plt.xlabel("#HC neighbors", fontsize=20)
    plt.ylabel("% trans-differentiating SCs", fontsize=20)
    plt.ylim([0, 25])
    plt.xticks(np.arange(P0_trans_rev_hist.size), labels=np.arange(P0_trans_rev_hist.size), fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()





    # no_SC_samples_list = [E17_diff, E17_trans_diff, P0_diff, P0_trans_diff]
    # no_SC_pairs_to_compare = [(0, 1), (0, 2), (2, 3)]
    # no_SC_fig, no_SC_ax, res = compare_and_plot_samples(no_SC_samples_list, no_SC_pairs_to_compare, continues=False,
    #                                                   plot_style="bar", color=["white"] * 4,
    #                                                   edge_color=["blue"] * 2 + ["red"] * 2,
    #                                                   show_statistics=True, show_N=True)
    # empty_no_SC_fig, empty_no_SC_ax, res = compare_and_plot_samples(no_SC_samples_list, no_SC_pairs_to_compare, continues=False,
    #                                                     plot_style="bar", color=["white"] * 4,
    #                                                     edge_color=["blue"] * 2 + ["red"] * 2,
    #                                                     show_statistics=False, show_N=False)
    #
    # no_transdiff_samples_list = [E17_diff, E17_ref_SC, P0_diff, P0_ref_SC]
    # no_transdiff_pairs_to_compare = [(0, 1), (0, 2), (2, 3)]
    # no_transdiff_fig, no_transdiff_ax, res = compare_and_plot_samples(no_transdiff_samples_list, no_transdiff_pairs_to_compare, continues=False,
    #                                                     plot_style="bar", color=["white"] * 4,
    #                                                     edge_color=["blue"] * 2 + ["red"] * 2,
    #                                                     show_statistics=True, show_N=True)
    # empty_no_transdiff_fig, empty_no_transdiff_ax, res = compare_and_plot_samples(no_transdiff_samples_list, no_transdiff_pairs_to_compare,
    #                                                                 continues=False,
    #                                                                 plot_style="bar", color=["white"] * 4,
    #                                                                 edge_color=["blue"] * 2 + ["red"] * 2,
    #                                                                 show_statistics=False, show_N=False)
    plt.show()

def compare_E17_P0_HC_contact_length_for_differentiation_and_trans_differentiation():
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["contact_length_by_type_differentiation_data"]*3,
                             ["HC contact length"]*3, normalization=10)
    E17_trans_diff = DataCollector("E17.5 trans-differentiating cells", E17_ablation_folders,
                             ["contact_length_by_type_promoted_differentiation_data"],
                             ["HC contact length"], normalization=10)
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", E17_folders,
                               ["contact_length_by_type_reference_SC_frame96_data",
                                          "contact_length_by_type_reference_SC_frame97_data",
                                          "contact_length_by_type_reference_SC_frame96_data"],
                             ["HC contact length"] * 3, normalization=10)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                             ["contact_length_by_type_differentiation_data"] * 3,
                             ["HC contact length"] * 3, normalization=10)
    P0_trans_diff = DataCollector("P0 trans-differentiating cells", P0_ablation_folders,
                             ["contact_length_by_type_promoted_differentiation_data"],
                             ["HC contact length"], normalization=10)
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["contact_length_by_type_reference_SC_frame96_data"]*3,
                             ["HC contact length"] * 3, normalization=10)
    # E17_trans_ref_SC = DataCollector("E17.5 trans-differentiating reference_SC",
    #                                  [
    #                                      "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-20-E17.5_ablation\\position3-analysis\\"],
    #                                  ["reference_SC_frame_96_data"],
    #                                  ["HC contact length"])
    # P0_trans_ref_SC = DataCollector("P0 trans-differentiating reference SC",
    #                                 [
    #                                     "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics"],
    #                                 ["neighbors_by_type_reference_SC_frame96_data"],
    #                                 ["HC contact length"])
    samples_list = [E17_diff, E17_trans_diff, E17_ref_SC, P0_diff, P0_trans_diff, P0_ref_SC]
    pairs_to_compare = [(0,3)]
    # samples_list = [E17_diff, E17_ref_SC,P0_diff, P0_ref_SC]
    # pairs_to_compare = [(0, 1)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                            plot_style="violin", color= ["cyan"] * 3 + ["pink"] *3, edge_color=["blue"] * 3 + ["red"] * 3,
                                            show_statistics=True, show_N=True, hirarchical=True)

    full_ax.set_ylabel("Apical contact length with neighboring HCs (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=["cyan"] * 3 + ["pink"] * 3,
                                                      edge_color=["blue"] * 3 + ["red"] * 3,
                                                      show_statistics=False, show_N=False)


    plt.show()

    E17_rev_hist = [None] * 3
    P0_rev_hist = [None] * 3
    for i in range(3):
        E17_diff_hist, E17_diff_bin_edges = np.histogram(E17_diff.get_partial_sample(i), bins=(
                    np.arange(np.max(E17_diff.get_partial_sample(i)) + 2) - 0.5))
        E17_ref_hist, E17_ref_bin_edges = np.histogram(E17_ref_SC.get_partial_sample(i),
                                                       bins=(np.arange(
                                                           np.max(E17_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        E17_rev_hist[i] = E17_diff_hist[:E17_ref_hist.size] / E17_ref_hist[:E17_diff_hist.size]
    for i in range(3):
        P0_diff_hist, P0_diff_bin_edges = np.histogram(P0_diff.get_partial_sample(i),
                                                       bins=(np.arange(
                                                           np.max(P0_diff.get_partial_sample(i)) + 2) - 0.5))
        P0_ref_hist, P0_ref_bin_edges = np.histogram(P0_ref_SC.get_partial_sample(i),
                                                     bins=(np.arange(
                                                         np.max(P0_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        P0_rev_hist[i] = P0_diff_hist[:P0_ref_hist.size] / P0_ref_hist[:P0_diff_hist.size]
    from itertools import zip_longest
    E17_rev_hist = np.array(list(zip_longest(*E17_rev_hist, fillvalue=0))).T
    P0_rev_hist = np.array(list(zip_longest(*P0_rev_hist, fillvalue=0))).T
    E17_rev_averages = np.average(E17_rev_hist, axis=0)
    P0_rev_averages = np.average(P0_rev_hist, axis=0)
    plt.bar(np.arange(E17_rev_averages.size) - 0.125, 100 * E17_rev_averages, width=0.25, color="cyan",
            edgecolor="blue")
    plt.bar(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_averages, width=0.25, color="pink", edgecolor="red")
    plt.legend(["E17.5", "P0"], prop={'size': 20})
    for i in range(E17_rev_hist.shape[0]):
        plt.scatter(np.arange(E17_rev_averages.size) - 0.125, 100 * E17_rev_hist[i], marker="*", color="black")
    for i in range(P0_rev_hist.shape[0]):
        plt.scatter(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_hist[i], marker="*", color="black")
    plt.xlabel("contact length with HC neighbors", fontsize=20)
    plt.ylabel("% differentiating SCs", fontsize=20)
    plt.xticks(np.arange(P0_rev_hist.size), labels=np.arange(P0_rev_hist.size), fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # plt.figure()
    # E17_trans_diff_hist, E17_trans_diff_bin_edges = np.histogram(E17_trans_diff.get_sample(),
    #                                                              bins=(np.arange(
    #                                                                  np.max(E17_trans_diff.get_sample()) + 2) - 0.5))
    # E17_trans_ref_hist, E17_trans_ref_bin_edges = np.histogram(E17_trans_ref_SC.get_sample(),
    #                                                            bins=(np.arange(
    #                                                                np.max(E17_trans_ref_SC.get_sample()) + 2) - 0.5))
    # E17_trans_rev_hist = E17_trans_diff_hist / E17_trans_ref_hist[:E17_trans_diff_hist.size]
    # P0_trans_diff_hist, P0_trans_diff_bin_edges = np.histogram(P0_trans_diff.get_sample(),
    #                                                            bins=(np.arange(
    #                                                                np.max(P0_trans_diff.get_sample()) + 2) - 0.5))
    # P0_trans_ref_hist, P0_trans_ref_bin_edges = np.histogram(P0_trans_ref_SC.get_sample(),
    #                                                          bins=(np.arange(
    #                                                              np.max(P0_trans_ref_SC.get_sample()) + 2) - 0.5))
    # P0_trans_rev_hist = P0_trans_diff_hist / P0_trans_ref_hist[:P0_trans_diff_hist.size]
    # plt.bar(np.arange(E17_trans_rev_hist.size) - 0.125, 100 * E17_trans_rev_hist, width=0.25, color="cyan",
    #         edgecolor="blue")
    # plt.bar(np.arange(P0_trans_rev_hist.size) + 0.125, 100 * P0_trans_rev_hist, width=0.25, color="pink",
    #         edgecolor="red")
    # plt.legend(["E17.5", "P0"], prop={'size': 20})
    # plt.xlabel("#HC neighbors", fontsize=20)
    # plt.ylabel("% trans-differentiating SCs", fontsize=20)
    # plt.ylim([0, 25])
    # plt.xticks(np.arange(P0_trans_rev_hist.size), labels=np.arange(P0_trans_rev_hist.size), fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.tight_layout()
    plt.show()

def compare_E17_E19_neighbors():
    E17_after_48h_HC_neighbors_for_HC = DataCollector("E17.5 +48h HC neigh for HC", E17_folders[:-1],
                               ["neighbors_by_type_reference_HC_frame191_data",
                                          "neighbors_by_type_reference_HC_frame199_data"],
                                         ["HC neighbors"] * 2)
    E17_after_48h_SC_neighbors_for_HC = DataCollector("E17.5 +48h SC neigh for HC", E17_folders[:-1],
                                                      ["neighbors_by_type_reference_HC_frame191_data",
                                                       "neighbors_by_type_reference_HC_frame199_data"],
                                                      ["SC neighbors"] * 2)
    E17_after_48h_HC_neighbors_for_SC = DataCollector("E17.5 +48h HC neigh for SC", E17_folders[:-1],
                                                      ["neighbors_by_type_reference_SC_frame191_data",
                                                       "neighbors_by_type_reference_SC_frame199_data"],
                                                      ["HC neighbors"] * 2)
    E17_after_48h_SC_neighbors_for_SC = DataCollector("E17.5 +48h SC neigh for SC", E17_folders[:-1],
                                                      ["neighbors_by_type_reference_SC_frame191_data",
                                                       "neighbors_by_type_reference_SC_frame199_data"],
                                                      ["SC neighbors"] * 2)
    E19_HC_neighbors_for_HC = DataCollector("E19.5 HC neigh for HC", E19_folders,
                                            ["neighbors_by_type_for_HC_data"]*len(E19_folders),
                                            ["HC neighbors"]*len(E19_folders))
    E19_SC_neighbors_for_HC = DataCollector("E19.5 SC neigh for HC", E19_folders,
                                            ["neighbors_by_type_for_HC_data"] * len(E19_folders),
                                            ["SC neighbors"] * len(E19_folders))
    E19_HC_neighbors_for_SC = DataCollector("E19.5 HC neigh for SC", E19_folders,
                                            ["neighbors_by_type_for_SC_data"] * len(E19_folders),
                                            ["HC neighbors"] * len(E19_folders))
    E19_SC_neighbors_for_SC = DataCollector("E19.5 SC neigh for SC", E19_folders,
                                            ["neighbors_by_type_for_SC_data"] * len(E19_folders),
                                            ["SC neighbors"] * len(E19_folders))
    samples_list = [E17_after_48h_HC_neighbors_for_HC, E19_HC_neighbors_for_HC,
                    E17_after_48h_HC_neighbors_for_SC, E19_HC_neighbors_for_SC,
                    E17_after_48h_SC_neighbors_for_HC, E19_SC_neighbors_for_HC,
                    E17_after_48h_SC_neighbors_for_SC, E19_SC_neighbors_for_SC]
    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="histogram", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                            show_statistics=True, show_N=True)

    full_ax.set_ylabel("# neighbors")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                                      show_statistics=False, show_N=False)


    plt.show()

def compare_E17_E19_contact_length():
    E17_after_48h_HC_contact_length_for_HC = DataCollector("E17.5 +48h HC contact length for HC", E17_folders[:-1],
                               ["contact_length_by_type_reference_HC_frame191_data",
                                          "contact_length_by_type_reference_HC_frame199_data"],
                                         ["HC contact length"] * 2, normalization=10)
    E17_after_48h_SC_contact_length_for_HC = DataCollector("E17.5 +48h SC contact length for HC", E17_folders[:-1],
                                                      ["contact_length_by_type_reference_HC_frame191_data",
                                                       "contact_length_by_type_reference_HC_frame199_data"],
                                                      ["SC contact length"] * 2, normalization=10)
    E17_after_48h_HC_contact_length_for_SC = DataCollector("E17.5 +48h HC contact length for SC", E17_folders[:-1],
                                                      ["contact_length_by_type_reference_SC_frame191_data",
                                                       "contact_length_by_type_reference_SC_frame199_data"],
                                                      ["HC contact length"] * 2, normalization=10)
    E17_after_48h_SC_contact_length_for_SC = DataCollector("E17.5 +48h SC contact length for SC", E17_folders[:-1],
                                                      ["contact_length_by_type_reference_SC_frame191_data",
                                                       "contact_length_by_type_reference_SC_frame199_data"],
                                                      ["SC contact length"] * 2, normalization=10)
    E19_HC_contact_length_for_HC = DataCollector("E19.5 HC contact length for HC", E19_folders,
                                            ["contact_length_by_type_for_HC_data"]*len(E19_folders),
                                            ["HC contact length"]*len(E19_folders), normalization=10)
    E19_SC_contact_length_for_HC = DataCollector("E19.5 SC contact length for HC", E19_folders,
                                            ["contact_length_by_type_for_HC_data"] * len(E19_folders),
                                            ["SC contact length"] * len(E19_folders), normalization=10)
    E19_HC_contact_length_for_SC = DataCollector("E19.5 HC contact length for HC", E19_folders,
                                            ["contact_length_by_type_for_SC_data"] * len(E19_folders),
                                            ["HC contact length"] * len(E19_folders), normalization=10)
    E19_SC_contact_length_for_SC = DataCollector("E19.5 SC contact length for HC", E19_folders,
                                            ["contact_length_by_type_for_SC_data"] * len(E19_folders),
                                            ["SC contact length"] * len(E19_folders), normalization=10)
    samples_list = [E17_after_48h_HC_contact_length_for_HC, E19_HC_contact_length_for_HC,
                    E17_after_48h_HC_contact_length_for_SC, E19_HC_contact_length_for_SC,
                    E17_after_48h_SC_contact_length_for_HC, E19_SC_contact_length_for_HC,
                    E17_after_48h_SC_contact_length_for_SC, E19_SC_contact_length_for_SC]
    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="violin", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                            show_statistics=True, show_N=True)

    full_ax.set_ylabel("Contact length (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                                      show_statistics=False, show_N=False)


    plt.show()

def compare_E17_E19_area_and_roundness():
    E17_after_48h_HC_area = DataCollector("E17.5 +48h HC area", E17_folders[:-1],
                               ["area_and_roundness_reference_HC_frame191_data",
                                          "area_and_roundness_reference_HC_frame199_data"],
                                         ["area"] * 2, normalization=100)
    E17_after_48h_SC_area = DataCollector("E17.5 +48h SC area", E17_folders[:-1],
                                                      ["area_and_roundness_reference_SC_frame191_data",
                                                       "area_and_roundness_reference_SC_frame199_data"],
                                                      ["area"] * 2, normalization=100)
    E17_after_48h_HC_roundness = DataCollector("E17.5 +48h HC roundness", E17_folders[:-1],
                                                      ["area_and_roundness_reference_HC_frame191_data",
                                                       "area_and_roundness_reference_HC_frame199_data"],
                                                      ["roundness"] * 2, normalization=1)
    E17_after_48h_SC_roundness = DataCollector("E17.5 +48h SC roundness", E17_folders[:-1],
                                                      ["area_and_roundness_reference_SC_frame191_data",
                                                       "area_and_roundness_reference_SC_frame199_data"],
                                                      ["roundness"] * 2, normalization=1)
    E19_HC_area = DataCollector("E19.5 HC area", E19_folders,
                                            ["area_and_roundness_for_HC_data"]*len(E19_folders),
                                            ["area"]*len(E19_folders), normalization=100)
    E19_SC_area = DataCollector("E19.5 SC area", E19_folders,
                                            ["area_and_roundness_for_SC_data"] * len(E19_folders),
                                            ["area"] * len(E19_folders), normalization=100)
    E19_HC_roundness = DataCollector("E19.5 HC roundness", E19_folders,
                                            ["area_and_roundness_for_HC_data"] * len(E19_folders),
                                            ["roundness"] * len(E19_folders), normalization=1)
    E19_SC_roundness = DataCollector("E19.5 SC roundness", E19_folders,
                                            ["area_and_roundness_for_SC_data"] * len(E19_folders),
                                            ["roundness"] * len(E19_folders), normalization=1)
    samples_list = [E17_after_48h_HC_area, E19_HC_area,
                    E17_after_48h_SC_area, E19_SC_area,
                    E17_after_48h_HC_roundness, E19_HC_roundness,
                    E17_after_48h_SC_roundness, E19_SC_roundness]
    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="violin", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                            show_statistics=True, show_N=True)

    full_ax.set_ylabel("Contact length (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color= ["cyan","pink"] *4,
                                                      edge_color=["blue", "red"] * 4,
                                                      show_statistics=False, show_N=False)


    plt.show()

def compare_E17_P0_neighbors_by_type_for_differentiation():
    E17_folder = E17_folders
    P0_folder = P0_folders
    # x_labels = ["Differentiating\ncells",  "All\nSCs", "Differentiating\ncells",
    #             "All\nSCs"]
    x_labels = [""]*4
    E17_data_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                           ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame97_data"]]
                           # ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]
    P0_reference_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                               ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]
                               # ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]

    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1,2), (0,2)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    # color = ["blue", "white", "red", "white"]
    color = ["white"]*4
    edge_color = ["blue", "blue", "red", "red"]

    compare_event_statistics((E17_folder,P0_folder), E17_data_files_list, P0_reference_files_list,
                             x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color,
                             show_statistics=True, show_N=False)

def compare_P0_neighbors_by_type_for_differentiation_and_transdiff():
    P0_folder = P0_folders
    transdiff_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics"]
    # x_labels = ["Differentiating\ncells",  "All\nSCs", "Differentiating\ncells",
    #             "All\nSCs"]
    x_labels = [""]*4
    P0_reference_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                               ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]
    trans_diffreference_files_list = [["neighbors_by_type_promoted_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]

    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (0,2), (1,2)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    color = ["white"]*4
    edge_color = ["red"]*4

    compare_event_statistics((P0_folder, transdiff_folders), P0_reference_files_list, trans_diffreference_files_list,
                             x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color,
                             show_statistics=False, show_N=False)

def compare_E17_neighbors_by_type_for_differentiation_and_transdiff():
    E17_folder = E17_folders
    transdiff_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-20-E17.5_ablation\\position3-analysis\\"]
    # x_labels = ["Differentiating\ncells",  "All\nSCs", "Trans-Differentiating\ncells",
    #             "All\nSCs"]
    x_labels = [""]*4
    E17_reference_files_list = [["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"],
                               ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame97_data"],
                               ["neighbors_by_type_differentiation_data", "neighbors_by_type_reference_SC_frame96_data"]]
    trans_diffreference_files_list = [["promoted_differentiation_neighbors_by_type_data", "reference_SC_frame_96_data"]]

    normalization_list = [1, 1]
    pairs_to_compare = [(0, 1), (1,2), (0,2)]
    y_labels = ["# SC neighbors", "# HC neighbors"]
    data_labels = ["SC neighbors", "HC neighbors"]
    # color = ["blue", "white", "blue", "white"]
    color = ["white"]*4
    edge_color = ["blue", "blue", "blue", "blue"]

    compare_event_statistics((E17_folder, transdiff_folders), E17_reference_files_list, trans_diffreference_files_list,
                             x_labels, pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=color, edge_color=edge_color, show_N=False,
                             show_statistics=False)


def compare_E17_P0_rho_inhibition_neighbors_by_type():
    E17_diff_rho = DataCollector("E17.5 differentiating cells", Rho_inhibition_E17_folders,
                             ["neighbors_by_type_differentiation_data"],
                             ["HC neighbors"])
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", Rho_inhibition_E17_folders,
                             ["neighbors_by_type_reference_SC_frame91_data"],
                             ["HC neighbors"])
    P0_diff_rho = DataCollector("P0 differentiating cells",
                            Rho_inhibition_P0_folders,
                            ["neighbors_by_type_differentiation_data"],
                            ["HC neighbors"])
    P0_ref_SC = DataCollector("P0 differentiating cells",
                            Rho_inhibition_P0_folders,
                            ["neighbors_by_type_reference_SC_frame93_data"],
                            ["HC neighbors"])
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["neighbors_by_type_differentiation_data"] * 3,
                             ["HC neighbors"] * 3)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                            ["neighbors_by_type_differentiation_data"] * 2,
                            ["HC neighbors"] * 2)

    samples_list = [E17_diff, P0_diff, E17_diff_rho, P0_diff_rho]
    pairs_to_compare = [(0, 1), (2, 3), (0, 2), (1,3)]
    color = ["cyan", "pink"] * 2
    edge_color = ["blue", "red", "blue", "red"]
    # samples_list = [E17_diff_rho, P0_diff_rho]
    # pairs_to_compare = [(0, 1)]
    color = ["cyan", "pink"]*2
    edge_color = ["blue", "red"]*2
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=True, show_N=True)
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False)
    # no_SC_samples_list = [E17_diff, P0_diff]
    # pairs_to_compare = [(0, 1)]
    # color = ["white"] * 2
    # edge_color = ["blue", "red"]
    # no_SC_full_fig, no_SC_full_ax, res = compare_and_plot_samples(no_SC_samples_list, pairs_to_compare, continues=False,
    #                                                   plot_style="bar", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=True, show_N=True)
    # no_SC_empty_fig, no_SC_empty_ax, _ = compare_and_plot_samples(no_SC_samples_list, pairs_to_compare, continues=False,
    #                                                   plot_style="bar", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=False, show_N=False)
    plt.figure()
    E17_diff_hist, E17_diff_bin_edges = np.histogram(E17_diff_rho.get_sample(),
                                                                 bins=(np.arange(
                                                                     np.max(E17_diff_rho.get_sample()) + 2) - 0.5))
    E17_ref_hist, E17_ref_bin_edges = np.histogram(E17_ref_SC.get_sample(),
                                                               bins=(np.arange(
                                                                   np.max(E17_ref_SC.get_sample()) + 2) - 0.5))
    E17_rev_hist = E17_diff_hist / E17_ref_hist[:E17_diff_hist.size]
    P0_trans_diff_hist, P0_trans_diff_bin_edges = np.histogram(P0_diff.get_sample(),
                                                               bins=(np.arange(
                                                                   np.max(P0_diff.get_sample()) + 2) - 0.5))
    P0_ref_hist, P0_ref_bin_edges = np.histogram(P0_ref_SC.get_sample(),
                                                             bins=(np.arange(
                                                                 np.max(P0_ref_SC.get_sample()) + 2) - 0.5))
    P0_rev_hist = P0_trans_diff_hist / P0_ref_hist[:P0_trans_diff_hist.size]
    plt.bar(np.arange(E17_rev_hist.size) - 0.125, 100 * E17_rev_hist, width=0.25, color="cyan",
            edgecolor="blue")
    plt.bar(np.arange(P0_rev_hist.size) + 0.125, 100 * P0_rev_hist, width=0.25, color="pink",
            edgecolor="red")
    plt.legend(["E17.5", "P0"], prop={'size': 20})
    plt.xlabel("#HC neighbors", fontsize=20)
    plt.ylabel("% differentiating SCs", fontsize=20)
    plt.ylim([0, 25])
    plt.xticks(np.arange(P0_rev_hist.size), labels=np.arange(P0_rev_hist.size), fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.show()


def compare_E17_P0_density():
    normalization = 0.01
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["HC_density_and_fraction_differentiation_data"]*3,
                             ["HC density"] * 3, normalization)
    E17_initial = DataCollector("E17.5 initial", E17_folders,
                                ["HC_density_and_fraction_reference_SC_frame1_data",
                                 "HC_density_and_fraction_reference_SC_frame1_data",
                                 "HC_density_and_fraction_reference_SC_frame1_data"],
                                ["HC density"]*3, normalization)
    E17_24h = DataCollector("E17.5 +24h", E17_folders,
                                ["HC_density_and_fraction_reference_SC_frame96_data",
                                 "HC_density_and_fraction_reference_SC_frame97_data",
                                 "HC_density_and_fraction_reference_SC_frame96_data"],
                                ["HC density"]*3, normalization)
    E17_48h = DataCollector("E17.5 +48h", E17_folders[:-1],
                                ["HC_density_and_fraction_reference_SC_frame191_data",
                                 "HC_density_and_fraction_reference_SC_frame199_data"],
                                ["HC density"]*2, normalization)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                            ["HC_density_and_fraction_differentiation_data"]*2,
                               ["HC density"] * 2, normalization)
    P0_initial = DataCollector("P0 initial", P0_folders,
                               ["HC_density_and_fraction_reference_SC_frame1_data",
                                          "HC_density_and_fraction_reference_SC_frame1_data"],
                               ["HC density"] * 2, normalization)
    P0_24h = DataCollector("P0 +24h", P0_folders,
                               ["HC_density_and_fraction_reference_SC_frame96_data",
                                "HC_density_and_fraction_reference_SC_frame96_data"],
                               ["HC density"] * 2, normalization)
    P0_48h = DataCollector("P0 +48h", P0_folders[:-1],
                               ["HC_density_and_fraction_reference_SC_frame165_data"],
                               ["HC density"], normalization)
    samples_list = [E17_initial, E17_24h, E17_48h, P0_initial, P0_24h, P0_48h]
    pairs_to_compare = [(0, 1), (1,2), (2,3), (3,4), (4,5)]
    color = ["blue", "blue", "blue", "red", "red", "red"]
    edge_color = ["blue", "blue", "blue", "red", "red", "red"]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=True, show_N=True)
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False)
    samples_list = [E17_initial, E17_24h, E17_48h, E17_diff, P0_initial, P0_24h, P0_48h, P0_diff]
    pairs_to_compare = [(0, 3), (1, 3), (2, 3), (4, 7), (5, 7), (6, 7)]
    color = ["blue", "blue", "blue", "cyan", "red", "red", "red", "pink"]
    edge_color = ["blue", "blue", "blue", "cyan", "red", "red", "red", "pink"]
    full_fig_with_diff, full_ax_with_diff, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=True, show_N=True)
    empty_fig_with_diff, empty_ax_with_diff, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False)
    plt.show()

def compare_E17_P0_number_of_neighbors():
    E17_folder = E17_folders
    P0_folder = P0_folders
    E17_data_files_list = [["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame96_data"],
                           ["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame97_data"],
                           ["number_of_neighbors_differentiation_data", "number_of_neighbors_reference_SC_frame96_data"]]
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
                           ["area_and_roundness_differentiation_data", "area_and_roundness_reference_SC_frame97_data"],
                           ["area_and_roundness_differentiation_data", "area_and_roundness_reference_SC_frame96_data"]]
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
                            "contach_length_reference_SC_frame97_data"],
                           ["contach_length_differentiation_data",
                            "contach_length_reference_SC_frame96_data"]
                           ]
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
    folders = (["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-20-E17.5_ablation\\position3-analysis\\"],
               ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\"]
               )
    E17_data_files_list = [["differentiation_distance_from_ablation_data",
                       "reference_SC_distance_from_ablation_data"]]
    P0_data_files_list = [["distance_from_ablation_differentiation_data",
                           "distance_from_ablation_reference_SC_frame1_data",
                          ]]
    # x_labels = ["Differentiating SCs E17.5", "All SCs E17.5", "Differentiating SCs P0", "All SCs P0"]
    x_labels = [""]*4
    normalization_list = [10, 10]
    pairs_to_compare = [(0, 1), (2,3)]
    y_labels = ["Distance from nearest\n ablation (microns)"]
    data_labels = ["Distance from ablation"]
    compare_event_statistics(folders, E17_data_files_list, P0_data_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=True, color=["blue", "blue", "red", "red"],
                             edge_color=["blue", "blue", "red", "red"], show_statistics=False, show_N=False)

def compare_normal_and_promoted_differentiation_HC_density_and_fraction():
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

def compare_normal_and_promoted_differentiation_HC_neighbors():
    folder = "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\"
    x_labels = ["Promoted differentiations", "Normal differentiations"]
    pairs_to_compare = [(0, 1)]

    data_files_list = ["neighbors_by_type_normal_differentiation_data"]
    reference_files_list = ["neighbors_by_type_promoted_differentiation_data"]
    normalization_list = [1, 1]
    y_labels = ["# HC neighbors"]
    data_labels = ["HC neighbors"]
    compare_event_statistics(folder, data_files_list, reference_files_list, x_labels,
                             pairs_to_compare,
                             normalization_list,
                             data_labels, y_labels, continues=False, color=["red", "purple"],
                             edge_color=["red", "purple"])

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
    data = pd.DataFrame({"stage": ["E17.5"] * 9 + ["P0"] * 6, #+ ["P0 with ablation"] * 3,
                         "type": ["division", "delamination", "differentiation"] * 5,
                        "number_of_events": [24, 30, 32, 11, 3, 28, 21, 18, 59, 0, 0, 14, 2, 10, 9],#, 2, 20, 25],
                        "frames": [173] * 3 + [221] * 3 + [119] *3 + [121] * 3 + [107] * 3,# + [168] * 3,
                        "area": [5661946.62] * 3 + [7256274.51] * 3 + [3573407.49] *3 + [2826552.3] * 3 + [3535219.79] * 3# + [5942120.84] * 3
                         })

    data['number of events per frame per (100 microns)^2'] = data.eval("10000 * 92 * number_of_events / area")
    font_size = 25
    plt.rcParams.update({'font.size': font_size})
    g = sns.catplot(x='type', y='number of events per frame per (100 microns)^2', hue='stage',  data=data,
                    kind="box", palette=["#FFA7A0", "#ABEAC9"], legend_out=True,
                    )
    g.map_dataframe(sns.stripplot, x='type', y='number of events per frame per (100 microns)^2', hue='stage',
                    palette=["#404040"], alpha=0.5, jitter=0, dodge=True, size=10)
    plt.show()

def plot_number_of_differentiations():
    data = pd.DataFrame({"stage": ["E17.5"] * 3 + ["P0"] * 2,
                         "type": ["differentiation"] * 5,
                         "number_of_events": [32, 28,  59, 14, 9],
                         "frames": [173, 221, 119, 121, 107],
                         "area": [5661946.62, 7256274.51, 3573407.49, 2826552.3, 3535219.79],
                         })
    # norm_data = data.eval("10000 * 4 * number_of_events / area")
    norm_data = data.eval("4 * number_of_events / frames")
    E17_data = DataCollector(name="E17.5 differentiations", sample=norm_data[:3])
    P0_data = DataCollector(name="P0 differentiations", sample=norm_data[3:])
    samples_list = [E17_data, P0_data]
    pairs_to_compare = []
    color = ["blue", "red"]
    edge_color = ["blue", "red"]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="box", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=True, show_N=True)
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="box", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False)
    plt.show()

def plot_DAPT_data():
    folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-06-E17.5_DAPT\\position2-results",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-20_P0_DAPT"]
    file_names = ["HC_fraction_vs_time_data"] *2
    colors = ["blue", "red"]
    for folder, file, color in zip(folders, file_names, colors):
        path = os.path.join(folder, file)
        if os.path.isfile(path):
            data = pd.read_pickle(path)
            time_in_hours = data["Frame"].to_numpy()/4
            HC_fraction = data["type_fraction average"].to_numpy()
            plt.plot(time_in_hours[time_in_hours<=48], HC_fraction[time_in_hours<=48], linewidth=7, color=color)
            plt.ylim([0,1])
    plt.show()

def plot_rho_inhibition_roundness():
    folders = [Rho_inhibition_E17_folders[0], Rho_inhibition_P0_folders[0], E17_folders[0], P0_folders[0]]
    file_name = "HC_roundness_vs_time_data"
    avg = []
    se = []
    frames = []
    x = np.array([0, 12, 24, 36, 48])
    for folder in folders:
        path = os.path.join(folder, file_name)
        if os.path.isfile(path):
            data = pd.read_pickle(path)
            avg.append(data["roundness average"].to_numpy())
            se.append(data["roundness se"].to_numpy())
            frames.append(data["Frame"].to_numpy())
    count = 0
    for f,a,s in zip(frames,avg,se):
        if count == 0:
            fmt = "o-b"
        elif count == 1:
            fmt = "o-r"
        elif count == 2:
            fmt = "o--b"
        else:
            fmt = "o--r"
        desired_frames = (x*4).astype(int)
        # existing_frames = np.intersect1d(desired_frames, f)
        existing_frames_idx = np.searchsorted(f, desired_frames)
        if count ==1:
            existing_frames_idx = existing_frames_idx[existing_frames_idx < a.size]
        else:
            existing_frames_idx[existing_frames_idx >= a.size] = a.size - 1
        # running_avg = np.convolve(a, (1/3)*np.ones((3,)))
        # running_se = np.sqrt(np.convolve(s*s, (1/3)*np.ones((3,))))
        plt.errorbar(f[existing_frames_idx]/4,a[existing_frames_idx],yerr=s[existing_frames_idx],fmt=fmt)
        count += 1
    plt.show()

def plot_differentiation_timing_by_n_HC_neighbors():
    E17dapt_timing = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\E17.5 DAPT\\differentiation_timing_by_n_hc_neigh_data")
    E17dapt_rate = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\E17.5 DAPT\\differentiation_rates_by_n_hc_neigh_data")
    E17_timing = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\E17.5 control\\differentiation_timing_by_n_HC_neigh_data")
    E17_rate = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\E17.5 control\\differentiation_rates_by_n_HC_neigh_data")
    P0dapt_timing = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\P0 DAPT\\diffrentiation_timing_by_HC_neigh_data")
    P0dapt_rate = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\P0 DAPT\\diffrentiation_rate_by_HC_neigh_data")
    P0_timing = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\P0 control\\differentiation_timing_by_n_HC_neigh_data")
    P0_rate = pd.read_pickle(
        "C:\\Users\\Kasirer\\Phd\\mouse_ear_project\\Results\\P0 control\\differentiation_rates_by_n_HC_neigh_data")
    fig, ax = plt.subplots()
    colors = ["orange", "red", "blue", "green"]
    for i in range(4):
        ax.plot(np.array([0] + E17dapt_timing['timing for %d HC neighbors' % i])/4,
                100*np.arange(len(E17dapt_timing['timing for %d HC neighbors' % i]) + 1)/E17dapt_timing['initial_abundance for %d HC neighbors' % i], label="%d HC neighbors - DAPT (N=%d)" % (i, E17dapt_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='solid')
    for i in range(4):
        ax.plot(np.array([0] + E17_timing['timing for %d HC neighbors' % i])/4,
                 100*np.arange(len(E17_timing['timing for %d HC neighbors' % i]) + 1)/E17_timing['initial_abundance for %d HC neighbors' % i], label="%d HC neighbors - Control (N=%d)" % (i, E17_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='dashed')
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Differentiated %")
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot(np.array([0] + P0dapt_timing['timing for %d HC neighbors' % i])/4,
                100*np.arange(len(P0dapt_timing['timing for %d HC neighbors' % i]) + 1)/P0dapt_timing['initial_abundance for %d HC neighbors' % i], label="%d HC neighbors - DAPT (N=%d)" % (i, P0dapt_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='solid')
    for i in range(3):
        ax.plot(np.array([0] + P0_timing['timing for %d HC neighbors' % i])/4,
                 100*np.arange(len(P0_timing['timing for %d HC neighbors' % i]) + 1)/P0_timing['initial_abundance for %d HC neighbors' % i], label="%d HC neighbors - Control (N=%d)" % (i, P0_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='dashed')
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Differentiated %")
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot(np.array([0] + E17dapt_rate['timing for %d HC neighbors' % i])/1,
                [0] + E17dapt_rate['rates for %d HC neighbors' % i], label="%d HC neighbors - DAPT (N=%d)" % (i, E17dapt_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='solid')
    for i in range(4):
        ax.plot(np.array([0] + E17_rate['timing for %d HC neighbors' % i])/1,
                 [0] + E17_rate['rates for %d HC neighbors' % i], label="%d HC neighbors - Control (N=%d)" % (i, E17_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='dashed')
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Differentiation Probability")
    fig, ax = plt.subplots()
    for i in range(4):
        ax.plot(np.array([0] + P0dapt_rate['timing for %d HC neighbors' % i])/1,
                [0] + P0dapt_rate['rates for %d HC neighbors' % i], label="%d HC neighbors - DAPT (N=%d)" % (i, P0dapt_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='solid')
    for i in range(3):
        ax.plot(np.array([0] + P0_rate['timing for %d HC neighbors' % i])/1,
                 [0] + P0_rate['rates for %d HC neighbors' % i], label="%d HC neighbors - Control (N=%d)" % (i, P0_timing['initial_abundance for %d HC neighbors' % i]), color=colors[i], linestyle='dashed')
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Differentiation Probability")
    plt.show()

if __name__ == "__main__":
    # plot_DAPT_data()
    # plot_rho_inhibition_roundness()

    fit_circular_ablation_results_to_circle(E17_circular_ablation_folders, P0_circular_ablation_folders, 60)
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
    # compare_E17_P0_density()
    # compare_E17_P0_neighbors_by_type()
    # compare_E17_P0_neighbors_by_type_for_differentiation()
    # compare_E17_P0_HC_neighbors_for_differentiation_and_trans_differentiation()
    # compare_E17_P0_HC_contact_length_for_differentiation_and_trans_differentiation()
    # compare_E17_P0_rho_inhibition_neighbors_by_type()
    # compare_P0_neighbors_by_type_for_differentiation_and_transdiff()
    # compare_E17_neighbors_by_type_for_differentiation_and_transdiff()
    # compare_E17_P0_number_of_neighbors()
    # compare_E17_P0_area_and_roundness()
    # compare_E17_P0_contact_length()

    # compare_distance_from_ablation()
    # compare_normal_and_promoted_differentiation_HC_neighbors()
    # plot_number_of_events()
    # plot_number_of_differentiations()
    # compare_deformability()
    # plot_E17_second_neighbors_by_type()
    # plot_P0_second_neighbors_by_type()
    # plot_E17_rho_inhibition_neighbors_by_type()
    # compare_E17_E19_neighbors()
    # compare_E17_E19_contact_length()
    # compare_E17_E19_area_and_roundness()
    # plot_differentiation_timing_by_n_HC_neighbors()