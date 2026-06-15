import shutil
import pathlib
import tempfile
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os, sys
import subprocess
import seaborn as sns

from statistical_analysis import compare_and_plot_samples, DataCollector

RAW_DATA_FOLDER = r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Raw Data"

# Change before running
E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\position3-analysis\\",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-23_E17.5_utricle_atoh_zo\\position4-analysis",
               "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-11-11-E17.5-utricle\\position4-analysis\\"]
P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position3-analysis\\",
              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-31_P0\\position2-analysis\\",
              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2024-08-04_P0 Utricle_Zo1GFP_atoh1 mCherry\\position2-analysis\\"]
Rho_inhibition_E17_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-12-01_E17.5_utricle_rho_inhibition\\position2_event_statistics\\",
                              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2023-10-05_E17.5_utricle_rho_inhibition_and_DAPT\\position1-analysis\\",
                              "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-01_E17.5_rho\\position1-analysis\\"]
Rho_inhibition_P0_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2023-06-25_P0_atoh_zo_rock_inhibitor\\position3_event_statistics\\",
                             "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-04-15_P0-utricle_Rho\\position1-analysis\\",
                             "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_rho\\position3-analysis\\"]
E17_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-20-E17.5_ablation\\position3-analysis\\",
                        "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-07-29_E17.5_utricle_and_cristae_ablation\\position4-analysis\\",
                        "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-07-14_E17.5_utricle_ablation\\position2-analysis\\"]
P0_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-06-05_P0_utricle_ablation\\position2-analysis\\Event statistics\\",
                       "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-08-29_p0_utricle_ablation\\position2-analysis\\",
                       "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2021-12-26_P0_utricle_ablation\\position1-analysis\\"]
E17_DAPT_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-01-06-E17.5_DAPT\\position2-results"]
P0_DAPT_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2022-02-20_P0_DAPT"]
E19_folders = ["D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2025-03-30_E19.5_utricle\\position1-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2025-03-30_E19.5_utricle\\position2-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2025-04-06_E19.5_utricle\\position1-analysis",
               "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2025-04-06_E19.5_utricle\\position3-analysis"
               ]
P2_folders = ["D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2022-01-03_P2_utricle\\position4-analysis\\",
              "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2022-05-24_P2_utricle\\position1-analysis\\",
              "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2022-05-24_P2_utricle\\position2-analysis\\",
              "D:\\Kasirer\\experimental_results\\fixed\\Vestibule\\2022-05-24_P2_utricle\\position4-analysis\\"
              ]
E17_circular_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-06-19_E17.5_utricle_circular_ablation\\utricle3",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-11_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-11_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-01_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-01_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-01_E17.5_utricle_circular_ablation\\utricle3",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle1",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle2",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle3",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle4",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle5",
                                 "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-01-15_E17.5_utricle_circular_ablation\\utricle6"]
P0_circular_ablation_folders = ["D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_circular_ablation\\utricle2",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-08-24_P0_utricle_circular_ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-14_P0_utricle_circular_ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-09-14_P0_utricle_circular_ablation\\utricle2",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-23_P0-circular_ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-23_P0-circular_ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-23_P0-circular_ablation\\utricle4",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-30_P0_utricle_circular_ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-30_P0_utricle_circular_ablation\\utricle2",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2025-11-30_P0_utricle_circular_ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-02-01-P0-circular-ablation\\utricle1",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-02-01-P0-circular-ablation\\utricle2",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-02-01-P0-circular-ablation\\utricle3",
                                "D:\\Kasirer\\experimental_results\\movies\\Utricle\\2026-02-01-P0-circular-ablation\\utricle4"]
E17_circular_ablation_touching_frames = [11, 11, 7, 7, 6, 10, 7, 5, 7, 5, 7, 10, 10, 7]
P0_circular_ablation_touching_frames = [9, 8, 12, 12, 11, 11, 9, 11, 4, 8, 11, 11, 11, 11]
output_dir = r"C:\Users\Kasirer\Phd\mouse_ear_project\papers\Dynamic lateral inhibition in the utricle\Experimental Data"


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

def fit_circular_ablation_results_to_circle(E17_folders, P0_folders, initial_radius, raw_data_output_folder=None):
    font_size = 40
    plt.rcParams.update({'font.size': font_size})
    # fig1, ax1 = plt.subplots(figsize=(10, 10))
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    colors = ["blue", "red"]
    labels = ["E17.5", "P0"]
    radii_by_time_point = []
    radii_avg = []
    radii_se = []
    stresses = []
    stresses_err = []
    final_radii = []
    final_radii_err = []
    young = []
    young_err = []
    touching_frames_lists = [E17_circular_ablation_touching_frames, P0_circular_ablation_touching_frames]
    if raw_data_output_folder is not None:
        radii_raw_data = []
    for (list_index, folder_list) in enumerate([E17_folders, P0_folders]):
        if not folder_list:
            continue
        fig1, ax1 = plt.subplots()
        touching_frames = touching_frames_lists[list_index]
        stage = "E17.5" if list_index == 0 else "P0"
        print("%s results:"%(stage))
        radii_data = [pd.read_pickle(os.path.join(folder,"inner_circle_radius_data")) for folder in folder_list]
        radii = [radius_data["inner circle:radius average"].values*0.1983 for radius_data in radii_data]
        radii_err = [radius_data["inner circle:radius se"].values*0.1983/np.sqrt(radius_data["N"]) for radius_data in radii_data]
        radii = [np.hstack([initial_radius, radius]) for radius in radii]
        radii_err = [np.hstack([0.1, radius_err]) for radius_err in radii_err]
        stresses.append([])
        stresses_err.append([])
        final_radii.append([])
        final_radii_err.append([])
        young.append([])
        young_err.append([])
        max_len = min(np.max(touching_frames), 11)
        time = np.arange(start=1, stop=max_len + 1, step=1)
        time = np.hstack([0, time])
        # Calculating bulk stress according to 10.7554/eLife.57964
        # calculated stress/viscosity in units of 1/time-unit
        fit_colors = matplotlib.cm.get_cmap('tab20', 20)
        color_index = 0
        for radius, radius_err, touching_frame in zip(radii, radii_err, touching_frames):
            touching_frame = min(max_len, touching_frame)
            radius = radius[:touching_frame]
            radius_err = radius_err[:touching_frame]
            current_time = time[:radius.size]
            if raw_data_output_folder is not None:
                radii_raw_data.append(pd.DataFrame({"Time (min)": current_time, "Radius (um)": radius, "Radius SE (um)": radius_err}))
            time_fit = np.linspace(0, np.max(current_time), 300)
            popt_radius, pcov_radius = curve_fit(lambda t, a, b: (initial_radius - a) * np.exp(-b * t) + a,
                                                 current_time, radius,
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
                    linewidth=3, color=fit_colors(color_index))
            ax1.plot(time[:radius.size], radius, "*",
                         markersize=10,
                         label="%s Data" % labels[list_index], linewidth=2, color=fit_colors(color_index))
            ax1.fill_between(time[:radius.size],  radius - radius_err, radius + radius_err,
                             color=fit_colors(color_index),alpha=0.2,linewidth=0)
            ymin, ymax = ax1.get_ylim()
            ticks = np.arange(np.ceil(ymin)+1, np.floor(ymax)+1, 2)
            ax1.set_yticks(ticks)
            ax1.set_xticks(np.arange(0,12,2))
            color_index += 1
            stresses[list_index].append(stress)
            stresses_err[list_index].append(stress_error)
            final_radii[list_index].append(final_radius)
            final_radii_err[list_index].append(final_radius_err)
            young[list_index].append(young_over_visc)
            young_err[list_index].append(young_over_visc_err)
            print("Bulk_stress_over_viscosity=%f+-%f" % (stress, stress_error))
        current_radii_by_time_point = []
        current_radii_avg = []
        current_radii_se = []
        for i in range(max_len + 1):
            existing_radii = [radii[j][i] for j in range(len(radii)) if len(radii[j]) > i]
            current_radii_by_time_point.append(existing_radii)
            current_radii_avg.append(np.average(existing_radii))
            current_radii_se.append(np.std(existing_radii)/np.sqrt(len(existing_radii)))
            print("time point %d has %d data points" % (i, len(existing_radii)))
        radii_by_time_point.append(current_radii_by_time_point)
        radii_avg.append(current_radii_avg)
        current_radii_se[0] = 0.1
        radii_se.append(current_radii_se)
        ax2.plot(time[:10], radii_avg[list_index][:10], "*-", markersize=30,
                     label="%s Data" %labels[list_index], linewidth=2, color=colors[list_index])
        ax2.fill_between(time[:10], np.array(radii_avg[list_index][:10]) - np.array(radii_se[list_index][:10]),
                         np.array(radii_avg[list_index][:10]) + np.array(radii_se[list_index][:10]),
                         color=colors[list_index], alpha=0.2, linewidth=0)
    if raw_data_output_folder is not None:
        overall_raw_data = pd.DataFrame({"Folder": E17_folders + P0_folders,
                                         "Touching frame": E17_circular_ablation_touching_frames + P0_circular_ablation_touching_frames,
                                         "Final radius (um)": final_radii[0] + final_radii[1],
                                         "Final radius SE (um)": final_radii_err[0] + final_radii_err[1],
                                         "Young's modulus over viscosity (1/min)": young[0] + young[1],
                                         "Young's modulus over viscosity SE (1/min)": young_err[0] + young_err[1],
                                         "Stress over viscosity (1/min)": stresses[0] + stresses[1],
                                         "Stress over viscosity SE (1/min)": stresses_err[0] + stresses_err[1],
                                         })
        average_radii_raw_data = pd.DataFrame({"Time (min)": time[:10],
                                  "E17.5 average radius (um)": radii_avg[0][:10],
                                  "E17.5 radius SE (um)": radii_se[0][:10],
                                  "P0 average radius (um)": radii_avg[1][:10],
                                  "P0 radius SE (um)": radii_se[1][:10],
                                  })

        sheet_names = ["E17.5 experiment %d" % i for i in range(len(E17_folders))] + ["P0 experiment %d" % i for i in range(len(P0_folders))]
        with pd.ExcelWriter(os.path.join(raw_data_output_folder, "circular_ablation_raw_data1.xlsx")) as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall data")
            average_radii_raw_data.to_excel(writer, sheet_name="Average radii")
            for df, name in zip(radii_raw_data, sheet_names):
                df.to_excel(writer, sheet_name=name, index=False)

    ax2.set_xlabel("Time (minutes)", fontsize=font_size)
    ax2.set_ylabel("Radius (microns)", fontsize=font_size)
    ax2.set_xlim([0, 10])
    color = ["cyan", "pink"]
    edge_color = ["blue", "red"]
    statistical_analysis_excel = None if raw_data_output_folder is None else os.path.join(raw_data_output_folder,
                                                                                          "circular_ablation_statistical_analysis.xlsx")
    E17_stress = DataCollector("E17 Stress", sample=np.array(stresses[0]))
    P0_stress = DataCollector("P0 Stress", sample=np.array(stresses[1]))
    fig3, ax3, res = compare_and_plot_samples([E17_stress, P0_stress], [(0, 1)], continues=True,
                                            plot_style="violin", color=color, edge_color=edge_color,
                                            show_statistics=True, show_N=True,
                                              save_to_excel=statistical_analysis_excel, excel_sheet="Stress")
    fig4, ax4, _ = compare_and_plot_samples([E17_stress, P0_stress], [(0, 1)], continues=True,
                                              plot_style="violin", color=color, edge_color=edge_color,
                                              show_statistics=False, show_N=False, scatter=True)
    ax4.set_ylim([0, 0.77])
    E17_touching_frame = DataCollector("E17 touching frames", sample=np.array(E17_circular_ablation_touching_frames))
    P0_touching_frame = DataCollector("P0 touching frames", sample=np.array(P0_circular_ablation_touching_frames))
    fig5, ax5, res = compare_and_plot_samples([E17_touching_frame, P0_touching_frame], [(0, 1)], continues=False,
                                              plot_style="bar", color=color, edge_color=edge_color,
                                              show_statistics=True, show_N=True,
                                              save_to_excel=statistical_analysis_excel, excel_sheet="Touching frame")
    fig6, ax6, _ = compare_and_plot_samples([E17_touching_frame, P0_touching_frame], [(0, 1)], continues=False,
                                            plot_style="bar", color=color, edge_color=edge_color,
                                            show_statistics=False, show_N=False, scatter=True)
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
        ["C:\\Program Files\\ImageMagick-7.1.0-Q16-HDRI\\convert", (movies_dir / "movie_*.png").as_posix(), out_path]
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
def compare_E17_P0_HC_neighbors_with_model():
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["neighbors_by_type_differentiation_data"] * 3,
                             ["HC neighbors"] * 3)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                            ["neighbors_by_type_differentiation_data"] * 3,
                            ["HC neighbors"] * 3)
    model_folder = r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_model"
    psigma_vals = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    model_res = [np.load(os.path.join(model_folder, "stress_dependent_on_random_0_psigma-%.1f_gammaSC-0.5_patoh-0.31 results HC with HC neighbors.npy"%psigma)) for psigma in psigma_vals]

    model0 = DataCollector("psigma=0 model",sample=model_res[0][0])
    # model2 = DataCollector("psigma=2 model", sample=model_res[1][0])
    # model4 = DataCollector("psigma=4 model", sample=model_res[2][0])
    # model6 = DataCollector("psigma=6 model", sample=model_res[3][0])
    model8 = DataCollector("psigma=8 model", sample=model_res[4][0])
    # model10 = DataCollector("psigma=10 model", sample=model_res[5][0])
    # model12 = DataCollector("psigma=12 model", sample=model_res[6][0])

    samples_list = [E17_diff, P0_diff, model0, model8]
    pairs_to_compare = [(0,2), (1,3)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=["cyan", "pink", "turquoise", "orange"],
                                                      edge_color=["blue", "red", "green", "purple"],
                                                      show_statistics=True, show_N=True, hirarchical=True,
                                                      scatter=False)
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=["cyan", "pink", "turquoise", "orange"],
                                                      edge_color=["blue", "red", "green", "purple"],
                                                      show_statistics=False, show_N=False, scatter=True)
    from matplotlib.ticker import MaxNLocator
    empty_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.set_ylim([-0.75, 3.5])
    # pvalues_E17 = [res["pvalues"][(0,i)] for i in range(2,7)]
    # pvalues_P0 = [res["pvalues"][(1, i)] for i in range(2,7)]
    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(pvalues_E17)), pvalues_E17[::-1], "b*-")
    # ax.plot(np.arange(len(pvalues_P0)), pvalues_P0[::-1], "r*-")

    plt.show()

def compare_E17_P0_HC_neighbors_for_differentiation_and_trans_differentiation(raw_data_output_folder=None):
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["neighbors_by_type_differentiation_data"]*3,
                             ["HC neighbors"]*3)
    E17_ablation_diff = DataCollector("E17.5 differentiating cells after ablation",
                            E17_ablation_folders,
                             ["neighbors_by_type_differentiation_data"]*3,
                             ["HC neighbors"]*3)
    E17_ablation_ref_SC = DataCollector("E17.5 reference_SC after ablation",
                                   E17_ablation_folders,
                                   ["neighbors_by_type_reference_SC_frame96_data"]*3,
                                   ["HC neighbors"]*3)
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
                             ["neighbors_by_type_promoted_differentiation_data"]*3,
                             ["HC neighbors"]*3)
    P0_ablation_diff = DataCollector("P0 differentiating cells after ablation",
                                  P0_ablation_folders,
                                  ["neighbors_by_type_differentiation_data"] * 3,
                                  ["HC neighbors"] * 3)
    P0_ablation_ref_SC = DataCollector("P0 reference SC after ablation",
                                  P0_ablation_folders,
                                  ["neighbors_by_type_reference_SC_frame96_data"]*3,
                                  ["HC neighbors"]*3)
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["neighbors_by_type_reference_SC_frame96_data"]*3,
                             ["HC neighbors"] * 3)
    E17_diff.save_sample(out_path=output_dir, by_groups=True)
    P0_diff.save_sample(out_path=output_dir, by_groups=True )
    if raw_data_output_folder is not None:
        excel_path = os.path.join(raw_data_output_folder,
                                  "differentiation_and_trans-differentiation_statistical_analysis.xlsx")
    else:
        excel_path = None
    samples_list = [E17_diff, E17_ref_SC, P0_diff, P0_trans_diff, P0_ref_SC]
    all_samples_list = samples_list + [E17_ablation_diff, E17_ablation_ref_SC, P0_ablation_diff, P0_ablation_ref_SC]
    pairs_to_compare = [(0,1),(0,2), (2,3), (2,4), (0,3)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="histogram", color= ["cyan"] * 2 + ["pink"] *3, edge_color=["blue"] * 2 + ["red"] * 3,
                                            show_statistics=True, show_N=True, hirarchical=True, scatter=False,
                                                      save_to_excel=excel_path, excel_sheet="HC neighbors")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=["cyan"] * 2 + ["pink"] * 3,
                                                      edge_color=["blue"] * 2 + ["red"] * 3,
                                                      show_statistics=False, show_N=False, scatter=True, hatch=[None]*3 + ['/', None])

    from matplotlib.ticker import MaxNLocator
    empty_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    E17_rev_hist = [None]*3
    P0_rev_hist = [None]*3
    E17_ablation_rev_hist = [None] * 3
    P0_ablation_rev_hist = [None] * 3
    for i in range(3):
        E17_diff_hist, E17_diff_bin_edges = np.histogram(E17_diff.get_partial_sample(i), bins=(np.arange(np.max(E17_diff.get_partial_sample(i)) + 2) - 0.5))
        E17_ref_hist, E17_ref_bin_edges = np.histogram(E17_ref_SC.get_partial_sample(i),
                                                    bins=(np.arange(np.max(E17_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        E17_rev_hist[i] = E17_diff_hist/E17_ref_hist[:E17_diff_hist.size]
        E17_ablation_diff_hist, E17_ablation_diff_bin_edges = np.histogram(E17_ablation_diff.get_partial_sample(i), bins=(
                np.arange(np.max(E17_ablation_diff.get_partial_sample(i)) + 2) - 0.5))
        E17_ablation_ref_hist, E17_ablation_ref_bin_edges = np.histogram(E17_ablation_ref_SC.get_partial_sample(i),
                                                                   bins=(np.arange(
                                                                       np.max(E17_ablation_ref_SC.get_partial_sample(
                                                                           i)) + 2) - 0.5))
        E17_ablation_rev_hist[i] = E17_ablation_diff_hist / E17_ablation_ref_hist[:E17_ablation_diff_hist.size]
    for i in range(3):
        P0_diff_hist, P0_diff_bin_edges = np.histogram(P0_diff.get_partial_sample(i),
                                                     bins=(np.arange(np.max(P0_diff.get_partial_sample(i)) + 2) - 0.5))
        P0_ref_hist, P0_ref_bin_edges = np.histogram(P0_ref_SC.get_partial_sample(i),
                                                   bins=(np.arange(np.max(P0_ref_SC.get_partial_sample(i)) + 2) - 0.5))
        P0_rev_hist[i] = P0_diff_hist / P0_ref_hist[:P0_diff_hist.size]
        P0_ablation_diff_hist, P0_ablation_diff_bin_edges = np.histogram(P0_ablation_diff.get_partial_sample(i),
                                                                   bins=(np.arange(
                                                                       np.max(P0_ablation_diff.get_partial_sample(
                                                                           i)) + 2) - 0.5))
        P0_ablation_ref_hist, P0_ablation_ref_bin_edges = np.histogram(P0_ablation_ref_SC.get_partial_sample(i),
                                                                 bins=(np.arange(
                                                                     np.max(P0_ablation_ref_SC.get_partial_sample(
                                                                         i)) + 2) - 0.5))
        P0_ablation_rev_hist[i] = P0_ablation_diff_hist / P0_ablation_ref_hist[:P0_ablation_diff_hist.size]
    from itertools import zip_longest
    E17_rev_hist = np.array(list(zip_longest(*E17_rev_hist, fillvalue=0))).T
    P0_rev_hist = np.array(list(zip_longest(*P0_rev_hist, fillvalue=0))).T
    E17_rev_averages = np.average(E17_rev_hist, axis=0)
    P0_rev_averages = np.average(P0_rev_hist, axis=0)
    E17_ablation_rev_hist = np.array(list(zip_longest(*E17_ablation_rev_hist, fillvalue=0))).T
    P0_ablation_rev_hist = np.array(list(zip_longest(*P0_ablation_rev_hist, fillvalue=0))).T
    E17_ablation_rev_averages = np.average(E17_ablation_rev_hist, axis=0)
    P0_ablation_rev_averages = np.average(P0_ablation_rev_hist, axis=0)
    plt.bar(np.arange(E17_rev_averages.size)-0.25, 100*E17_rev_averages, width=0.125, color="cyan", edgecolor="blue")
    plt.bar(np.arange(P0_rev_averages.size)+0.125, 100*P0_rev_averages, width=0.125, color="pink", edgecolor="red")
    plt.bar(np.arange(E17_ablation_rev_averages.size) - 0.125, 100 * E17_ablation_rev_averages, width=0.125, color="cyan",
            edgecolor="blue", hatch='/')
    plt.bar(np.arange(P0_ablation_rev_averages.size) + 0.25, 100 * P0_ablation_rev_averages, width=0.125, color="pink",
            edgecolor="red", hatch='/')
    for i in range(E17_rev_hist.shape[0]):
        plt.scatter(np.arange(E17_rev_averages.size) - 0.25, 100 * E17_rev_hist[i], marker=".", s=30,
                    color="black")
    for i in range(P0_rev_hist.shape[0]):
        plt.scatter(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_hist[i], marker=".", s=30,
                    color="black")
    for i in range(E17_ablation_rev_hist.shape[0]):
        plt.scatter(np.arange(E17_ablation_rev_averages.size) - 0.125, 100 * E17_ablation_rev_hist[i], marker=".", s=30,
                    color="black")
    for i in range(P0_ablation_rev_hist.shape[0]):
        plt.scatter(np.arange(P0_ablation_rev_averages.size) + 0.25, 100 * P0_ablation_rev_hist[i], marker=".", s=30,
                    color="black")
    from statistical_analysis import TwoByTwoCompare
    for n_neighbors in range(len(E17_rev_hist[0])):
        E17_diff = np.array([E17_rev_hist[i][n_neighbors] for i in range(3)])
        P0_diff = np.array([P0_rev_hist[i][n_neighbors] for i in range(3)])
        E17_ablation = np.array([E17_ablation_rev_hist[i][n_neighbors] for i in range(3)])
        P0_ablation = np.array([P0_ablation_rev_hist[i][n_neighbors] for i in range(3)])
        comparer = TwoByTwoCompare(E17_diff, E17_ablation, P0_diff, P0_ablation,
                                   factorA_name="Stage",
                                   factorB_name="Ablation",
                                   A_levels=("E17.5", "P0"),
                                   B_levels=("Normal development", "HC ablation"))

        res = comparer.compare(verbose=True, save_to_excel=excel_path, sheet="Differentiating Percentage",label="%d HC neig."%n_neighbors)
    plt.xlabel("#HC neighbors", fontsize=20)
    plt.ylabel("% differentiating SCs", fontsize=20)
    plt.ylim([0, 30])
    plt.xticks(np.arange(P0_rev_hist.size), labels=np.arange(P0_rev_hist.size), fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([-0.35, 2.35])
    plt.tight_layout()

    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "differentiation_and_trans-differentiation_patterns_raw_data1.xlsx")
        E17_data = {"Stage": ["E17.5"] * 6, "Experiment #": np.arange(6), "Condition": ["Normal development"]*3 + ["HC ablation"]*3}
        E17_data.update({"%% differentiating with %d HC neighbors" % j: [100 * E17_rev_hist[i][j] for i in
                                                                         range(E17_rev_hist.shape[0])] + [100 * E17_ablation_rev_hist[i][j] for i in
                                                                         range(E17_ablation_rev_hist.shape[0])] for j in
                         range(E17_rev_hist.shape[1])})
        E17_df = pd.DataFrame(E17_data)
        P0_data = {"Stage": ["P0"] * 6, "Experiment #": np.arange(6), "Condition": ["Normal development"]*3 + ["HC ablation"]*3}
        P0_data.update(
            {"%% differentiating with %d HC neighbors" % j: [100 * P0_rev_hist[i][j] for i in
                                                             range(P0_rev_hist.shape[0])] + [100 * P0_ablation_rev_hist[i][j] for i in
                                                             range(P0_ablation_rev_hist.shape[0])] for j in
             range(P0_rev_hist.shape[1])})
        P0_df = pd.DataFrame(P0_data)
        diff_percentage = pd.concat([E17_df, P0_df], axis=0)
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in all_samples_list],
                                         "# HC neighbors average": [sample.get_average_of_groups() for sample in
                                                                    all_samples_list],
                                         "# HC neighbors SE": [sample.get_se_of_groups() for sample in all_samples_list],
                                         "% no HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 0) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 1 HC neighbor": [
                                             100 * np.count_nonzero(sample.get_sample() == 1) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 2 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 2) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 3 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 3) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 4 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 4) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 5 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 5) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 6 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 6) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         "% 7 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 7) / sample.get_sample_size()
                                             for sample in
                                             all_samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall HC neighbors data")
            diff_percentage.to_excel(writer, sheet_name="Differentiating Percentage data")
        for sample in all_samples_list:
            sample.save_to_excel(out_path, "# HC neighbors")
    plt.show()

def compare_E17_P0_HC_contact_length_for_differentiation_and_trans_differentiation(raw_data_output_folder=None):
    E17_diff = DataCollector("E17.5 differentiating cells contacts length", E17_folders,
                             ["contact_length_by_type_differentiation_data"]*3,
                             ["HC contact length"]*3, normalization=10)
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", E17_folders,
                               ["contact_length_by_type_reference_SC_frame96_data",
                                          "contact_length_by_type_reference_SC_frame97_data",
                                          "contact_length_by_type_reference_SC_frame96_data"],
                             ["HC contact length"] * 3, normalization=10)
    P0_diff = DataCollector("P0 differentiating cells contact length", P0_folders,
                             ["contact_length_by_type_differentiation_data"] * 3,
                             ["HC contact length"] * 3, normalization=10)
    P0_trans_diff = DataCollector("P0 trans-differentiating cells", P0_ablation_folders,
                             ["contact_length_by_type_promoted_differentiation_data"]*3,
                             ["HC contact length"]*3, normalization=10)
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["contact_length_by_type_reference_SC_frame96_data"]*3,
                             ["HC contact length"] * 3, normalization=10)
    E17_ablation_diff = DataCollector("E17.5 differentiating cells after ablation",
                                      E17_ablation_folders,
                                      ["contact_length_by_type_differentiation_data"] * 3,
                                      ["HC contact length"] * 3)
    E17_ablation_ref_SC = DataCollector("E17.5 reference_SC after ablation",
                                        E17_ablation_folders,
                                        ["contact_length_by_type_reference_SC_frame96_data"] * 3,
                                        ["HC contact length"] * 3)
    P0_ablation_diff = DataCollector("P0 differentiating cells after ablation",
                                     P0_ablation_folders,
                                     ["contact_length_by_type_differentiation_data"] * 3,
                                     ["HC contact length"] * 3)
    P0_ablation_ref_SC = DataCollector("P0 reference SC after ablation",
                                       P0_ablation_folders,
                                       ["contact_length_by_type_reference_SC_frame96_data"] * 3,
                                       ["HC contact length"] * 3)
    if raw_data_output_folder is not None:
        excel_path = os.path.join(raw_data_output_folder,
                                  "differentiation_and_trans-differentiation_statistical_analysis.xlsx")
    else:
        excel_path = None
    samples_list = [E17_diff, E17_ref_SC, P0_diff, P0_trans_diff, P0_ref_SC]
    all_samples_list = samples_list + [E17_ablation_diff, E17_ablation_ref_SC, P0_ablation_diff, P0_ablation_ref_SC]
    pairs_to_compare = [(0,1),(0,2), (0,3), (2,3), (2,4)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                            plot_style="violin", color= ["cyan"] * 2 + ["pink"] *3, edge_color=["blue"] * 2 + ["red"] * 3,
                                            show_statistics=True, show_N=True, hirarchical=True, scatter=True,
                                                      save_to_excel=excel_path, excel_sheet="HC neighbors")

    full_ax.set_ylabel("Apical contact length with neighboring HCs (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=["cyan"] * 2 + ["pink"] * 3,
                                                      edge_color=["blue"] * 2 + ["red"] * 3,
                                                      show_statistics=False, show_N=False, scatter=True,
                                                      hatch=[None]*3 + ['/', None])

    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "differentiation_and_trans-differentiation_patterns_raw_data.xlsx")
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in all_samples_list],
                                         "Apical contact length with HC average (um)": [sample.get_average_of_groups() for sample in
                                                                    all_samples_list],
                                         "Apical contact length with HC SE (um)": [sample.get_se_of_groups() for sample in all_samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall HC contact data")
        for sample in all_samples_list:
            sample.save_to_excel(out_path, "HC contact len")
    plt.show()


def compare_E17_E19_and_P0_P2_neighbors(raw_data_output_folder=None):
    E17_after_48h_HC_neighbors_for_HC = DataCollector("E17.5 +48h HC neigh for HC", E17_folders,
                                                      ["neighbors_by_type_reference_HC_frame191_data",
                                                       "neighbors_by_type_reference_HC_frame199_data",
                                                       "neighbors_by_type_reference_HC_frame120_data"],
                                                      ["HC neighbors"] * 3)
    E17_after_48h_HC_neighbors_for_SC = DataCollector("E17.5 +48h HC neigh for SC", E17_folders,
                                                      ["neighbors_by_type_reference_SC_frame191_data",
                                                       "neighbors_by_type_reference_SC_frame199_data",
                                                       "neighbors_by_type_reference_SC_frame120_data"],
                                                      ["HC neighbors"] * 3)
    E19_HC_neighbors_for_HC = DataCollector("E19.5 HC neigh for HC", E19_folders,
                                            ["neighbors_by_type_reference_HC_frame1_data"]*len(E19_folders),
                                            ["HC neighbors"]*len(E19_folders))
    E19_HC_neighbors_for_SC = DataCollector("E19.5 HC neigh for SC", E19_folders,
                                            ["neighbors_by_type_reference_SC_frame1_data"] * len(E19_folders),
                                            ["HC neighbors"] * len(E19_folders))
    P0_after_48h_HC_neighbors_for_HC = DataCollector("P0 +48h HC neigh for HC", P0_folders,
                                                      ["neighbors_by_type_reference_HC_frame165_data",
                                                       "neighbors_by_type_reference_HC_frame144_data",
                                                       "neighbors_by_type_reference_HC_frame130_data"],
                                                      ["HC neighbors"] * 3)
    P0_after_48h_HC_neighbors_for_SC = DataCollector("P0 +48h HC neigh for SC", P0_folders,
                                                      ["neighbors_by_type_reference_SC_frame165_data",
                                                       "neighbors_by_type_reference_SC_frame144_data",
                                                       "neighbors_by_type_reference_SC_frame130_data"],
                                                      ["HC neighbors"] * 3)
    P2_HC_neighbors_for_HC = DataCollector("P2 HC neigh for HC", P2_folders,
                                            ["neighbors_by_type_reference_HC_frame1_data"] * len(P2_folders),
                                            ["HC neighbors"] * len(P2_folders))
    P2_HC_neighbors_for_SC = DataCollector("P2 HC neigh for SC", P2_folders,
                                            ["neighbors_by_type_reference_SC_frame1_data"] * len(P2_folders),
                                            ["HC neighbors"] * len(P2_folders))
    samples_list = [E17_after_48h_HC_neighbors_for_HC, E19_HC_neighbors_for_HC,
                    P0_after_48h_HC_neighbors_for_HC, P2_HC_neighbors_for_HC,
                    E17_after_48h_HC_neighbors_for_SC, E19_HC_neighbors_for_SC,
                    P0_after_48h_HC_neighbors_for_SC, P2_HC_neighbors_for_SC,
                    ]
    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                            plot_style="histogram", color= ["cyan", "turquoise", "pink", "yellow"]*2,
                                                      edge_color=["blue", "green", "red", "orange"]*2,
                                            show_statistics=True, show_N=True, hirarchical=True,
                                                      save_to_excel=os.path.join(raw_data_output_folder, "fixed_live_comparison_statistical_analysis.xlsx"),
                                                      excel_sheet="HC neighbors")

    full_ax.set_ylabel("# neighbors")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color= ["cyan", "turquoise", "pink", "yellow"]*2,
                                                      edge_color=["blue", "green", "red", "orange"]*2,
                                                      show_statistics=False, show_N=False, scatter=True)

    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "fixed_live_comparison_data.xlsx")
        mode = "a" if os.path.isfile(out_path) else "w"
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in samples_list],
                                         "# HC neighbors average": [sample.get_average_of_groups() for sample in samples_list],
                                         "# HC neighbors SE": [sample.get_se_of_groups() for sample in samples_list],
                                         "% no HC neighbors": [100*np.count_nonzero(sample.get_sample() == 0)/sample.get_sample_size() for sample in
                                                               samples_list],
                                         "% 1 HC neighbor": [100*np.count_nonzero(sample.get_sample() == 1)/sample.get_sample_size() for sample in
                                                             samples_list],
                                         "% 2 HC neighbors": [100*np.count_nonzero(sample.get_sample() == 2)/sample.get_sample_size() for sample in
                                                               samples_list],
                                         "% 3 HC neighbors": [100*np.count_nonzero(sample.get_sample() == 3)/sample.get_sample_size() for sample in
                                                              samples_list],
                                         "% 4 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 4) / sample.get_sample_size()
                                             for sample in
                                             samples_list],
                                         "% 5 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 5) / sample.get_sample_size()
                                             for sample in
                                             samples_list],
                                         "% 6 HC neighbors": [
                                             100 * np.count_nonzero(sample.get_sample() == 6) / sample.get_sample_size()
                                             for sample in
                                             samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode=mode) as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall HC neighbors data")
        for sample in samples_list:
            sample.save_to_excel(out_path, "# HC neighbors")
    plt.show()

def compare_E17_E19_and_P0_P2_contact_length(raw_data_output_folder=None):
    E17_after_48h_HC_contact_length_for_HC = DataCollector("E17.5 +48h HC HC contact len", E17_folders,
                               ["contact_length_by_type_reference_HC_frame191_data",
                                          "contact_length_by_type_reference_HC_frame199_data",
                                          "contact_length_by_type_reference_HC_frame120_data"],
                                         ["HC contact length"] * 3, normalization=10)
    E17_after_48h_HC_contact_length_for_SC = DataCollector("E17.5 +48h HC SC contact len", E17_folders,
                                                      ["contact_length_by_type_reference_SC_frame191_data",
                                                       "contact_length_by_type_reference_SC_frame199_data",
                                                       "contact_length_by_type_reference_SC_frame120_data"],
                                                      ["HC contact length"] * 3, normalization=10)
    E19_HC_contact_length_for_HC = DataCollector("E19.5 HC HC contact len", E19_folders,
                                            ["contact_length_by_type_reference_HC_frame1_data"]*len(E19_folders),
                                            ["HC contact length"]*len(E19_folders), normalization=10)
    E19_HC_contact_length_for_SC = DataCollector("E19.5 HC SC contact len", E19_folders,
                                            ["contact_length_by_type_reference_SC_frame1_data"] * len(E19_folders),
                                            ["HC contact length"] * len(E19_folders), normalization=10)
    P0_after_48h_HC_contact_length_for_HC = DataCollector("P0 +48h HC HC contact len", P0_folders,
                                                           ["contact_length_by_type_reference_HC_frame165_data",
                                                            "contact_length_by_type_reference_HC_frame144_data",
                                                            "contact_length_by_type_reference_HC_frame130_data"],
                                                           ["HC contact length"] * 3, normalization=10)
    P0_after_48h_HC_contact_length_for_SC = DataCollector("P0 +48h HC SC contact len", P0_folders,
                                                           ["contact_length_by_type_reference_SC_frame165_data",
                                                            "contact_length_by_type_reference_SC_frame144_data",
                                                            "contact_length_by_type_reference_SC_frame130_data"],
                                                           ["HC contact length"] * 3, normalization=10)
    P2_HC_contact_length_for_HC = DataCollector("P2 HC HC contact len", P2_folders,
                                                 ["contact_length_by_type_reference_HC_frame1_data"] * len(P2_folders),
                                                 ["HC contact length"] * len(P2_folders), normalization=10)
    P2_HC_contact_length_for_SC = DataCollector("P2 HC SC contact len", P2_folders,
                                                 ["contact_length_by_type_reference_SC_frame1_data"] * len(P2_folders),
                                                 ["HC contact length"] * len(P2_folders), normalization=10)

    samples_list = [E17_after_48h_HC_contact_length_for_HC, E19_HC_contact_length_for_HC,
                    P0_after_48h_HC_contact_length_for_HC, P2_HC_contact_length_for_HC,
                    E17_after_48h_HC_contact_length_for_SC, E19_HC_contact_length_for_SC,
                    P0_after_48h_HC_contact_length_for_SC, P2_HC_contact_length_for_SC,
                    ]
    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                            plot_style="violin", color= ["cyan", "turquoise", "pink", "yellow"]*2,
                                                      edge_color=["blue", "green", "red", "orange"]*2,
                                            show_statistics=True, show_N=True, hirarchical=True,
                                                      save_to_excel=os.path.join(raw_data_output_folder,
                                                                                 "fixed_live_comparison_statistical_analysis.xlsx"),
                                                      excel_sheet="HC contact len")

    full_ax.set_ylabel("Contact length (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                                      plot_style="violin", color= ["cyan", "turquoise", "pink", "yellow"]*2,
                                                      edge_color=["blue", "green", "red", "orange"]*2,
                                                      show_statistics=False, show_N=False, scatter=True)
    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "fixed_live_comparison_data1.xlsx")
        mode = "a" if os.path.isfile(out_path) else "w"
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in samples_list],
                                         "Apical contact length with HCs average (um)": [sample.get_average_of_groups() for sample in samples_list],
                                         "Apical contact length with HCs SE (um)": [sample.get_se_of_groups() for sample in samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode=mode) as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall apical contact len data")
        for sample in samples_list:
            sample.save_to_excel(out_path, "Apical contact length with HCs (um)")
    plt.show()

def compare_roundness_with_model(raw_data_output_folder=None):
    E17_after_48h_SC_roundness = DataCollector("E17.5 +48h SC roundness", E17_folders,
                                               ["area_and_roundness_reference_SC_frame191_data",
                                                "area_and_roundness_reference_SC_frame199_data",
                                                "area_and_roundness_reference_SC_frame120_data"],
                                               ["roundness"] * 3, normalization=1)
    E17_after_48h_HC_roundness = DataCollector("E17.5 +48h HC roundness", E17_folders,
                                                      ["area_and_roundness_reference_HC_frame191_data",
                                                       "area_and_roundness_reference_HC_frame199_data",
                                                       "area_and_roundness_reference_HC_frame120_data"],
                                                      ["roundness"] * 3, normalization=1)
    P0_after_48h_SC_roundness = DataCollector("P0 +48h SC roundness", P0_folders,
                                              ["area_and_roundness_reference_SC_frame165_data",
                                               "area_and_roundness_reference_SC_frame144_data",
                                               "area_and_roundness_reference_SC_frame130_data"],
                                              ["roundness"] * 3, normalization=1)
    P0_after_48h_HC_roundness = DataCollector("P0 +48h HC roundness", P0_folders,
                                               ["area_and_roundness_reference_HC_frame165_data",
                                                "area_and_roundness_reference_HC_frame144_data",
                                                "area_and_roundness_reference_HC_frame130_data"],
                                               ["roundness"] * 3, normalization=1)
    model_folder = r"C:\Users\Kasirer\Phd\mouse_ear_project\tissue_model"
    gammaSC_vals = [0.01]
    psigma_vals = [0.0]
    gammaHC_ratio_vals = [2.0, 4.0, 6.0, 8.0, 10.0, 20.0]
    alphaHC_ratio_vals = [1.0]
    model_res = [np.load(os.path.join(model_folder,
                                      "stress_dependent_on_random_0_psigma-%.1f_gammaSC-0.5_patoh-0.31 results HC with HC neighbors.npy" % psigma))
                 for psigma in psigma_vals]

    model0 = DataCollector("psigma=0 model", sample=model_res[0][0])
    # model2 = DataCollector("psigma=2 model", sample=model_res[1][0])
    # model4 = DataCollector("psigma=4 model", sample=model_res[2][0])
    # model6 = DataCollector("psigma=6 model", sample=model_res[3][0])
    model8 = DataCollector("psigma=8 model", sample=model_res[4][0])
    # model10 = DataCollector("psigma=10 model", sample=model_res[5][0])
    # model12 = DataCollector("psigma=12 model", sample=model_res[6][0])

    pairs_to_compare = [(0,1), (2,3), (4,5), (6,7)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                            plot_style="violin", color=["cyan", "turquoise", "pink", "yellow"] * 4,
                                                      edge_color=["blue", "green", "red", "orange"] * 4,
                                            show_statistics=True, show_N=True, hirarchical=True)

    full_ax.set_ylabel("Contact length (microns)")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                                      plot_style="violin", color= ["cyan", "turquoise", "pink", "yellow"] * 4,
                                                      edge_color=["blue", "green", "red", "orange"] * 4,
                                                      show_statistics=False, show_N=False, scatter=True)

    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "fixed_live_comparison_data1.xlsx")
        mode = "a" if os.path.isfile(out_path) else "w"
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in samples_list],
                                         "Roundness average": [sample.get_average() for sample in samples_list],
                                         "Roundness SE": [sample.get_se() for sample in samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode=mode) as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall roundness data")
        for sample in samples_list:
            sample.save_to_excel(out_path, "Roundness")
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


def compare_E17_P0_rho_inhibition_neighbors_by_type(raw_data_output_folder=None):
    E17_diff_rho = DataCollector("Rho i E17.5 differentiating cells", Rho_inhibition_E17_folders,
                             ["neighbors_by_type_differentiation_data"]*3,
                             ["HC neighbors"]*3)
    E17_ref_SC_rho = DataCollector("Rho i E17.5 reference SC +24h", Rho_inhibition_E17_folders,
                             ["neighbors_by_type_reference_SC_frame91_data",
                              "neighbors_by_type_reference_SC_frame96_data",
                              "neighbors_by_type_reference_SC_frame86_data"],
                             ["HC neighbors"]*3)
    P0_diff_rho = DataCollector("Rho i P0 differentiating cells",
                            Rho_inhibition_P0_folders,
                            ["neighbors_by_type_differentiation_data"]*3,
                            ["HC neighbors"]*3)
    P0_ref_SC_rho = DataCollector("Rho i P0 reference SC +24h",
                            Rho_inhibition_P0_folders,
                            ["neighbors_by_type_reference_SC_frame93_data"] + ["neighbors_by_type_reference_SC_frame96_data"]*2,
                            ["HC neighbors"]*3)
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["neighbors_by_type_differentiation_data"] * 3,
                             ["HC neighbors"] * 3)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                            ["neighbors_by_type_differentiation_data"] * 3,
                            ["HC neighbors"] * 3)
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", E17_folders,
                               ["neighbors_by_type_reference_SC_frame96_data",
                                "neighbors_by_type_reference_SC_frame97_data",
                                "neighbors_by_type_reference_SC_frame96_data"],
                               ["HC neighbors"] * 3)
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["neighbors_by_type_reference_SC_frame96_data"] * 3,
                              ["HC neighbors"] * 3)

    if raw_data_output_folder is not None:
        excel_path = os.path.join(raw_data_output_folder,
                                  "ROCK_inhibition_statistical_analysis.xlsx")
    else:
        excel_path = None
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
                                                      show_statistics=True, show_N=True, hirarchical=True,
                                                      save_to_excel=excel_path, excel_sheet="HC neighbors")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="histogram", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False, hirarchical=True, scatter=True)
    from matplotlib.ticker import MaxNLocator
    empty_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.set_ylim([-0.75, 3.5])
    plt.figure()
    E17_rev_hist = [None] * 3
    P0_rev_hist = [None] * 3
    for i in range(3):
        E17_rho_diff_hist, E17_rho_diff_bin_edges = np.histogram(E17_diff_rho.get_partial_sample(i),
                                                                     bins=(np.arange(
                                                                         np.max(E17_diff_rho.get_sample()) + 2) - 0.5))
        E17_rho_ref_hist, E17_rho_ref_bin_edges = np.histogram(E17_ref_SC_rho.get_partial_sample(i),
                                                                   bins=(np.arange(
                                                                       np.max(E17_ref_SC.get_sample()) + 2) - 0.5))

        P0_rho_diff_hist, P0_rho_diff_bin_edges = np.histogram(P0_diff_rho.get_partial_sample(i),
                                                                   bins=(np.arange(
                                                                       np.max(P0_diff.get_sample()) + 2) - 0.5))
        P0_rho_ref_hist, P0_ref_bin_edges = np.histogram(P0_ref_SC_rho.get_partial_sample(i),
                                                                 bins=(np.arange(
                                                                     np.max(P0_ref_SC.get_sample()) + 2) - 0.5))

        E17_rev_hist[i] = E17_rho_diff_hist / E17_rho_ref_hist[:E17_rho_diff_hist.size]
        P0_rev_hist[i] = P0_rho_diff_hist / P0_rho_ref_hist[:P0_rho_diff_hist.size]
    from itertools import zip_longest
    E17_rev_hist = np.array(list(zip_longest(*E17_rev_hist, fillvalue=0))).T
    P0_rev_hist = np.array(list(zip_longest(*P0_rev_hist, fillvalue=0))).T
    E17_rev_averages = np.average(E17_rev_hist, axis=0)
    P0_rev_averages = np.average(P0_rev_hist, axis=0)
    plt.bar(np.arange(E17_rev_averages.size) - 0.125, 100 * E17_rev_averages, width=0.25, color="cyan",
            edgecolor="blue")
    plt.bar(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_averages, width=0.25, color="pink", edgecolor="red")

    for i in range(E17_rev_hist.shape[0]):
        plt.scatter(np.arange(E17_rev_averages.size) - 0.125, 100 * E17_rev_hist[i], marker=".", s=30,
                    color="black")
    for i in range(P0_rev_hist.shape[0]):
        plt.scatter(np.arange(P0_rev_averages.size) + 0.125, 100 * P0_rev_hist[i], marker=".", s=30,
                    color="black")
    from statistical_analysis import TwoSampleCompare
    for n_neighbors in range(len(E17_rev_hist[0])):
        E17_diff_percentage = np.array([E17_rev_hist[i][n_neighbors] for i in range(3)])
        P0_diff_percentage = np.array([P0_rev_hist[i][n_neighbors] for i in range(3)])
        samples = [E17_diff_percentage, P0_diff_percentage]
        labels = ["E17.5 diff", "P0 diff"]
        pairs = [(0, 1)]
        for pair in pairs:
            comparer = TwoSampleCompare(samples[pair[0]], samples[pair[1]], sample1_label=labels[pair[0]],
                                        sample2_label=labels[pair[1]], continues=True)
            pval = comparer.compare_samples(save_to_excel=excel_path, sheet="Differentiating percentage", label="%d HC neighbors" % n_neighbors)
            print("pval for %d HC neighbors between %s and %s = %f" % (
            n_neighbors, labels[pair[0]], labels[pair[1]], pval))

    plt.xlabel("#HC neighbors", fontsize=20)
    plt.ylabel("% differentiating SCs", fontsize=20)
    plt.ylim([0, 22])
    plt.xticks(np.arange(P0_rev_hist.size), labels=np.arange(P0_rev_hist.size), fontsize=20)

    plt.yticks(fontsize=20, ticks=[0, 5, 10, 15, 20])
    plt.xlim([-0.32, 2.32])
    plt.tight_layout()
    all_samples_list = [E17_diff_rho, P0_diff_rho, E17_ref_SC_rho, P0_ref_SC_rho, E17_diff, P0_diff, E17_ref_SC, P0_ref_SC]
    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "ROCK_inhibition_raw_data.xlsx")
        E17_data = {"Stage" : ["E17.5"]*3, "Experiment #" : np.arange(3)}
        E17_data.update({"%% differentiating with %d HC neighbors" % j: [100*E17_rev_hist[i][j] for i in range(E17_rev_hist.shape[0])] for j in range(E17_rev_hist.shape[1])})
        E17_df = pd.DataFrame(E17_data)

        E17_df = pd.DataFrame(E17_data)
        P0_data = {"Stage": ["P0"] * 3, "Experiment #": np.arange(3)}
        P0_data.update(
            {"%% differentiating with %d HC neighbors" % j: [100*P0_rev_hist[i][j] for i in range(P0_rev_hist.shape[0])] for j in range(P0_rev_hist.shape[1])})
        P0_df = pd.DataFrame(P0_data)
        diff_percentage = pd.concat([E17_df, P0_df], axis=0)

        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in all_samples_list],
                                         "# HC neighbors average": [sample.get_average_of_groups() for sample in all_samples_list],
                                         "# HC neighbors SE": [sample.get_se_of_groups() for sample in all_samples_list],
                                         "% no HC neighbors": [100*np.count_nonzero(sample.get_sample() == 0)/sample.get_sample_size() for sample in
                                                               all_samples_list],
                                         "% 1 HC neighbor": [100*np.count_nonzero(sample.get_sample() == 1)/sample.get_sample_size() for sample in
                                                             all_samples_list],
                                         "% 2 HC neighbors": [100*np.count_nonzero(sample.get_sample() == 2)/sample.get_sample_size() for sample in
                                                               all_samples_list],
                                         "% 3 HC neighbors": [100*np.count_nonzero(sample.get_sample() == 3)/sample.get_sample_size() for sample in
                                                              all_samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall HC neighbors data")
            diff_percentage.to_excel(writer, sheet_name="Differentiating Percentage data")
        for sample in all_samples_list:
            sample.save_to_excel(out_path, "# HC neighbors")
    plt.show()

def compare_E17_P0_rho_inhibition_contact_length(raw_data_output_folder=None):
    E17_diff_rho = DataCollector("Rho i E17.5 differentiating cells", Rho_inhibition_E17_folders,
                             ["contact_length_by_type_differentiation_data"]*3,
                             ["HC contact length"]*3, normalization=10)
    E17_ref_SC_rho = DataCollector("Rho i E17.5 reference SC +24h", Rho_inhibition_E17_folders,
                             ["contact_length_by_type_reference_SC_frame91_data",
                              "contact_length_by_type_reference_SC_frame96_data",
                              "contact_length_by_type_reference_SC_frame86_data"],
                             ["HC contact length"]*3, normalization=10)
    P0_diff_rho = DataCollector("Rho i P0 differentiating cells",
                            Rho_inhibition_P0_folders,
                            ["contact_length_by_type_differentiation_data"]*3,
                            ["HC contact length"]*3, normalization=10)
    P0_ref_SC_rho = DataCollector("Rho i P0 reference SC +24h",
                            Rho_inhibition_P0_folders,
                            ["contact_length_by_type_reference_SC_frame93_data"] + ["contact_length_by_type_reference_SC_frame96_data"]*2,
                            ["HC contact length"]*3, normalization=10)
    E17_diff = DataCollector("E17.5 differentiating cells", E17_folders,
                             ["contact_length_by_type_differentiation_data"] * 3,
                             ["HC contact length"] * 3, normalization=10)
    P0_diff = DataCollector("P0 differentiating cells", P0_folders,
                            ["contact_length_by_type_differentiation_data"] * 3,
                            ["HC contact length"] * 3, normalization=10)
    E17_ref_SC = DataCollector("E17.5 reference SC +24h", E17_folders,
                               ["contact_length_by_type_reference_SC_frame96_data",
                                "contact_length_by_type_reference_SC_frame97_data",
                                "contact_length_by_type_reference_SC_frame96_data"],
                               ["HC contact length"] * 3, normalization=10)
    P0_ref_SC = DataCollector("P0 reference SC +24h", P0_folders,
                              ["contact_length_by_type_reference_SC_frame96_data"] * 3,
                              ["HC contact length"] * 3, normalization=10)

    if raw_data_output_folder is not None:
        excel_path = os.path.join(raw_data_output_folder,
                                  "ROCK_inhibition_statistical_analysis.xlsx")
    else:
        excel_path = None
    samples_list = [E17_diff, P0_diff, E17_diff_rho, P0_diff_rho]
    pairs_to_compare = [(0, 1), (2, 3), (0, 2), (1,3)]
    color = ["cyan", "pink"] * 2
    edge_color = ["blue", "red", "blue", "red"]
    # samples_list = [E17_diff_rho, P0_diff_rho]
    # pairs_to_compare = [(0, 1)]
    color = ["cyan", "pink"]*2
    edge_color = ["blue", "red"]*2
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=True, show_N=True, hirarchical=True,
                                                      save_to_excel=excel_path, excel_sheet="HC contact len")
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                                      plot_style="violin", color=color,
                                                      edge_color=edge_color,
                                                      show_statistics=False, show_N=False, hirarchical=True, scatter=True)
    from matplotlib.ticker import MaxNLocator
    empty_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    empty_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # empty_ax.set_ylim([-0.75, 3.5])
    all_samples_list = [E17_diff_rho, P0_diff_rho, E17_ref_SC_rho, P0_ref_SC_rho, E17_diff, P0_diff, E17_ref_SC, P0_ref_SC]
    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "ROCK_inhibition_raw_data.xlsx")
        overall_raw_data = pd.DataFrame({"name": [sample.name for sample in all_samples_list],
                                         "Apical contact length with neighboring HCs average (um)": [sample.get_average_of_groups() for sample in all_samples_list],
                                         "Apical contact length with neighboring HCs SE (um)": [sample.get_se_of_groups() for sample in all_samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall HC contact len data")
        for sample in all_samples_list:
            sample.save_to_excel(out_path, "HC contact len")
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

def compare_distance_from_ablation(raw_data_output_folder=None):
    E17_diff = DataCollector("E17.5 differentiating cells",
                                   E17_ablation_folders,
                                   ["differentiation_distance_from_ablation_data"] + ["distance_from_ablation_differentiation_data"]*2,
                                   ["Distance from ablation"]*3, normalization=10)
    E17_ref_SC = DataCollector("E17.5 reference_SC",
                                     E17_ablation_folders,
                                     ["reference_SC_distance_from_ablation_data"] + ["distance_from_ablation_reference_SC_frame1_data"]*2,
                                     ["Distance from ablation"]*3, normalization=10)
    P0_diff = DataCollector("P0 differentiating cells",
                             P0_ablation_folders,
                             ["distance_from_ablation_differentiation_data"]*3,
                             ["Distance from ablation"]*3, normalization=10)
    P0_ref_SC = DataCollector("P0 reference_SC",
                               P0_ablation_folders,
                               ["distance_from_ablation_reference_SC_frame1_data"]*3,
                               ["Distance from ablation"]*3, normalization=10)

    if raw_data_output_folder is not None:
        excel_path = os.path.join(raw_data_output_folder,
                                  "distance_from_ablation_statistical_analysis.xlsx")
    else:
        excel_path = None
    samples_list = [E17_diff,  E17_ref_SC, P0_diff, P0_ref_SC]
    pairs_to_compare = [(0, 1), (2, 3)]
    full_fig, full_ax, res = compare_and_plot_samples(samples_list, pairs_to_compare, continues=True,
                                                      plot_style="violin", color=["cyan"] * 2 + ["pink"] * 2,
                                                      edge_color=["blue"] * 2 + ["red"] * 2,
                                                      show_statistics=True, show_N=True, hirarchical=True, scatter=True,
                                                      save_to_excel=excel_path, excel_sheet="dist from ablation"
                                                      )
    empty_fig, empty_ax, _ = compare_and_plot_samples(samples_list, pairs_to_compare, continues=False,
                                                      plot_style="violin", color=["cyan"] * 2 + ["pink"] * 2,
                                                      edge_color=["blue"] * 2 + ["red"] * 2,
                                                      show_statistics=False, show_N=False, scatter=True)
    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "distance_from_ablation_raw_data.xlsx")
        overall_raw_data = pd.DataFrame({"Name": [sample.name for sample in samples_list],
                                         "Distance from ablation average": [sample.get_average_of_groups() for sample in samples_list],
                                         "Distance from ablation SE": [sample.get_se_of_groups() for sample in samples_list],
                                         })
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall dist from ablation")
        for sample in samples_list:
            sample.save_to_excel(out_path, "Distance from ablation")
    plt.show()

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
    # Time in units of days
    E17_normal_time = np.array([48,46,30])/24
    P0_normal_time = np.array([35,36,32])/24
    E17_ablation_time = np.array([36,50,36])/24
    P0_ablation_time = np.array([44,49,48])/24
    E17_rho_time = np.array([67,48,48])/24
    P0_rho_time = np.array([48,48,47])/24

    # Area in units of (100um)^2
    E17_normal_area = np.array([32704, 30016,32832])/10000
    P0_normal_area = np.array([33042, 23361, 32280])/10000
    E17_ablation_area = np.array([25252, 32058, 27804])/10000
    P0_ablation_area = np.array([26645, 35369, 22941])/10000
    E17_rho_area = np.array([26923, 27638, 30466])/10000
    P0_rho_area = np.array([33055, 24908, 33298])/10000

    E17_normal_diff = DataCollector("E17.5 normal differentiation events", sample=np.array([32, 27, 59])/(E17_normal_time*E17_normal_area))
    E17_normal_div = DataCollector("E17.5 normal division events", sample=np.array([24, 28, 21])/(E17_normal_time*E17_normal_area))
    E17_normal_del = DataCollector("E17.5 normal delamination events", sample=np.array([30, 3, 18])/(E17_normal_time*E17_normal_area))
    P0_normal_diff = DataCollector("P0 normal differentiation events", sample=np.array([14, 9, 4])/(P0_normal_time*P0_normal_area))
    P0_normal_div = DataCollector("P0 normal division events", sample=np.array([0, 2, 0])/(P0_normal_time*P0_normal_area))
    P0_normal_del = DataCollector("P0 normal delamination events", sample=np.array([0, 10, 1])/(P0_normal_time*P0_normal_area))

    E17_ablation_diff = DataCollector("E17.5 ablation differentiation events", sample=np.array([42, 67, 37])/(E17_ablation_time*E17_ablation_area))
    E17_ablation_div = DataCollector("E17.5 ablation division events", sample=np.array([33, 82, 18])/(E17_ablation_time*E17_ablation_area))
    E17_ablation_del = DataCollector("E17.5 ablation delamination events", sample=np.array([16, 62, 14])/(E17_ablation_time*E17_ablation_area))
    P0_ablation_diff = DataCollector("P0 ablation differentiation events", sample=np.array([25, 24, 9])/(P0_ablation_time*P0_ablation_area))
    P0_ablation_div = DataCollector("P0 ablation division events", sample=np.array([2, 1, 1])/(P0_ablation_time*P0_ablation_area))
    P0_ablation_del = DataCollector("P0 ablation delamination events", sample=np.array([20, 2, 4])/(P0_ablation_time*P0_ablation_area))

    E17_rho_diff = DataCollector("E17.5 rho_ differentiation events", sample=np.array([29, 51, 29])/E17_rho_time)
    E17_rho_div = DataCollector("E17.5 rho_ division events", sample=np.array([8, 5, 1])/E17_rho_time)
    E17_rho_del = DataCollector("E17.5 rho_ delamination events", sample=np.array([36, 63, 3])/E17_rho_time)
    P0_rho_diff = DataCollector("P0 rho_ differentiation events", sample=np.array([4, 21, 15])/P0_rho_time)
    P0_rho_div = DataCollector("P0 rho_ division events", sample=np.array([0, 0, 0])/P0_rho_time)
    P0_rho_del = DataCollector("P0 rho_ delamination events", sample=np.array([21, 6, 17])/P0_rho_time)

    E17_normal = [E17_normal_diff, E17_normal_div, E17_normal_del]
    E17_ablation = [E17_ablation_diff, E17_ablation_div,E17_ablation_del]
    P0_normal = [P0_normal_diff, P0_normal_div,P0_normal_del]
    P0_ablation = [P0_ablation_diff, P0_ablation_div, P0_ablation_del]
    E17_normal_averages = [s.get_average() for s in E17_normal]
    P0_normal_averages = [s.get_average() for s in P0_normal]
    E17_ablation_averages = [s.get_average() for s in E17_ablation]
    P0_ablation_averages = [s.get_average() for s in P0_ablation]

    E17_samples = [E17_normal_diff, E17_ablation_diff, E17_normal_div, E17_ablation_div, E17_normal_del, E17_ablation_del]
    P0_samples = [P0_normal_diff, P0_ablation_diff, P0_normal_div, P0_ablation_div, P0_normal_del, P0_ablation_del]
    E17_averages = [s.get_average() for s in E17_samples]
    P0_averages = [s.get_average() for s in P0_samples]

    plt.bar(2*np.arange(len(E17_normal_averages)) - 0.125, E17_normal_averages, width=0.25, color="cyan",
            edgecolor="blue")
    plt.bar(2*np.arange(len(E17_ablation_averages)) + 0.125, E17_ablation_averages, width=0.25, color="cyan", edgecolor="blue", hatch='/')

    plt.bar(1+2 * np.arange(len(P0_normal_averages)) - 0.125, P0_normal_averages, width=0.25, color="pink",
            edgecolor="red")
    plt.bar(1+2 * np.arange(len(P0_ablation_averages)) + 0.125, P0_ablation_averages, width=0.25, color="pink", edgecolor="red",  hatch='/')

    for i in range(len(E17_normal)):
        plt.scatter(np.full_like(E17_normal[i].get_sample(), 2*i) - 0.125, E17_normal[i].get_sample(), marker=".", s=30,
                    color="black")
        plt.scatter(np.full_like(E17_ablation[i].get_sample(), 2 * i) + 0.125, E17_ablation[i].get_sample(), marker=".",
                    s=30,
                    color="black")
    for i in range(len(P0_normal)):
        plt.scatter(np.full_like(P0_normal[i].get_sample(), 1+2*i) - 0.125, P0_normal[i].get_sample(), marker=".", s=30,
                    color="black")
        plt.scatter(np.full_like(P0_ablation[i].get_sample(), 1+2*i) + 0.125, P0_ablation[i].get_sample(), marker=".", s=30,
                    color="black")
    from statistical_analysis import TwoByTwoCompare
    samples = E17_samples + P0_samples
    groups_to_compare = [(i, i+1, i+len(E17_samples), i+len(E17_samples) + 1) for i in np.arange(start=0, stop=len(E17_samples), step=2)]
    labels = ["Differentiation", "Divisions", "Delaminations"]
    for i in range(len(groups_to_compare)):
        group = groups_to_compare[i]
        A1B1 = samples[group[0]].get_sample()
        A1B2 = samples[group[1]].get_sample()
        A2B1 = samples[group[2]].get_sample()
        A2B2 = samples[group[3]].get_sample()
        comparer = TwoByTwoCompare(A1B1, A1B2, A2B1, A2B2,
                 factorA_name="Stage",
                 factorB_name="Ablation",
                 A_levels=("E17.5", "P0"),
                 B_levels=("Normal development", "HC ablation"))
        res = comparer.compare(verbose=True, save_to_excel=os.path.join(RAW_DATA_FOLDER, "events_counting_statistical_analysis.xlsx"),
                               label=labels[i], sheet="Events counting")


    plt.ylabel("# Events per (100um)^2 per day", fontsize=20)
    # plt.ylim([0, 22])
    plt.xticks(np.arange(len(P0_averages)), labels=np.arange(len(P0_averages)), fontsize=20)

    # plt.yticks(fontsize=20, ticks=[0, 5, 10, 15, 20])
    plt.ylim([0, 16])
    plt.tight_layout()
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

def plot_rho_inhibition_HC_SC_roundness(raw_data_output_folder=None):
    E17_rho_0h_HC_roundness = DataCollector("E17.5 rho +0h HC roundness", Rho_inhibition_E17_folders,
                                               ["area_and_roundness_reference_HC_frame1_data"]*3,
                                               ["roundness"] * 3, normalization=1)
    E17_rho_12h_HC_roundness = DataCollector("E17.5 rho +12h HC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_HC_frame48_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    E17_rho_24h_HC_roundness = DataCollector("E17.5 rho +24h HC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_HC_frame96_data"] * 2 +
                                             ["area_and_roundness_reference_HC_frame86_data"],
                                            ["roundness"] * 3, normalization=1)
    E17_rho_36h_HC_roundness = DataCollector("E17.5 rho +36h HC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_HC_frame144_data"] * 2 +
                                             ["area_and_roundness_reference_HC_frame121_data"],
                                            ["roundness"] * 3, normalization=1)
    E17_rho_48h_HC_roundness = DataCollector("E17.5 rho +48h HC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_HC_frame192_data",
                                             "area_and_roundness_reference_HC_frame191_data",
                                             "area_and_roundness_reference_HC_frame169_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_rho_0h_HC_roundness = DataCollector("P0 rho +0h HC roundness", Rho_inhibition_P0_folders,
                                            ["area_and_roundness_reference_HC_frame1_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    P0_rho_12h_HC_roundness = DataCollector("P0 rho +12h HC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_HC_frame48_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_24h_HC_roundness = DataCollector("P0 rho +24h HC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_HC_frame96_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_36h_HC_roundness = DataCollector("P0 rho +36h HC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_HC_frame144_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_48h_HC_roundness = DataCollector("P0 rho +48h HC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_HC_frame191_data",
                                              "area_and_roundness_reference_HC_frame192_data",
                                               "area_and_roundness_reference_HC_frame180_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_0h_HC_roundness = DataCollector("E17.5 +0h HC roundness", E17_folders,
                                            ["area_and_roundness_reference_HC_frame1_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    E17_12h_HC_roundness = DataCollector("E17.5 +12h HC roundness", E17_folders,
                                             ["area_and_roundness_reference_HC_frame48_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    E17_24h_HC_roundness = DataCollector("E17.5 +24h HC roundness", E17_folders,
                                             ["area_and_roundness_reference_HC_frame96_data",
                                              "area_and_roundness_reference_HC_frame97_data",
                                              "area_and_roundness_reference_HC_frame96_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_36h_HC_roundness = DataCollector("E17.5 +36h HC roundness", E17_folders,
                                          ["area_and_roundness_reference_HC_frame144_data"] * 2+
                                         ["area_and_roundness_reference_HC_frame120_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_48h_HC_roundness = DataCollector("E17.5 +48h HC roundness", E17_folders[:2],
                                             ["area_and_roundness_reference_HC_frame191_data",
                                              "area_and_roundness_reference_HC_frame199_data"],
                                             ["roundness"] * 2, normalization=1)
    P0_0h_HC_roundness = DataCollector("P0 +0h HC roundness", P0_folders,
                                           ["area_and_roundness_reference_HC_frame1_data"] * 3,
                                           ["roundness"] * 3, normalization=1)
    P0_12h_HC_roundness = DataCollector("P0 +12h HC roundness", P0_folders,
                                            ["area_and_roundness_reference_HC_frame48_data"] * 2+
                                            ["area_and_roundness_reference_HC_frame46_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_24h_HC_roundness = DataCollector("P0 +24h HC roundness", P0_folders,
                                            ["area_and_roundness_reference_HC_frame96_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    P0_36h_HC_roundness = DataCollector("P0 +36h HC roundness", P0_folders,
                                            ["area_and_roundness_reference_HC_frame144_data"] * 2+
                                            ["area_and_roundness_reference_HC_frame130_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_48h_HC_roundness = DataCollector("P0 +48h HC roundness", [P0_folders[0]],
                                            ["area_and_roundness_reference_HC_frame165_data"],
                                            ["roundness"] * 1, normalization=1)
    E17_HC_rho_samples = [E17_rho_0h_HC_roundness,
                    E17_rho_12h_HC_roundness,
                    E17_rho_24h_HC_roundness,
                    E17_rho_36h_HC_roundness,
                    E17_rho_48h_HC_roundness]
    P0_HC_rho_samples = [P0_rho_0h_HC_roundness,
                    P0_rho_12h_HC_roundness,
                    P0_rho_24h_HC_roundness,
                    P0_rho_36h_HC_roundness,
                    P0_rho_48h_HC_roundness]
    E17_HC_samples = [E17_0h_HC_roundness,
                   E17_12h_HC_roundness,
                   E17_24h_HC_roundness,
                   E17_36h_HC_roundness,
                   E17_48h_HC_roundness]
    P0_HC_samples = [P0_0h_HC_roundness,
                  P0_12h_HC_roundness,
                  P0_24h_HC_roundness,
                  P0_36h_HC_roundness,
                  P0_48h_HC_roundness]
    HC_samples_list = []
    for i in range(5):
        HC_samples_list.append(E17_HC_samples[i])
        HC_samples_list.append(E17_HC_rho_samples[i])
    for i in range(5):
        HC_samples_list.append(P0_HC_samples[i])
        HC_samples_list.append(P0_HC_rho_samples[i])
    # pairs_to_compare = [(2*i, 2*i+1) for i in range(10)]
    # color = ["cyan", "green"] * 5 + ["pink", "magenta"] * 5
    # edge_color = ["blue", "black"] * 5 + ["red", "purple"] * 5
    # full_fig, full_ax, res = compare_and_plot_samples(HC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=True, show_N=True, hirarchical=True)
    # empty_fig, empty_ax, _ = compare_and_plot_samples(HC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=False, show_N=False, hirarchical=True,
    #                                           scatter=True)

    E17_rho_0h_SC_roundness = DataCollector("E17.5 rho +0h SC roundness", Rho_inhibition_E17_folders,
                                               ["area_and_roundness_reference_SC_frame1_data"]*3,
                                               ["roundness"] * 3, normalization=1)
    E17_rho_12h_SC_roundness = DataCollector("E17.5 rho +12h SC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_SC_frame48_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    E17_rho_24h_SC_roundness = DataCollector("E17.5 rho +24h SC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_SC_frame96_data"] * 2 +
                                             ["area_and_roundness_reference_SC_frame86_data"],
                                            ["roundness"] * 3, normalization=1)
    E17_rho_36h_SC_roundness = DataCollector("E17.5 rho +36h SC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_SC_frame144_data"] * 2 +
                                             ["area_and_roundness_reference_SC_frame121_data"],
                                            ["roundness"] * 3, normalization=1)
    E17_rho_48h_SC_roundness = DataCollector("E17.5 rho +48h SC roundness", Rho_inhibition_E17_folders,
                                            ["area_and_roundness_reference_SC_frame192_data",
                                             "area_and_roundness_reference_SC_frame191_data",
                                             "area_and_roundness_reference_SC_frame169_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_rho_0h_SC_roundness = DataCollector("P0 rho +0h SC roundness", Rho_inhibition_P0_folders,
                                            ["area_and_roundness_reference_SC_frame1_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    P0_rho_12h_SC_roundness = DataCollector("P0 rho +12h SC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_SC_frame48_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_24h_SC_roundness = DataCollector("P0 rho +24h SC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_SC_frame96_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_36h_SC_roundness = DataCollector("P0 rho +36h SC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_SC_frame144_data", ] * 3,
                                             ["roundness"] * 3, normalization=1)
    P0_rho_48h_SC_roundness = DataCollector("P0 rho +48h SC roundness", Rho_inhibition_P0_folders,
                                             ["area_and_roundness_reference_SC_frame191_data",
                                              "area_and_roundness_reference_SC_frame192_data",
                                               "area_and_roundness_reference_SC_frame180_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_0h_SC_roundness = DataCollector("E17.5 +0h SC roundness", E17_folders,
                                            ["area_and_roundness_reference_SC_frame1_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    E17_12h_SC_roundness = DataCollector("E17.5 +12h SC roundness", E17_folders,
                                             ["area_and_roundness_reference_SC_frame48_data"] * 3,
                                             ["roundness"] * 3, normalization=1)
    E17_24h_SC_roundness = DataCollector("E17.5 +24h SC roundness", E17_folders,
                                             ["area_and_roundness_reference_SC_frame96_data",
                                              "area_and_roundness_reference_SC_frame97_data",
                                              "area_and_roundness_reference_SC_frame96_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_36h_SC_roundness = DataCollector("E17.5 +36h SC roundness", E17_folders,
                                          ["area_and_roundness_reference_SC_frame144_data"] * 2 +
                                         ["area_and_roundness_reference_SC_frame120_data"],
                                             ["roundness"] * 3, normalization=1)
    E17_48h_SC_roundness = DataCollector("E17.5 +48h SC roundness", E17_folders[:2],
                                             ["area_and_roundness_reference_SC_frame191_data",
                                              "area_and_roundness_reference_SC_frame199_data"],
                                             ["roundness"] * 2, normalization=1)
    P0_0h_SC_roundness = DataCollector("P0 +0h SC roundness", P0_folders,
                                           ["area_and_roundness_reference_SC_frame1_data"] * 3,
                                           ["roundness"] * 3, normalization=1)
    P0_12h_SC_roundness = DataCollector("P0 +12h SC roundness", P0_folders,
                                            ["area_and_roundness_reference_SC_frame48_data"] * 2+
                                            ["area_and_roundness_reference_SC_frame46_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_24h_SC_roundness = DataCollector("P0 +24h SC roundness", P0_folders,
                                            ["area_and_roundness_reference_SC_frame96_data"] * 3,
                                            ["roundness"] * 3, normalization=1)
    P0_36h_SC_roundness = DataCollector("P0 +36h SC roundness", P0_folders,
                                        ["area_and_roundness_reference_SC_frame144_data"] * 2 +
                                        ["area_and_roundness_reference_SC_frame130_data"],
                                            ["roundness"] * 3, normalization=1)
    P0_48h_SC_roundness = DataCollector("P0 +48h SC roundness", [P0_folders[0]],
                                            ["area_and_roundness_reference_SC_frame165_data"],
                                            ["roundness"] * 1, normalization=1)
    E17_SC_rho_samples = [E17_rho_0h_SC_roundness,
                    E17_rho_12h_SC_roundness,
                    E17_rho_24h_SC_roundness,
                    E17_rho_36h_SC_roundness,
                    E17_rho_48h_SC_roundness]
    P0_SC_rho_samples = [P0_rho_0h_SC_roundness,
                    P0_rho_12h_SC_roundness,
                    P0_rho_24h_SC_roundness,
                    P0_rho_36h_SC_roundness,
                    P0_rho_48h_SC_roundness]
    E17_SC_samples = [E17_0h_SC_roundness,
                   E17_12h_SC_roundness,
                   E17_24h_SC_roundness,
                   E17_36h_SC_roundness,
                   E17_48h_SC_roundness]
    P0_SC_samples = [P0_0h_SC_roundness,
                  P0_12h_SC_roundness,
                  P0_24h_SC_roundness,
                  P0_36h_SC_roundness,
                  P0_48h_SC_roundness]
    # E17_24h_SC_roundness.save_sample(out_path=output_dir)
    # E17_24h_HC_roundness.save_sample(out_path=output_dir)
    # P0_24h_SC_roundness.save_sample(out_path=output_dir)
    # P0_24h_HC_roundness.save_sample(out_path=output_dir)
    SC_samples_list = []
    for i in range(5):
        SC_samples_list.append(E17_SC_samples[i])
        SC_samples_list.append(E17_SC_rho_samples[i])
    for i in range(5):
        SC_samples_list.append(P0_SC_samples[i])
        SC_samples_list.append(P0_SC_rho_samples[i])
    # pairs_to_compare = [(2 * i, 2 * i + 1) for i in range(10)]
    # color = ["cyan", "green"] * 5 + ["pink", "magenta"] * 5
    # edge_color = ["blue", "black"] * 5 + ["red", "purple"] * 5
    # full_fig, full_ax, res = compare_and_plot_samples(SC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=True, show_N=True, hirarchical=True)
    # empty_fig, empty_ax, _ = compare_and_plot_samples(SC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=False, show_N=False, hirarchical=True,
    #                                                   scatter=True)
    avg_SC_fig, avg_SC_ax = plt.subplots()
    avg_HC_fig, avg_HC_ax = plt.subplots()
    avg_ratio_fig, avg_ratio_ax = plt.subplots()
    fmt_list = ["o-b", "o-r", "o--b", "o--r"]
    color_list = ["blue", "red"] * 2
    x = [0, 12, 24, 36, 48]
    i=0
    overall_raw_data = pd.DataFrame()
    labels = ["E17.5 ROCK inhibition", "P0 ROCK inhibition", "E17.5 control", "P0 control"]
    for HC_sample_list, SC_sample_list in zip([E17_HC_rho_samples, P0_HC_rho_samples, E17_HC_samples, P0_HC_samples],
                                              [E17_SC_rho_samples, P0_SC_rho_samples, E17_SC_samples, P0_SC_samples]):
        HC_avg = []
        HC_se = []
        SC_avg = []
        SC_se = []
        ratio_avg = []
        ratio_se = []
        fmt = fmt_list[i]
        for j in range(len(HC_sample_list)):
            HC_avg.append(HC_sample_list[j].get_average_of_groups())
            HC_se.append(HC_sample_list[j].get_se_of_groups())
            SC_avg.append(SC_sample_list[j].get_average_of_groups())
            SC_se.append(SC_sample_list[j].get_se_of_groups())
            ratio_avg.append(HC_sample_list[j].get_average_of_groups()/SC_sample_list[j].get_average_of_groups())
            ratio_se.append(np.sqrt((HC_sample_list[j].get_se_of_groups()/SC_sample_list[j].get_average_of_groups())**2 +
                                    (HC_sample_list[j].get_average_of_groups()*HC_sample_list[j].get_se_of_groups()/SC_sample_list[j].get_average_of_groups())**2))
        HC_avg = np.array(HC_avg)
        HC_se = np.array(HC_se)
        SC_avg = np.array(SC_avg)
        SC_se = np.array(SC_se)
        ratio_avg = np.array(ratio_avg)
        ratio_se = np.array(ratio_se)
        HC_avg = HC_avg[HC_se > 0]
        HC_se = HC_se[HC_se > 0]
        SC_avg = SC_avg[SC_se > 0]
        SC_se = SC_se[SC_se > 0]
        ratio_avg = ratio_avg[ratio_se > 0]
        ratio_se = ratio_se[ratio_se > 0]
        avg_HC_ax.plot(x[:HC_avg.size], HC_avg, fmt)
        avg_HC_ax.fill_between(x[:HC_avg.size], HC_avg - HC_se, HC_avg + HC_se, color=color_list[i], alpha=0.2, linewidth=0)
        avg_SC_ax.plot(x[:SC_avg.size], SC_avg, fmt)
        avg_SC_ax.fill_between(x[:SC_avg.size], SC_avg - SC_se, SC_avg + SC_se, color=color_list[i], alpha=0.2, linewidth=0)
        avg_ratio_ax.plot(x[:ratio_avg.size], ratio_avg, fmt)
        avg_ratio_ax.fill_between(x[:ratio_avg.size], ratio_avg - ratio_se, ratio_avg + ratio_se, color=color_list[i], alpha=0.2, linewidth=0)
        if raw_data_output_folder is not None:
            overall_raw_data_line = pd.DataFrame({"name": labels[i],
                                                  "Initial HC roundness average": HC_avg[0],
                                                  "HC roundness after 12 hours average": HC_avg[1],
                                                  "HC roundness after 24 hours average": HC_avg[2],
                                                  "HC roundness after 36 hours average": HC_avg[3],
                                                  "HC roundness after 48 hours average": HC_avg[4],
                                                  "Initial HC roundness SE": HC_se[0],
                                                  "HC roundness after 12 hours SE": HC_se[1],
                                                  "HC roundness after 24 hours SE": HC_se[2],
                                                  "HC roundness after 36 hours SE": HC_se[3],
                                                  "HC roundness after 48 hours SE": HC_se[4],
                                                  "Initial SC roundness average": SC_avg[0],
                                                  "SC roundness after 12 hours average": SC_avg[1],
                                                  "SC roundness after 24 hours average": SC_avg[2],
                                                  "SC roundness after 36 hours average": SC_avg[3],
                                                  "SC roundness after 48 hours average": SC_avg[4],
                                                  "Initial SC roundness SE": SC_se[0],
                                                  "SC roundness after 12 hours SE": SC_se[1],
                                                  "SC roundness after 24 hours SE": SC_se[2],
                                                  "SC roundness after 36 hours SE": SC_se[3],
                                                  "SC roundness after 48 hours SE": SC_se[4],
                                                  "Initial HC/SC roundness ratio average": ratio_avg[0],
                                                  "HC/SC roundness ratio after 12 hours average": ratio_avg[1],
                                                  "HC/SC roundness ratio after 24 hours average": ratio_avg[2],
                                                  "HC/SC roundness ratio after 36 hours average": ratio_avg[3],
                                                  "HC/SC roundness ratio after 48 hours average": ratio_avg[4],
                                                  "Initial HC/SC roundness ratio SE": ratio_se[0],
                                                  "HC/SC roundness ratio after 12 hours SE": ratio_se[1],
                                                  "HC/SC roundness ratio after 24 hours SE": ratio_se[2],
                                                  "HC/SC roundness ratio after 36 hours SE": ratio_se[3],
                                                  "HC/SC roundness ratio after 48 hours SE": ratio_se[4]
                                             }, index=[i])
            overall_raw_data = pd.concat([overall_raw_data, overall_raw_data_line])
        i+=1
    avg_HC_ax.set_xticks(ticks=[0, 12, 24, 36, 48])
    avg_SC_ax.set_xticks(ticks=[0, 12, 24, 36, 48])
    avg_ratio_ax.set_xticks(ticks=[0, 12, 24, 36, 48])
    avg_HC_ax.set_ylabel("Average HC roundness")
    avg_HC_ax.set_xlabel("Time (hours)")
    avg_SC_ax.set_ylabel("Average SC roundness")
    avg_SC_ax.set_xlabel("Time (hours)")
    avg_ratio_ax.set_ylabel("Average HC roundness/Average SC roundness")
    avg_ratio_ax.set_xlabel("Time (hours)")
    if raw_data_output_folder is not None:
        out_path = os.path.join(raw_data_output_folder, "ROCK_inhibition_roundness_raw_data.xlsx")
        with pd.ExcelWriter(out_path, mode="w") as writer:
            overall_raw_data.to_excel(writer, sheet_name="Overall roundness data")
        all_samples_list = E17_HC_rho_samples + P0_HC_rho_samples + E17_HC_samples + P0_HC_samples + \
                           E17_SC_rho_samples + P0_SC_rho_samples + E17_SC_samples + P0_SC_samples
        for sample in all_samples_list:
            sample.save_to_excel(out_path, "Roundness")
    # E17_rho_0h_HC_roundness_ratio = DataCollector("E17.5 rho +0h HC roundness ratio", Rho_inhibition_E17_folders,
    #                                         ["area_and_roundness_reference_HC_frame1_data"] * 3,
    #                                         ["roundness"] * 3, normalization=E17_rho_0h_SC_roundness.get_group_avg())
    # E17_rho_12h_HC_roundness_ratio = DataCollector("E17.5 rho +12h HC roundness", Rho_inhibition_E17_folders,
    #                                          ["area_and_roundness_reference_HC_frame48_data"] * 3,
    #                                          ["roundness"] * 3, normalization=E17_rho_12h_SC_roundness.get_group_avg())
    # E17_rho_24h_HC_roundness_ratio = DataCollector("E17.5 rho +24h HC roundness ratio", Rho_inhibition_E17_folders,
    #                                          ["area_and_roundness_reference_HC_frame96_data"] * 2 +
    #                                          ["area_and_roundness_reference_HC_frame86_data"],
    #                                          ["roundness"] * 3, normalization=E17_rho_24h_SC_roundness.get_group_avg())
    # E17_rho_36h_HC_roundness_ratio = DataCollector("E17.5 rho +36h HC roundness ratio", Rho_inhibition_E17_folders,
    #                                          ["area_and_roundness_reference_HC_frame144_data"] * 2 +
    #                                          ["area_and_roundness_reference_HC_frame121_data"],
    #                                          ["roundness"] * 3, normalization=E17_rho_36h_SC_roundness.get_group_avg())
    # E17_rho_48h_HC_roundness_ratio = DataCollector("E17.5 rho +48h HC roundness ratio", Rho_inhibition_E17_folders,
    #                                          ["area_and_roundness_reference_HC_frame192_data",
    #                                           "area_and_roundness_reference_HC_frame191_data",
    #                                           "area_and_roundness_reference_HC_frame169_data"],
    #                                          ["roundness"] * 3, normalization=E17_rho_48h_SC_roundness.get_group_avg())
    # P0_rho_0h_HC_roundness_ratio = DataCollector("P0 rho +0h HC roundness ratio", Rho_inhibition_P0_folders,
    #                                        ["area_and_roundness_reference_HC_frame1_data"] * 3,
    #                                        ["roundness"] * 3, normalization=P0_rho_0h_SC_roundness.get_group_avg())
    # P0_rho_12h_HC_roundness_ratio = DataCollector("P0 rho +12h HC roundness ratio", Rho_inhibition_P0_folders,
    #                                         ["area_and_roundness_reference_HC_frame48_data"] * 3,
    #                                         ["roundness"] * 3, normalization=P0_rho_12h_SC_roundness.get_group_avg())
    # P0_rho_24h_HC_roundness_ratio = DataCollector("P0 rho +24h HC roundness", Rho_inhibition_P0_folders,
    #                                         ["area_and_roundness_reference_HC_frame96_data"] * 3,
    #                                         ["roundness"] * 3, normalization=P0_rho_24h_SC_roundness.get_group_avg())
    # P0_rho_36h_HC_roundness_ratio = DataCollector("P0 rho +36h HC roundness ratio", Rho_inhibition_P0_folders,
    #                                         ["area_and_roundness_reference_HC_frame144_data"] * 3,
    #                                         ["roundness"] * 3, normalization=P0_rho_36h_SC_roundness.get_group_avg())
    # P0_rho_48h_HC_roundness_ratio = DataCollector("P0 rho +48h HC roundness ratio", Rho_inhibition_P0_folders,
    #                                         ["area_and_roundness_reference_HC_frame191_data",
    #                                          "area_and_roundness_reference_HC_frame192_data",
    #                                          "area_and_roundness_reference_HC_frame180_data"],
    #                                         ["roundness"] * 3, normalization=P0_rho_48h_SC_roundness.get_group_avg())
    # E17_0h_HC_roundness_ratio = DataCollector("E17.5 +0h HC roundness ratio", E17_folders,
    #                                     ["area_and_roundness_reference_HC_frame1_data"] * 3,
    #                                     ["roundness"] * 3, normalization=E17_0h_SC_roundness.get_group_avg())
    # E17_12h_HC_roundness_ratio = DataCollector("E17.5 +12h HC roundness ratio", E17_folders,
    #                                      ["area_and_roundness_reference_HC_frame48_data"] * 3,
    #                                      ["roundness"] * 3, normalization=E17_12h_SC_roundness.get_group_avg())
    # E17_24h_HC_roundness_ratio = DataCollector("E17.5 +24h HC roundness ratio", E17_folders,
    #                                      ["area_and_roundness_reference_HC_frame96_data",
    #                                       "area_and_roundness_reference_HC_frame97_data",
    #                                       "area_and_roundness_reference_HC_frame96_data"],
    #                                      ["roundness"] * 3, normalization=E17_24h_SC_roundness.get_group_avg())
    # E17_36h_HC_roundness_ratio = DataCollector("E17.5 +36h HC roundness ratio", E17_folders,
    #                                      ["area_and_roundness_reference_HC_frame144_data"] * 2 +
    #                                      ["area_and_roundness_reference_HC_frame120_data"],
    #                                      ["roundness"] * 3, normalization=E17_36h_SC_roundness.get_group_avg())
    # E17_48h_HC_roundness_ratio = DataCollector("E17.5 +48h HC roundness ratio", E17_folders[:1],
    #                                      ["area_and_roundness_reference_HC_frame191_data",
    #                                       "area_and_roundness_reference_HC_frame200_data"],
    #                                      ["roundness"] * 2, normalization=E17_48h_SC_roundness.get_group_avg())
    # P0_0h_HC_roundness_ratio = DataCollector("P0 +0h HC roundness ratio", P0_folders,
    #                                    ["area_and_roundness_reference_HC_frame1_data"] * 3,
    #                                    ["roundness"] * 3, normalization=P0_0h_SC_roundness.get_group_avg())
    # P0_12h_HC_roundness_ratio = DataCollector("P0 +12h HC roundness ratio", P0_folders,
    #                                     ["area_and_roundness_reference_HC_frame48_data"] * 2 +
    #                                     ["area_and_roundness_reference_HC_frame46_data"],
    #                                     ["roundness"] * 3, normalization=P0_12h_SC_roundness.get_group_avg())
    # P0_24h_HC_roundness_ratio = DataCollector("P0 +24h HC roundness ratio", P0_folders,
    #                                     ["area_and_roundness_reference_HC_frame96_data"] * 3,
    #                                     ["roundness"] * 3, normalization=P0_24h_SC_roundness.get_group_avg())
    # P0_36h_HC_roundness_ratio = DataCollector("P0 +36h HC roundness ratio", P0_folders,
    #                                     ["area_and_roundness_reference_HC_frame144_data"] * 2 +
    #                                     ["area_and_roundness_reference_HC_frame130_data"],
    #                                     ["roundness"] * 3, normalization=P0_36h_SC_roundness.get_group_avg())
    # P0_48h_HC_roundness_ratio = DataCollector("P0 +48h HC roundness ratio", [P0_folders[0]],
    #                                     ["area_and_roundness_reference_HC_frame165_data"],
    #                                     ["roundness"] * 1, normalization=P0_48h_SC_roundness.get_group_avg())
    # E17_HC_rho_samples = [E17_rho_0h_HC_roundness_ratio,
    #                       E17_rho_12h_HC_roundness_ratio,
    #                       E17_rho_24h_HC_roundness_ratio,
    #                       E17_rho_36h_HC_roundness_ratio,
    #                       E17_rho_48h_HC_roundness_ratio]
    # P0_HC_rho_samples = [P0_rho_0h_HC_roundness_ratio,
    #                      P0_rho_12h_HC_roundness_ratio,
    #                      P0_rho_24h_HC_roundness_ratio,
    #                      P0_rho_36h_HC_roundness_ratio,
    #                      P0_rho_48h_HC_roundness_ratio]
    # E17_HC_samples = [E17_0h_HC_roundness_ratio,
    #                   E17_12h_HC_roundness_ratio,
    #                   E17_24h_HC_roundness_ratio,
    #                   E17_36h_HC_roundness_ratio,
    #                   E17_48h_HC_roundness_ratio]
    # P0_HC_samples = [P0_0h_HC_roundness_ratio,
    #                  P0_12h_HC_roundness_ratio,
    #                  P0_24h_HC_roundness_ratio,
    #                  P0_36h_HC_roundness_ratio,
    #                  P0_48h_HC_roundness_ratio]
    # HC_samples_list = []
    # for i in range(5):
    #     HC_samples_list.append(E17_HC_samples[i])
    #     HC_samples_list.append(E17_HC_rho_samples[i])
    # for i in range(5):
    #     HC_samples_list.append(P0_HC_samples[i])
    #     HC_samples_list.append(P0_HC_rho_samples[i])
    # pairs_to_compare = [(2 * i, 2 * i + 1) for i in range(10)]
    # color = ["cyan", "green"] * 5 + ["pink", "magenta"] * 5
    # edge_color = ["blue", "black"] * 5 + ["red", "purple"] * 5
    # full_fig, full_ax, res = compare_and_plot_samples(HC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=True, show_N=True, hirarchical=True)
    # empty_fig, empty_ax, _ = compare_and_plot_samples(HC_samples_list, pairs_to_compare, continues=True,
    #                                                   plot_style="violin", color=color,
    #                                                   edge_color=edge_color,
    #                                                   show_statistics=False, show_N=False, hirarchical=True,
    #                                                   scatter=True)
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


    ### Data to compare with model ###

    compare_E17_P0_HC_neighbors_with_model()




    ## CREATE PAPER FIGURES ##
    # Figure S1 #
    # plot_number_of_events()

    # Figure S2 #
    # compare_E17_E19_and_P0_P2_neighbors(raw_data_output_folder=RAW_DATA_FOLDER)
    # compare_E17_E19_and_P0_P2_contact_length(raw_data_output_folder=RAW_DATA_FOLDER)

    # Figure2 #
    # compare_E17_P0_HC_neighbors_for_differentiation_and_trans_differentiation(raw_data_output_folder=RAW_DATA_FOLDER)
    # compare_distance_from_ablation(raw_data_output_folder=RAW_DATA_FOLDER)

    # Figure S3 #
    # compare_E17_P0_HC_contact_length_for_differentiation_and_trans_differentiation(raw_data_output_folder=RAW_DATA_FOLDER)

    # Figure 3 and S4 #
    # fit_circular_ablation_results_to_circle(E17_circular_ablation_folders, P0_circular_ablation_folders, 60, raw_data_output_folder=RAW_DATA_FOLDER)

    # Figure 4 #
    # compare_E17_P0_rho_inhibition_neighbors_by_type(raw_data_output_folder=RAW_DATA_FOLDER)

    # Figure S4 #
    # plot_rho_inhibition_HC_SC_roundness(raw_data_output_folder=None)
    # compare_E17_P0_rho_inhibition_contact_length(raw_data_output_folder=RAW_DATA_FOLDER)