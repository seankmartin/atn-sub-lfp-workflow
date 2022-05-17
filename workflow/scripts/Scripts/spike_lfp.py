from copy import deepcopy
import os

import simuran
import numpy as np
import matplotlib.pyplot as plt

from lfp_atn_simuran.Scripts.lfp_clean import LFPClean
from skm_pyutils.py_table import list_to_df, df_from_file, df_to_file
from skm_pyutils.py_plot import UnicodeGrabber

here = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.abspath(os.path.join(here, "..", "sim_results", "spike_lfp"))


def plot_phase(graph_data):
    phBins = graph_data["phBins"]
    phCount = graph_data["phCount"]

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.bar(
        phBins * np.pi / 180,
        phCount,
        width=3 * np.pi / 180,
        color="k",
        alpha=0.8,
        bottom=np.max(phCount) / 2,
        rasterized=True,
    )
    ax.plot(
        [0, graph_data["meanTheta"]],
        [0, 1.5 * np.max(phCount)],
        linewidth=3,
        color="red",
        linestyle="--",
    )
    plt.title("LFP phase distribution (red= mean direction)")

    return fig


def spike_lfp_headings():
    headers = [
        "Mean_Phase_SUB",
        "Mean_Phase_Count_SUB",
        "Resultant_Phase_Vector_SUB",
        "Phase_Vector_95_SUB",
        "Mean_Phase_RSC",
        "Mean_Phase_Count_RSC",
        "Resultant_Phase_Vector_RSC",
        "Phase_Vector_95_RSC",
        "Theta_SFC_SUB",
        "Theta_SFC_RSC",
        "STA_SUB",
        "SFC_SUB",
        "STA_RSC",
        "SFC_RSC",
        "Time",
        "Frequency",
        "RandomSFC_SUB",
        "RandomSFC_RSC",
        "Spatial",
    ]
    return headers


def recording_spike_lfp(recording, clean_method="avg", **kwargs):
    clean_kwargs = kwargs.get("clean_kwargs", {})
    lc = LFPClean(method=clean_method, visualise=False)
    fmin = 0
    fmax = 100
    for i in range(len(recording.signals)):
        recording.signals[i].load()
    signals_grouped_by_region = lc.clean(
        recording.signals, fmin, fmax, method_kwargs=clean_kwargs
    )["signals"]

    simuran.set_plot_style()
    fmt = kwargs.get("image_format", "png")
    fwin = (kwargs.get("theta_min", 6), kwargs.get("theta_max", 10))
    base_dir = kwargs.get("cfg_base_dir")
    name_start = recording.get_name_for_save(base_dir)
    os.makedirs(output_dir, exist_ok=True)

    sub_sig = signals_grouped_by_region["SUB"]
    nc_sig = sub_sig.to_neurochat()

    if "RSC" in signals_grouped_by_region.keys():
        rsc_sig = signals_grouped_by_region["RSC"]
        nc_sig2 = rsc_sig.to_neurochat()
    else:
        nc_sig2 = None

    NUM_RESULTS = len(spike_lfp_headings())

    output = {}
    # To avoid overwriting what has been set to analyse
    all_analyse = deepcopy(recording.get_set_units())

    # Unit contains probe/tetrode info, to_analyse are list of cells
    for unit, to_analyse in zip(recording.units, all_analyse):

        # Two cases for empty list of cells
        if to_analyse is None:
            continue
        if len(to_analyse) == 0:
            continue

        unit.load()
        # Loading can overwrite units_to_use, so reset these after load
        unit.units_to_use = to_analyse
        out_str_start = str(unit.group)
        no_data_loaded = unit.underlying is None
        available_units = unit.underlying.get_unit_list()

        for cell in to_analyse:
            name_for_save = out_str_start + "_" + str(cell)
            output[name_for_save] = [np.nan] * NUM_RESULTS

            # Check to see if this data is ok
            if no_data_loaded:
                continue
            if cell not in available_units:
                continue

            spatial = unit.info.get(cell, {})
            spatial = spatial.get("class", "")
            if spatial.startswith("S"):
                spatial = "Spatial"
            elif spatial.startswith("NS"):
                spatial = "NS"
            else:
                spatial = "Unknown"

            unit.underlying.set_unit_no(cell)
            spike_train = unit.underlying.get_unit_stamp()

            # Do analysis on that unit
            g_data = nc_sig.plv(spike_train, mode="bs", fwin=[0, 20])
            sta = g_data["STAm"]
            sfc = g_data["SFCm"]
            t = g_data["t"]
            f = g_data["f"]

            if nc_sig2 is not None:
                g_data = nc_sig2.plv(spike_train, mode="bs", fwin=[0, 20])
                sta_rsc = g_data["STAm"]
                sfc_rsc = g_data["SFCm"]
            else:
                sta_rsc = None
                sfc_rsc = None

            g_data = nc_sig.phase_dist(spike_train, fwin=fwin)
            mean_phase = nc_sig.get_results()["LFP Spike Mean Phase"]
            phase_count = nc_sig.get_results()["LFP Spike Mean Phase Count"]
            spike_phase_vect = nc_sig.get_results()["LFP Spike Phase Res Vect"]
            os.makedirs(os.path.join(output_dir, "spike_phase_plots"), exist_ok=True)
            name = os.path.join(
                output_dir,
                "spike_phase_plots",
                f"{name_start}_{name_for_save}_SUB_Phase.{fmt}",
            )
            fig = plot_phase(g_data)
            fig.savefig(name, dpi=400)
            plt.close(fig)

            if nc_sig2 is not None:
                g_data = nc_sig2.phase_dist(spike_train, fwin=fwin)
                mean_phase2 = nc_sig2.get_results()["LFP Spike Mean Phase"]
                phase_count2 = nc_sig2.get_results()["LFP Spike Mean Phase Count"]
                spike_phase_vect2 = nc_sig2.get_results()["LFP Spike Phase Res Vect"]
                name = os.path.join(
                    output_dir,
                    "spike_phase_plots",
                    f"{name_start}_{name_for_save}_RSC_Phase.{fmt}",
                )
                fig = plot_phase(g_data)
                fig.savefig(name, dpi=400)
                plt.close(fig)
            else:
                mean_phase2 = None
                phase_count2 = None
                spike_phase_vect2 = None

            # Spike shuffling
            number_of_shuffles = kwargs.get("number_of_shuffles_sta", 500)
            shuffled_times = unit.underlying.shift_spike_times(number_of_shuffles, None)
            shuffle_sfc_sub = np.zeros(shape=(number_of_shuffles, len(sfc)))
            spike_phase_vects = np.zeros(number_of_shuffles)

            if nc_sig2 is not None:
                shuffle_sfc_rsc = np.zeros(shape=(number_of_shuffles, len(sfc)))
                spike_phase_vects_rsc = np.zeros(number_of_shuffles)
            else:
                shuffle_sfc_rsc = None

            for i in range(number_of_shuffles):
                spike_times = shuffled_times[i]

                g_data = nc_sig.plv(spike_times, mode="bs", fwin=[0, 20], nrep=20)
                sfc_sub_rand = g_data["SFCm"]
                shuffle_sfc_sub[i] = sfc_sub_rand

                if nc_sig2 is not None:
                    g_data = nc_sig2.plv(spike_times, mode="bs", fwin=[0, 20], nrep=20)
                    sfc_rsc_rand = g_data["SFCm"]
                    shuffle_sfc_rsc[i] = sfc_rsc_rand

                nc_sig.phase_dist(spike_train, fwin=fwin)
                spike_phase_vects[i] = nc_sig.get_results()["LFP Spike Phase Res Vect"]

                if nc_sig2 is not None:
                    nc_sig2.phase_dist(spike_train, fwin=fwin)
                    spike_phase_vects_rsc[i] = nc_sig2.get_results()[
                        "LFP Spike Phase Res Vect"
                    ]

            spike_phase_vect_sub_ci = np.percentile(spike_phase_vects, 95)

            if nc_sig2 is not None:
                spike_phase_vect_rsc_ci = np.percentile(spike_phase_vects_rsc, 95)
            else:
                spike_phase_vect_rsc_ci = None

            theta_vals = []
            for f_val, sfc_val in zip(f, sfc):
                if fwin[0] <= f_val <= fwin[1]:
                    theta_vals.append(sfc_val)
            sfc_theta_sub = np.nanmean(theta_vals)

            sfc_theta_rsc = None
            if nc_sig2 is not None:
                theta_vals = []
                for f_val, sfc_val in zip(f, sfc_rsc):
                    if fwin[0] <= f_val <= fwin[1]:
                        theta_vals.append(sfc_val)
                sfc_theta_rsc = np.nanmean(theta_vals)

            output[name_for_save] = [
                mean_phase,
                phase_count,
                spike_phase_vect,
                spike_phase_vect_sub_ci,
                mean_phase2,
                phase_count2,
                spike_phase_vect2,
                spike_phase_vect_rsc_ci,
                sfc_theta_sub,
                sfc_theta_rsc,
                sta,
                sfc,
                sta_rsc,
                sfc_rsc,
                t,
                f,
                shuffle_sfc_sub,
                shuffle_sfc_rsc,
                spatial,
            ]
            unit.underlying.reset_results()

    return output


def combine_results(info, extra, **kwargs):
    import os
    import simuran
    import seaborn as sns

    base_dir = kwargs.get("cfg_base_dir")
    cell_list_name = extra[1]

    out_dir, filename = extra
    base, ext = os.path.splitext(os.path.basename(filename))

    here = os.path.dirname(os.path.abspath(__file__))
    cell_list_location = os.path.join(here, "..", "cell_lists", cell_list_name)
    df = df_from_file(cell_list_location)

    for out_region in ["sub", "rsc"]:
        new_list1 = []
        new_list2 = []
        for row in info.itertuples():
            dir_ = row.Directory[len(base_dir + os.sep) :]
            group = dir_[0]

            if group == "C":
                group = "Control"
            elif group == "L":
                group = "ATNx (Lesion)"
            elif group == "m":
                group = "Muscimol"
            else:
                raise ValueError("unsupported group {}".format(group))
            if out_region == "sub":
                sta = row.STA_SUB
                sfc = row.SFC_SUB
                random = np.mean(row.RandomSFC_SUB, axis=0)
            else:
                sta = row.STA_RSC
                if sta is None:
                    continue
                sfc = row.SFC_RSC
                random = np.mean(row.RandomSFC_RSC, axis=0)
            t = row.Time
            f = row.Frequency

            if group == "Muscimol":
                spatial = df[
                    (df["Filename"] == row.Filename)
                    & (df["Group"] == row.Group)
                    & (df["Unit"] == row.Unit)
                ]["Directory"].values.flatten()[0]
                spatial = spatial[len(base_dir + os.sep) :].split(os.sep)[-1]
                if "before" in spatial or "next" in spatial:
                    spatial = "Control"
                else:
                    spatial = "Muscimol"
            else:
                spatial = (
                    df[
                        (df["Filename"] == row.Filename)
                        & (df["Group"] == row.Group)
                        & (df["Unit"] == row.Unit)
                    ]["class"]
                    .values.flatten()[0]
                    .split("_")[0]
                )
            if group == "ATNx (Lesion)" and spatial == "S":
                raise RuntimeError("Incorrect parsing")

            for i in range(len(sta)):
                new_list1.append([group, float(sta[i]), float(t[i]), spatial])
            for i in range(len(sfc)):
                new_list2.append(
                    [
                        group,
                        float(sfc[i]) / 100,
                        float(random[i]) / 100,
                        float(f[i]),
                        spatial,
                    ]
                )

        headers1 = ["Group", "STA", "Time (s)", "Spatial"]
        headers2 = ["Group", "SFC", "Shuffled SFC", "Frequency (Hz)", "Spatial"]
        df1 = list_to_df(new_list1, headers=headers1)
        df2 = list_to_df(new_list2, headers=headers2)

        simuran.set_plot_style()

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df1,
            x="Time (s)",
            y="STA",
            ax=ax,
            style="Group",
            hue="Spatial",
            ci=None,
        )
        simuran.despine()
        mc = UnicodeGrabber.get("micro")
        ax.set_ylabel(f"Spike triggered average ({mc}V)")
        name = f"average_sta_{out_region}"
        fig.savefig(os.path.join(out_dir, name + "." + "pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        if "Muscimol" in df2["Group"]:
            sns.lineplot(
                data=df2,
                x="Frequency (Hz)",
                y="SFC",
                ax=ax,
                hue="Spatial",
            )
        else:
            sns.lineplot(
                data=df2,
                x="Frequency (Hz)",
                y="SFC",
                ax=ax,
                style="Group",
                hue="Spatial",
            )
        simuran.despine()
        ax.set_ylabel("Spike field coherence")
        name = f"average_sfc_{out_region}"
        fig.savefig(os.path.join(out_dir, name + "." + "pdf"))
        plt.close(fig)

        fig, ax = plt.subplots()
        sns.lineplot(
            data=df2,
            x="Frequency (Hz)",
            y="Shuffled SFC",
            ax=ax,
            style="Group",
            hue="Spatial",
        )
        simuran.despine()
        ax.set_ylabel("Spike field coherence")
        name = f"average_sfc_shuffled_{out_region}"
        fig.savefig(os.path.join(out_dir, name + "." + "pdf"))
        plt.close(fig)

        out_fname = os.path.join(out_dir, base + f"__sta_{out_region}" + ext)
        df_to_file(df1, out_fname, index=False)
        out_fname = os.path.join(out_dir, base + f"__sfc_{out_region}" + ext)
        df_to_file(df2, out_fname, index=False)
