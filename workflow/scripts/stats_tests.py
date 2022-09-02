"""
Grab the output CSV files and run stats on them.

See jasp folder for JASP tests (though the data in these tests may be old).
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import simuran as smr
from skm_pyutils.stats import corr, mwu, wilcoxon
from skm_pyutils.table import df_from_file

here = os.path.dirname(os.path.abspath(__file__))


def pt(title, n_dashes=20, start="\n"):
    str_ = "-" * n_dashes + title + "-" * n_dashes
    print(start + str_)


@dataclass
class PathAndDataGetter(object):
    plot_location: Union[str, Path]

    def get_df(self, filename, describe=False):
        df = df_from_file(filename)
        if describe:
            print(f"Processing {filename}")
            print("Overall")
            print(df.describe())
        try:
            control_df = df[df["Condition"] == "Control"]
            lesion_df = df[df["Condition"] == "Lesion"]
        except KeyError:
            return df
        if describe:
            print(f"Processing {filename}")
            print("Overall")
            print(df.describe())
            print("Control")
            print(control_df.describe())
            print("Lesion")
            print(lesion_df.describe())
        return df, control_df, lesion_df

    def get_musc_df(self, filename, describe=False):
        def process_condition(row):
            to_check = row["Spatial"]
            return (
                "Control"
                if ("before" in to_check) or ("next" in to_check)
                else "Muscimol"
            )

        df = df_from_file(filename)
        if describe:
            print("Processing {}".format(filename))
            print("Overall")
            print(df.describe())
        df["group"] = df.apply(lambda row: process_condition(row), axis=1)
        try:
            control_df = df[df["group"] == "Control"]
            lesion_df = df[df["group"] == "Muscimol"]
        except KeyError:
            return df
        if describe:
            print("Processing {}".format(filename))
            print("Overall")
            print(df.describe())
            print("Control")
            print(control_df.describe())
            print("Lesion")
            print(lesion_df.describe())
        return df, control_df, lesion_df

    def process_fig(self, res, name):
        fig = res["figure"]
        fig.savefig(os.path.join(self.plot_dir, name), dpi=400)
        plt.close(fig)


def power_stats(input_path, overall_kwargs, get_obj):
    pt("Open field power", start="")
    df, control_df, lesion_df = get_obj.get_df(input_path)

    t1_kwargs = {
        **overall_kwargs,
        **{"value": "subicular relative theta powers (unitless)"},
    }
    res = mwu(
        control_df["SUB_theta"], lesion_df["SUB_theta_rel"], t1_kwargs, do_plot=True
    )
    get_obj.process_fig(res, "sub_theta_openfield.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{"value": "retrospenial relative theta powers (unitless)"},
    }
    res = mwu(
        control_df["RSC_theta_rel"], lesion_df["RSC_theta_rel"], t2_kwargs, do_plot=True
    )
    get_obj.process_fig(res, "rsc_theta_openfield.pdf")


def coherence_stats(input_path, overall_kwargs, get_obj):
    pt("Open field coherence")
    df, control_df, lesion_df = get_obj.get_df(input_path)

    t1_kwargs = {
        **overall_kwargs,
        **{"value": "theta coherence (unitless)"},
    }
    res = mwu(
        control_df["Peak_Theta_coherence"],
        lesion_df["Peak_Theta_coherence"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "theta_coherence_openfield.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{"value": "delta coherence (unitless)"},
    }
    res = mwu(
        control_df["Delta_Coherence"],
        lesion_df["Delta_Coherence"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "delta_coherence_openfield.pdf")


def speed_stats(input_path, overall_kwargs, get_obj):
    pt("Open field Speed LFP power relationship")
    speed_df, speed_ctrl, speed_lesion = get_obj.get_df(input_path)
    test_kwargs = {
        **overall_kwargs,
        **{
            "group": "in control",
            "value1": "mean speed (cm/s)",
            "value2": "relative subicular theta power (unitless)",
            "trim": True,
            "offset": 0,
        },
    }
    speed_ctrl = speed_ctrl[speed_df["region"] == "SUB"]
    res = corr(
        speed_ctrl["Speed"],
        speed_ctrl["LFP amplitude"],
        test_kwargs,
        do_plot=False,
    )

    test_kwargs = {
        **overall_kwargs,
        **{
            "group": "in ATNx",
            "value1": "mean speed (cm/s)",
            "value2": "relative subicular theta power (unitless)",
            "trim": True,
            "offset": 0,
        },
    }
    speed_lesion = speed_lesion[speed_df["region"] == "SUB"]
    res = corr(
        speed_lesion["Speed"],
        speed_lesion["LFP amplitude"],
        test_kwargs,
        do_plot=False,
    )


def spike_lfp_stats(input_paths, overall_kwargs, get_obj):
    pt("Spike LFP openfield")
    df, control_df, lesion_df = get_obj.get_df(input_path)
    control_nspatial = control_df[control_df["Spatial"] == "Non-Spatial"]
    sub_control = control_nspatial[control_nspatial["Region"] == "SUB"]
    sub_lesion = lesion_df[lesion_df["Region"] == "SUB"]

    in_theta_range

    t1_kwargs = {
        **overall_kwargs,
        **{
            "value": "subicular theta spike field coherence for non-spatially tuned cells (percent)"
        },
    }

    res = mwu(
        control_nspatial_spikes["SFC"],
        lesion_df["SFC"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_sfc.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{
            "value": "retrospenial theta spike field coherence for non-spatially tuned cells (percent)"
        },
    }

    res = mwu(
        control_nspatial_spikes["Theta_SFC_RSC"],
        lesion_df["Theta_SFC_RSC"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "rsc_sfc.pdf")

    t3_kwargs = {
        **overall_kwargs,
        **{"value": "subicular theta phase (degrees)"},
    }

    res = mwu(
        control_nspatial_spikes["Mean_Phase_SUB"],
        lesion_df["Mean_Phase_SUB"],
        t3_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_phase.pdf")

    t4_kwargs = {
        **overall_kwargs,
        **{"value": "retrospenial theta phase (degrees)"},
    }

    res = mwu(
        control_nspatial_spikes["Mean_Phase_RSC"],
        lesion_df["Mean_Phase_RSC"],
        t4_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "rsc_phase.pdf")


def tmaze_stats(overall_kwargs, get_obj):
    pt("Tmaze stats")
    df, control_df, lesion_df = get_obj.get_df("tmaze--tmaze-times_results.csv")
    bit_to_get = (control_df["part"] == "choice") & (control_df["trial"] == "Correct")
    control_choice = control_df[bit_to_get]

    bit_to_get = (lesion_df["part"] == "choice") & (lesion_df["trial"] == "Correct")
    lesion_choice = lesion_df[bit_to_get]

    t1_kwargs = {
        **overall_kwargs,
        **{
            "value": "subicular to retronspenial LFP theta coherence during correct trials"
        },
    }

    res = mwu(
        control_choice["Peak 12Hz Theta coherence"],
        lesion_choice["Peak 12Hz Theta coherence"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "t-maze_coherence_correct.pdf")

    t1a_kwargs = {
        **overall_kwargs,
        **{"value": "subicular LFP theta power during correct trials"},
    }

    res = mwu(
        control_choice["SUB_theta"],
        lesion_choice["SUB_theta"],
        t1a_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "t-maze_subpower_correct.pdf")

    bit_to_get = (control_df["part"] == "choice") & (control_df["trial"] == "Incorrect")
    control_choice = control_df[bit_to_get]

    bit_to_get = (lesion_df["part"] == "choice") & (lesion_df["trial"] == "Incorrect")
    lesion_choice = lesion_df[bit_to_get]

    t2_kwargs = {
        **overall_kwargs,
        **{
            "value": "subicular to retronspenial LFP theta coherence during incorrect trials"
        },
    }

    res = mwu(
        control_choice["Peak 12Hz Theta coherence"],
        lesion_choice["Peak 12Hz Theta coherence"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "t-maze_coherence_incorrect.pdf")

    t2a_kwargs = {
        **overall_kwargs,
        **{"value": "subicular LFP theta power during incorrect trials"},
    }

    res = mwu(
        control_choice["SUB_theta"],
        lesion_choice["SUB_theta"],
        t2a_kwargs,
        do_plot=True,
    )
    process_fig(res, "t-maze_subpower_incorrect.pdf")

    bit_to_get = (control_df["part"] == "choice") & (control_df["trial"] == "Correct")
    control_choice1 = control_df[bit_to_get]

    bit_to_get = (control_df["part"] == "choice") & (control_df["trial"] == "Incorrect")
    control_choice2 = control_df[bit_to_get]

    t3_kwargs = {
        "value": "subicular to retronspenial LFP theta coherence during choice trials in control",
        "group1": "correct",
        "group2": "incorrect",
        "show_quartiles": overall_kwargs["show_quartiles"],
    }

    res = mwu(
        control_choice1["Peak 12Hz Theta coherence"],
        control_choice2["Peak 12Hz Theta coherence"],
        t3_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "t-maze_coherence_ctrl.pdf")

    bit_to_get = (lesion_df["part"] == "choice") & (lesion_df["trial"] == "Correct")
    lesion_choice1 = lesion_df[bit_to_get]

    bit_to_get = (lesion_df["part"] == "choice") & (lesion_df["trial"] == "Incorrect")
    lesion_choice2 = lesion_df[bit_to_get]

    t4_kwargs = {
        "value": "subicular to retronspenial LFP theta coherence during choice trials in ATNx",
        "group1": "correct",
        "group2": "incorrect",
        "show_quartiles": overall_kwargs["show_quartiles"],
    }

    res = mwu(
        lesion_choice1["Peak 12Hz Theta coherence"],
        lesion_choice2["Peak 12Hz Theta coherence"],
        t4_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "t-maze_coherence_lesion.pdf")


def muscimol_stats(overall_kwargs, get_obj):
    pt("Spike LFP muscimol")
    df, control_df, lesion_df = get_obj.get_df(
        "spike_lfp--CTRL_Lesion_cells_filled_eeg_recording_spike_lfp_results.csv"
    )

    t1_kwargs = {
        **overall_kwargs,
        **{"value": "subicular theta spike field coherence (percent)"},
    }

    res = mwu(
        control_df["Theta_SFC_SUB"],
        lesion_df["Theta_SFC_SUB"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_sfc_musc.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{"value": "retrospenial theta spike field coherence (percent)"},
    }

    res = mwu(
        control_df["Theta_SFC_RSC"],
        lesion_df["Theta_SFC_RSC"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "rsc_sfc_musc.pdf")

    t3_kwargs = {
        **overall_kwargs,
        **{"value": "subicular theta phase (degrees)"},
    }

    res = mwu(
        control_df["Mean_Phase_SUB"],
        lesion_df["Mean_Phase_SUB"],
        t3_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_phase_musc.pdf")

    t4_kwargs = {
        **overall_kwargs,
        **{"value": "retrospenial theta phase (degrees)"},
    }

    res = mwu(
        control_df["Mean_Phase_RSC"],
        lesion_df["Mean_Phase_RSC"],
        t4_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "rsc_phase_musc.pdf")


def main(input_paths, plot_dir, config_path, show_quartiles=False):
    cfg = smr.config_from_file(config_path)
    overall_kwargs_ttest = {
        "show_quartiles": show_quartiles,
        "group1": "control",
        "group2": "ATNx",
    }

    overall_kwargs_corr = {
        "show_quartiles": show_quartiles,
    }

    overall_kwargs_musc = {
        "show_quartiles": show_quartiles,
        "group1": "before muscimol",
        "group2": "after muscimol",
    }

    get_obj = PathAndDataGetter(plot_dir)
    # 1. Power overall
    theta_min, theta_max = cfg["theta_min"], cfg["theta_max"]
    power_stats(input_paths[0], overall_kwargs_ttest, get_obj)

    # 2. Coherence in the open-field
    coherence_stats(input_paths[1], overall_kwargs_ttest, get_obj)

    # 3. Speed to LFP power relationship
    speed_stats(input_paths[2], overall_kwargs_corr, get_obj)

    # 5. STA in openfield
    spike_lfp_stats(input_paths[3], overall_kwargs_ttest, get_obj)

    # 6. T-maze
    tmaze_stats(overall_kwargs_ttest)

    # 7. Muscimol stats
    muscimol_stats(overall_kwargs_musc)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        Path(snakemake.input[0]).parent,
        Path(snakemake.output[0]),
        snakemake.config["simuran_config"],
        snakemake.params["show_quartiles"],
    )
