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
from skm_pyutils.table import df_from_file, df_to_file

here = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PathAndDataGetter(object):
    plot_dir: Union[str, Path]
    control_name: str = "Control"
    lesion_name: str = "Lesion"
    full_str: str = ""

    def update_group(self, df):
        def group_type(name):
            ctrl = self.control_name
            lesion = self.lesion_name
            return lesion if name.lower().startswith("l") else ctrl

        df["Condition"] = df["Condition"].apply(lambda x: group_type(x))
        return df

    def get_df(self, filename, describe=False):
        df = df_from_file(filename)
        if ("Group" in df.columns) and ("Condition" not in df.columns):
            df.loc[:, "Condition"] = df["Group"]
        try:
            df = self.update_group(df)
            control_df = df[df["Condition"] == self.control_name]
            lesion_df = df[df["Condition"] == self.lesion_name]
        except KeyError:
            if describe:
                self.describe_df(filename, df)
            return df
        if describe:
            self.describe_df_grouped(filename, df, control_df, lesion_df)
        return df, control_df, lesion_df

    def get_musc_df(self, filename, spatial=True, describe=False):
        df = df_from_file(filename)
        if describe:
            self.describe_df(filename, df)
        if ("Treatment" in df.columns) and ("Condition" not in df.columns):
            df.loc[:, "Condition"] = df["Treatment"]
        try:
            if spatial:
                control_df = df[df["Condition"] == "Control"]
                lesion_df = df[df["Condition"] == "Muscimol"]
        except KeyError:
            return df
        if describe:
            self.describe_df_grouped(filename, df, control_df, lesion_df)
        return df, control_df, lesion_df

    def describe_df_grouped(self, filename, df, control_df, lesion_df):
        self.describe_df(filename, df)
        print("Control")
        print(control_df.describe())
        print("Lesion")
        print(lesion_df.describe())

    def describe_df(self, filename, df):
        print(f"Processing {filename}")
        print("Overall")
        print(df.describe())

    def process_fig(self, res, name):
        fig = res["figure"]
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir, exist_ok=True)
        fig.savefig(os.path.join(self.plot_dir, name), dpi=400)
        plt.close(fig)

        self.process_str(res)

    def process_str(self, res):
        str_bit = res["output"]
        self.full_str = f"{self.full_str}\n{str_bit}"

    def to_file(self, filename):
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(self.full_str)

    def pt(self, title, n_dashes=20, start="\n"):
        str_ = "-" * n_dashes + title + "-" * n_dashes
        print(start + str_)
        self.full_str = f"{self.full_str}\n{str_}"

    def save_df(self, df, filename):
        out_dir = self.plot_dir.parent.parent / "sheets"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = out_dir / filename
        df_to_file(df, out_name)


def power_stats(input_path, overall_kwargs, get_obj):
    get_obj.pt("Open field power", start="")
    df, control_df, lesion_df = get_obj.get_df(input_path)

    names = ["Delta", "Theta", "Beta", "Low Gamma", "High Gamma"]
    for name in names:

        t1_kwargs = {
            **overall_kwargs,
            **{
                "value": f"subicular relative {name} powers (unitless)",
                "n_decimals": 4,
                "np_decimals": 4,
            },
        }
        res = mwu(
            control_df[f"SUB {name} Rel"],
            lesion_df[f"SUB {name} Rel"],
            t1_kwargs,
            do_plot=True,
        )
        get_obj.process_fig(res, f"sub_{name}_openfield.pdf")

        t2_kwargs = {
            **overall_kwargs,
            **{
                "value": f"retrospenial relative {name} powers (unitless)",
                "n_decimals": 4,
                "np_decimals": 4,
            },
        }
        res = mwu(
            control_df[control_df["RSC on target"]][f"RSC {name} Rel"],
            lesion_df[lesion_df["RSC on target"]][f"RSC {name} Rel"],
            t2_kwargs,
            do_plot=True,
        )
        get_obj.process_fig(res, f"rsc_{name}_openfield.pdf")


def coherence_stats(input_path, overall_kwargs, get_obj):
    get_obj.pt("Open field coherence")
    df, control_df, lesion_df = get_obj.get_df(input_path)
    df = df[df["RSC on target"]]
    control_df = control_df[control_df["RSC on target"]]
    lesion_df = lesion_df[lesion_df["RSC on target"]]

    for band in ["Delta", "Theta", "Beta", "Low Gamma", "High Gamma"]:
        t1_kwargs = {
            **overall_kwargs,
            **{"value": f"{band} coherence (unitless)"},
        }
        res = mwu(
            control_df[f"Mean {band} Coherence"],
            lesion_df[f"Mean {band} Coherence"],
            t1_kwargs,
            do_plot=True,
        )
        get_obj.process_fig(res, f"{band}_coherence_openfield.pdf")


def speed_stats(input_path, overall_kwargs, get_obj):
    get_obj.pt("Open field Speed LFP power relationship")
    speed_df, speed_ctrl, speed_lesion = get_obj.get_df(input_path)

    for region, name in zip(["SUB", "RSC"], ["subicular", "retrospenial"]):
        if region == "RSC":
            speed_ctrl_df = speed_ctrl[speed_ctrl["RSC on target"]]
            speed_lesion_df = speed_lesion[speed_lesion["RSC on target"]]
        else:
            speed_ctrl_df = speed_ctrl
            speed_lesion_df = speed_lesion
        test_kwargs = {
            **overall_kwargs,
            **{
                "group": "in control",
                "value1": "mean speed (cm/s)",
                "value2": f"relative {name} theta power (unitless)",
                "trim": True,
                "offset": 0,
            },
        }
        speed_ctrl_df = speed_ctrl_df[speed_ctrl_df["region"] == region]
        res = corr(
            speed_lesion_df.loc[~nan_values]["speed"],
            speed_lesion_df.loc[~nan_values]["power"],
            test_kwargs,
            do_plot=False,
        )
        get_obj.process_str(res)

        test_kwargs = {
            **overall_kwargs,
            **{
                "group": "in ATNx",
                "value1": "mean speed (cm/s)",
                "value2": f"relative {name} theta power (unitless)",
                "trim": True,
                "offset": 0,
            },
        }
        speed_lesion_df = speed_lesion_df[speed_lesion_df["region"] == region]
        nan_values = speed_lesion_df["power"].isna()
        res = corr(
            speed_lesion_df.loc[~nan_values]["speed"],
            speed_lesion_df.loc[~nan_values]["power"],
            test_kwargs,
            do_plot=False,
        )
        get_obj.process_str(res)


def spike_lfp_stats(input_paths, overall_kwargs, get_obj):
    get_obj.pt("Spike LFP openfield")
    df, control_df, lesion_df = get_obj.get_df(input_paths[0])
    control_df = control_df[control_df["Region"] == "SUB"]
    lesion_df = lesion_df[lesion_df["Region"] == "SUB"]

    t1_kwargs = {
        **overall_kwargs,
        **{"value": "subicular theta spike field coherence for spatial vs non"},
    }

    res = mwu(
        control_df[control_df["Spatial"] == "Spatial"]["AVG Theta SFC"],
        control_df[control_df["Spatial"] == "Non-Spatial"]["AVG Theta SFC"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_theta_sfc_spatial_non.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{
            "value": "subicular spike field coherence for non-spatial CTRL vs non-spatial lesion"
        },
    }

    res = mwu(
        control_df[control_df["Spatial"] == "Non-Spatial"]["AVG Theta SFC"],
        lesion_df[lesion_df["Spatial"] == "Non-Spatial"]["AVG Theta SFC"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_delta_sfc_non_spatial_only.pdf")

    for ip in input_paths[1:]:
        df = df_from_file(ip)
        t_kwargs = {**overall_kwargs, **{"value": "muscimol vs control sub theta sfc"}}
        res = wilcoxon(df["AVG Theta SFC"], df["AVG Theta SFC_musc"], t_kwargs)


def tmaze_stats(input_path, overall_kwargs, get_obj):
    get_obj.pt("Tmaze stats")
    df, control_df, lesion_df = get_obj.get_df(input_path)

    bit_to_get = (
        (control_df["part"] == "choice")
        & (control_df["trial"] == "Correct")
        & (control_df["RSC on target"])
    )
    control_choice = control_df[bit_to_get]

    bit_to_get = (
        (lesion_df["part"] == "choice")
        & (lesion_df["trial"] == "Correct")
        & (lesion_df["RSC on target"])
    )
    lesion_choice = lesion_df[bit_to_get]

    bit_to_get = (
        (df["part"] == "choice") & (df["trial"] == "Correct") & (df["RSC on target"])
    )
    get_obj.save_df(df[bit_to_get], "tmaze_coherence_correct.csv")

    for band in ("Delta", "Theta", "Beta", "Low Gamma", "High Gamma"):
        t1_kwargs = {
            **overall_kwargs,
            **{
                "value": f"subicular to retronspenial LFP {band} coherence in choice parts during correct trials"
            },
        }

        res = mwu(
            control_choice[f"Full {band} Coherence"],
            lesion_choice[f"Full {band} Coherence"],
            t1_kwargs,
            do_plot=True,
        )
        get_obj.process_fig(res, f"t-maze_coherence_correct_{band}.pdf")

    bit_to_get = (
        (control_df["part"] == "choice")
        & (control_df["trial"] == "Incorrect")
        & (control_df["RSC on target"])
    )
    control_choice = control_df[bit_to_get]

    bit_to_get = (
        (lesion_df["part"] == "choice")
        & (lesion_df["trial"] == "Incorrect")
        & (lesion_df["RSC on target"])
    )
    lesion_choice = lesion_df[bit_to_get]

    bit_to_get = (
        (df["part"] == "choice") & (df["trial"] == "Incorrect") & (df["RSC on target"])
    )

    get_obj.save_df(df[bit_to_get], "tmaze_coherence_incorrect.csv")

    for band in ("Delta", "Theta", "Beta", "Low Gamma", "High Gamma"):
        t2_kwargs = {
            **overall_kwargs,
            **{
                "value": f"subicular to retronspenial LFP {band} coherence during incorrect trials"
            },
        }

        res = mwu(
            control_choice[f"Full {band} Coherence"],
            lesion_choice[f"Full {band} Coherence"],
            t2_kwargs,
            do_plot=True,
        )
        get_obj.process_fig(res, f"t-maze_coherence_incorrect_{band}.pdf")


def muscimol_stats(input_path, overall_kwargs, get_obj):
    get_obj.pt("Spike LFP muscimol")
    df, control_df, lesion_df = get_obj.get_musc_df(input_path)
    sub_control = control_df[control_df["Region"] == "SUB"]
    rsc_control = control_df[control_df["Region"] == "RSC"]
    sub_lesion = lesion_df[lesion_df["Region"] == "SUB"]
    rsc_lesion = lesion_df[lesion_df["Region"] == "RSC"]
    rsc_control = rsc_control[rsc_control["RSC on target"]]
    rsc_lesion = rsc_lesion[rsc_lesion["RSC on target"]]

    t1_kwargs = {
        **overall_kwargs,
        **{"value": "subicular theta spike field coherence (percent)"},
    }

    res = mwu(
        sub_control["Peak Theta SFC"],
        rsc_control["Peak Theta SFC"],
        t1_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "sub_sfc_musc.pdf")

    t2_kwargs = {
        **overall_kwargs,
        **{"value": "retrospenial theta spike field coherence (percent)"},
    }

    res = mwu(
        sub_lesion["Peak Theta SFC"],
        rsc_lesion["Peak Theta SFC"],
        t2_kwargs,
        do_plot=True,
    )
    get_obj.process_fig(res, "rsc_sfc_musc.pdf")


def sleep_stats(spindles_path, ripples_path, overall_kwargs, get_obj):
    get_obj.pt("Sleep")
    df = df_from_file(ripples_path)

    t_kwargs = {**overall_kwargs, **{"value": "control/muscimol ripples in SUB"}}
    mwu(
        df[(df["Condition"] == "CanControl") & (df["Brain Region"] == "Kay_SUB")][
            "Ripples/min"
        ],
        df[(df["Condition"] == "Muscimol") & (df["Brain Region"] == "Kay_SUB")][
            "Ripples/min"
        ],
        t_kwargs,
    )

    t_kwargs = {**overall_kwargs, **{"value": "control/muscimol ripples in CA1"}}
    mwu(
        df[(df["Condition"] == "CanControl") & (df["Brain Region"] == "Kay_CA1")][
            "Ripples/min"
        ],
        df[(df["Condition"] == "Muscimol") & (df["Brain Region"] == "Kay_CA1")][
            "Ripples/min"
        ],
        t_kwargs,
    )

    t_kwargs = {**overall_kwargs, **{"value": "control/lesion ripples in SUB"}}
    mwu(
        df[(df["Condition"] == "Control") & (df["Brain Region"] == "Kay_SUB")][
            "Ripples/min"
        ],
        df[(df["Condition"] == "Lesion") & (df["Brain Region"] == "Kay_SUB")][
            "Ripples/min"
        ],
        t_kwargs,
    )


def main(input_paths, plot_dir, output_file, show_quartiles=False):
    overall_kwargs_ttest = {
        "show_quartiles": show_quartiles,
        "group1": "Control",
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
    power_stats(input_paths[0], overall_kwargs_ttest, get_obj)

    # 2. Coherence in the open-field
    coherence_stats(input_paths[1], overall_kwargs_ttest, get_obj)

    # 3. Speed to LFP power relationship
    speed_stats(input_paths[2], overall_kwargs_corr, get_obj)

    # 5. STA in openfield
    spike_lfp_stats(input_paths[3:6], overall_kwargs_ttest, get_obj)

    # 6. T-maze
    tmaze_stats(input_paths[6], overall_kwargs_ttest, get_obj)

    # 7. Muscimol stats
    muscimol_stats(input_paths[7], overall_kwargs_musc, get_obj)

    # 8. Sleep ripples and spindles
    sleep_stats(input_paths[8], input_paths[9], overall_kwargs_ttest, get_obj)

    get_obj.to_file(output_file)


if __name__ == "__main__":
    try:
        a = snakemake
    except NameError:
        use_snakemake = False
    else:
        use_snakemake = True

    if use_snakemake:
        smr.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.input,
            Path(snakemake.output[0]),
            Path(snakemake.output[1]),
            snakemake.params["show_quartiles"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            [
                here / "results/summary/signal_bandpowers.csv",
                here / "results/summary/coherence_stats.csv",
                here / "results/summary/speed_theta_avg.csv",
                here / "results/summary/openfield_peak_sfc.csv",
                here / "results/summary/musc_spike_lfp_sub_pairs.csv",
                here / "results/summary/musc_spike_lfp_sub_pairs_later.csv",
                here / "results/tmaze/results.csv",
                here / "results/summary/muscimol_peak_sfc.csv",
                here / "results/sleep/spindles2.csv",
                here / "results/sleep/ripples2.csv",
            ],
            here / "results" / "plots" / "stats",
            here / "results" / "stats_output.txt",
            True,
        )
