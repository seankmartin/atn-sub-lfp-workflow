import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, list_to_df, filter_table


def main(ripples_pkl, spindles_pkl, metadata_file, output_dir, config_path):
    config = smr.config_from_file(config_path)
    df = df_from_file(metadata_file)
    with open(ripples_pkl, "rb") as f:
        ripples_data = pickle.load(f)
        plot_ripples(ripples_data, output_dir, config, df)
    with open(spindles_pkl, "rb") as f:
        spindles_data = pickle.load(f)
        plot_spindles(spindles_data, ripples_data, output_dir, config, df)


def plot_ripples(ripples_data, output_dir, config, df):
    l = []
    for full_data in ripples_data:
        filename, all_data, ratio_rest, resting_groups, duration = full_data
        if ("CanCSRCa2" in filename) or ("CanCSRetCa2" in filename):
            continue
        for brain_region, data in all_data.items():
            times, times_nrest = data
            metadata = df[df["nwb_file"] == filename]
            try:
                treatment = metadata["treatment"].values[0]
            except IndexError:
                continue
            duration = metadata["duration"].values[0]
            l.append(
                [
                    filename,
                    treatment,
                    brain_region,
                    ratio_rest * duration,
                    60 * len(times) / (ratio_rest * duration),
                ]
            )
    df = list_to_df(
        l,
        headers=[
            "Filename",
            "Condition",
            "Brain Region",
            "Resting time",
            "Ripples/min",
        ],
    )
    df2 = create_long_style_df(df)
    df_to_file(df, output_dir.parent.parent / "sleep" / "ripples2.csv")
    df_to_file(df2, output_dir.parent.parent / "sleep" / "ripples_jasp.csv")
    fig, ax = plt.subplots()
    smr.set_plot_style()

    def map_to_br(value):
        if value == "Kay_CA1":
            return "CA1"
        if value == "Kay_SUB":
            return "SUB"

    df["Brain Region"] = df["Brain Region"].apply(map_to_br)
    df = filter_table(
        df,
        {
            "Brain Region": ["SUB", "CA1"],
            # "Condition": ["CanControl", "Muscimol"],
        },
    )
    sns.boxplot(
        data=df,
        hue="Condition",
        hue_order=["Control", "Lesion", "CanControl", "Muscimol"],
        # hue_order=["CanControl", "Muscimol"],
        order=["SUB", "CA1"],
        y="Ripples/min",
        x="Brain Region",
        ax=ax,
        palette="pastel",
        showfliers=False,
        width=0.9,
    )
    sns.stripplot(
        hue="Condition",
        y="Ripples/min",
        x="Brain Region",
        hue_order=["Control", "Lesion", "CanControl", "Muscimol"],
        # hue_order=["CanControl", "Muscimol"],
        order=["SUB", "CA1"],
        ax=ax,
        data=df,
        palette=["0.4", "0.75"],
        alpha=0.95,
        dodge=True,
        edgecolor="k",
        linewidth=1,
        size=4.5,
        legend=False,
    )
    ax.set_title("Sharp wave ripples in sleep")
    smr_fig = smr.SimuranFigure(fig, output_dir / "ripples", done=True)
    smr_fig.save()


def create_long_style_df(df):
    br_headers = sorted(list(set(df["Brain Region"])))

    data = {
        "Filename": [],
        "Condition": [],
        "Resting time": [],
    }

    for br in br_headers:
        data[br] = []

    for i, row in df.iterrows():
        if row["Filename"] not in data["Filename"]:
            data["Filename"].append(row["Filename"])
            data["Condition"].append(row["Condition"])
            data["Resting time"].append(row["Resting time"])
            for br in br_headers:
                if len(data[br]) < len(data["Filename"]) - 1:
                    data[br].append(np.nan)
        data[row["Brain Region"]].append(row["Ripples/min"])

    for br in br_headers:
        if len(data[br]) < len(data["Filename"]):
            data[br].append(np.nan)

    return pd.DataFrame(data)


def plot_spindles(spindles_data, ripples_data, output_dir, config, df):
    l = []
    for spindles in spindles_data:
        filename, sp_dict, ratio_rest, resting_group, duration = spindles
        if ("CanCSRCa2" in filename) or ("CanCSRetCa2" in filename):
            continue

        metadata = df[df["nwb_file"] == filename]
        try:
            treatment = metadata["treatment"].values[0]
        except IndexError:
            continue
        treatment = metadata["treatment"].values[0]
        duration = metadata["duration"].values[0]
        for br, sp in sp_dict.items():
            num_spindles = 0 if sp is None else len(sp)
            l.append(
                [filename, treatment, br, 60 * num_spindles / (ratio_rest * duration)]
            )
    df = list_to_df(
        l, headers=["Filename", "Condition", "Brain Region", "Spindles/min"]
    )
    df_to_file(df, output_dir.parent.parent / "sleep" / "spindles2.csv")
    fig, ax = plt.subplots()
    smr.set_plot_style()
    sns.boxplot(data=df, x="Condition", y="Spindles/min", hue="Brain Region", ax=ax)
    ax.set_title("Spindles in sleep")
    smr_fig = smr.SimuranFigure(fig, output_dir / "spindles", done=True)
    smr_fig.save()


if __name__ == "__main__":
    try:
        snakemake
    except Exception:
        use_snakemake = False
    else:
        use_snakemake = True
    if use_snakemake:
        smr.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.input[0],
            snakemake.input[1],
            snakemake.input[2],
            Path(snakemake.output[0]),
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "sleep" / "ripples.pkl",
            here / "results" / "sleep" / "spindles.pkl",
            here / "results" / "every_processed_nwb.csv",
            here / "results" / "plots" / "sleep",
            here / "config" / "simuran_params.yml",
        )
