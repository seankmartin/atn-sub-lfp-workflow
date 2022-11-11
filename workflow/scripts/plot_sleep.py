import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, list_to_df


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
        for brain_region, data in all_data.items():
            times, times_nrest = data
            metadata = df[df["nwb_file"] == filename]
            treatment = metadata["treatment"].values[0]
            duration = metadata["duration"].values[0]
            l.append(
                [
                    filename,
                    treatment,
                    brain_region,
                    60 * len(times) / (ratio_rest * duration),
                ]
            )
    df = list_to_df(l, headers=["Filename", "Condition", "Brain Region", "Ripples/min"])
    df_to_file(df, output_dir.parent.parent / "sleep" / "ripples2.csv")
    fig, ax = plt.subplots()
    smr.set_plot_style()
    sns.barplot(data=df, x="Condition", y="Ripples/min", hue="Brain Region", ax=ax)
    ax.set_title("Sharp wave ripples in sleep")
    smr_fig = smr.SimuranFigure(fig, output_dir / "ripples", done=True)
    smr_fig.save()


def plot_spindles(spindles_data, ripples_data, output_dir, config, df):
    l = []
    for spindles in spindles_data:
        filename, sp_dict, ratio_rest, resting_group, duration = spindles
        metadata = df[df["nwb_file"] == filename]
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
    sns.barplot(data=df, x="Condition", y="Spindles/min", hue="Brain Region", ax=ax)
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
