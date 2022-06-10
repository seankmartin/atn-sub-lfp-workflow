import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file

module_logger = logging.getLogger("simuran.custom.speed_vs_lfp")


def df_from_rc(recording_container):
    dfs = []
    for recording in recording_container.load_iter():
        nwbfile = recording.data
        speed_df = nwbfile.processing["speed_theta"]["speed_lfp_table"].to_dataframe()
        speed_df.loc[:, "Group"] = recording.attrs["treatment"].capitalize()
        dfs.append(speed_df)
    return pd.concat(dfs, ignore_index=True)


def plot_speed_vs_lfp(df, out_dir, max_speed):
    smr.set_plot_style()
    df.replace("Control", "Control (ATN,   N = 6)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx, N = 5)", inplace=True)

    brain_regions = sorted(list(set(df["region"])))
    for region in brain_regions:
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df[(df["region"] == region) & (df["speed"] <= max_speed)],
            x="speed",
            y="power",
            style="Group",
            hue="Group",
            ci=95,
            estimator=np.median,
            ax=ax,
        )
        smr.despine()
        ax.set_xlabel("Speed (cm / s)")
        ax.set_ylabel("Theta Power (relative)")
        ax.set_title(f"{region} LFP power vs speed (median)")

        fname = out_dir / f"{region}--speed_theta"
        fig = smr.SimuranFigure(fig, fname)
        fig.save()


def main(input_df_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)

    speed_df = df_from_rc(rc)
    plot_speed_vs_lfp(speed_df, out_dir, config["max_speed"])


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent.parent,
        snakemake.config["simuran_config"],
    )
