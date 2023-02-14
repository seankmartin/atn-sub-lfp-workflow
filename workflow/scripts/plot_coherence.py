from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file


def fix_notch_freqs(df, freqs_to_fix):
    fnames = sorted(list(set(df["Fname"])))
    for fname in fnames:
        fname_bit = df["Fname"] == fname
        df_bit = df[fname_bit]
        if len(df_bit) == 0:
            continue
        for f in freqs_to_fix:
            start_val = (
                df_bit[df_bit["Frequency (Hz)"].between(f - 5, f - 4)]["Coherence"]
                .iloc[0]
                .data
            )
            end_val = (
                df_bit[df_bit["Frequency (Hz)"].between(f + 4, f + 5)]["Coherence"]
                .iloc[-1]
                .data
            )
            freqs = df_bit["Frequency (Hz)"].between(f - 5, f + 5)
            interp = np.linspace(
                start_val, end_val, np.count_nonzero(freqs), endpoint=True
            )
            df.loc[freqs & fname_bit, "Coherence"] = interp


def plot_coherence(df, out_dir, max_frequency=40):
    smr.set_plot_style()

    df.replace("Control", "Control (ATN)", inplace=True)
    df.replace("Lesion", "Lesion  (ATNx)", inplace=True)
    df_sub = df[df["RSC on target"]]

    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_sub[df_sub["Frequency (Hz)"] <= 120],
        x="Frequency (Hz)",
        y="Coherence",
        style="Group",
        hue="Group",
        # estimator="median",
        estimator="mean",
        errorbar=("ci", 95),
        n_boot=10000,
        ax=ax,
    )

    plt.ylim(0, 1)
    smr.despine()
    filename = out_dir / "coherence_on_target"
    fig = smr.SimuranFigure(fig, filename)
    fig.save()

    df_sub = df[~df["RSC on target"]]
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_sub[df_sub["Frequency (Hz)"] <= 120],
        x="Frequency (Hz)",
        y="Coherence",
        style="Group",
        hue="Group",
        # estimator="median",
        estimator="mean",
        errorbar=("ci", 95),
        ax=ax,
    )

    plt.ylim(0, 1)
    smr.despine()
    filename = out_dir / "coherence_off_target"
    fig = smr.SimuranFigure(fig, filename)
    fig.save()


def main(input_df_path, out_dir, config_path):
    config = smr.config_from_file(config_path)
    coherence_df = df_from_file(input_df_path)
    fix_notch_freqs(coherence_df, config["notch_freqs"])
    plot_coherence(coherence_df, out_dir, config["max_psd_freq"])


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent.parent,
        snakemake.config["simuran_config"],
    )
