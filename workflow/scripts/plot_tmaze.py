from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file


def main(input_dir, config, out_dir):
    config = smr.config_from_file(config)
    coh_df, power_df, res_df = load_saved_results(input_dir)
    plot_coherence_results(res_df, coh_df, power_df, out_dir)


def plot_coherence_choice(coherence_df, out_dir):
    coherence_df["Trial result"] = coherence_df["Trial"]
    coherence_df = coherence_df[coherence_df["RSC on target"]]
    coherence_df_sub_bit = coherence_df[
        (coherence_df["Part"] == "choice") & (coherence_df["Trial"] != "Forced")
    ]
    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=coherence_df_sub_bit,
        x="Frequency (Hz)",
        y="Coherence",
        hue="Group",
        style="Trial result",
        errorbar=("ci", 95),
        estimator=np.median,
    )
    ax.set_ylim(0, 1)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / "choice_coherence_ci")
    fig.save()


def plot_choice_power(power_df, out_dir):
    power_df["Trial result"] = power_df["Trial"]
    power_df_sub_bit = power_df[
        (power_df["Part"] == "choice") & (power_df["Trial"] != "Forced")
    ]
    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        data=power_df_sub_bit,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Group",
        style="Trial result",
        estimator=np.median,
        errorbar=("ci", 95),
    )
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / "choice_power_ci")
    fig.save()


def plot_banded_coherence(out_dir, res_df):
    plot_bar_coherence(res_df, "Theta", out_dir)
    plot_bar_coherence(res_df, "Delta", out_dir)


def plot_total_coherence(out_dir, res_df):
    res_df = res_df[res_df["RSC on target"]]
    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.boxplot(data=res_df, y="Full Theta Coherence", x="Group", ax=ax)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / "total_coherence")
    fig.save()


def plot_grouped_power_coherence(out_dir, coherence_df, power_df):
    for group in ("Control", "Lesion (ATNx)"):
        coherence_df_sub = coherence_df[coherence_df["Group"] == group]
        power_df_sub = power_df[power_df["Group"] == group]
        plot_group_coherence(group, coherence_df_sub, out_dir)
        plot_group_power(group, power_df_sub, out_dir)


def plot_group_power(group, power_df_sub, out_dir):
    fig, ax = plt.subplots()
    smr.set_plot_style()
    sns.lineplot(
        data=power_df_sub,
        x="Frequency (Hz)",
        y="Power (dB)",
        hue="Part",
        style="Trial",
        errorbar=("ci", 95),
        estimator=np.median,
        ax=ax,
    )
    ax.set_xlim(0, 40)
    smr.despine()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_power_ci")
    fig.save()


def plot_group_coherence(group, coherence_df_sub, out_dir):
    smr.set_plot_style()
    fig, ax = plt.subplots()
    coherence_df_sub = coherence_df_sub[coherence_df_sub["RSC on target"]]
    for ci, ci_name in zip((None, ("ci", 95)), ("", "_ci")):
        sns.lineplot(
            data=coherence_df_sub,
            x="Frequency (Hz)",
            y="Coherence",
            hue="Part",
            style="Trial",
            errorbar=ci,
            estimator=np.median,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        smr.despine()
        fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{group}_coherence{ci_name}")
        fig.save()


def plot_bar_coherence(res_df, band: str, out_dir):
    res_df = res_df[res_df["RSC on target"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    for i, trial_part in enumerate(["Forced", "Correct", "Incorrect"]):
        res_df_part = res_df[res_df["trial"] == trial_part]
        smr.set_plot_style()
        sns.boxplot(
            data=res_df_part,
            x="part",
            y=f"{band} Coherence",
            hue="Group",
            ax=axes[i],
        )
        axes[i].set_ylim(0, 1.0)
        axes[i].set_title(f"{trial_part}")
        smr.despine()
    plt.tight_layout()
    fig = smr.SimuranFigure(fig=fig, name=out_dir / f"{band} coherence")
    fig.save()


def plot_coherence_results(res_df, coherence_df, power_df, out_dir):
    plot_banded_coherence(out_dir, res_df)
    plot_total_coherence(out_dir, res_df)
    plot_grouped_power_coherence(out_dir, coherence_df, power_df)
    plot_choice_power(power_df, out_dir)
    plot_coherence_choice(coherence_df, out_dir)


def load_saved_results(input_dir):
    coh_loc = input_dir / "coherence.csv"
    power_loc = input_dir / "power.csv"
    results_loc = input_dir / "results.csv"
    coherence_df = df_from_file(coh_loc)
    power_df = df_from_file(power_loc)
    res_df = df_from_file(results_loc)

    return coherence_df, power_df, res_df


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
            Path(snakemake.input[0]).parent,
            snakemake.config["simuran_config"],
            Path(snakemake.output[0]),
        )
    else:
        here = Path(__file__).parent.parent.parent
        main(
            here / "results" / "tmaze",
            here / "config" / "simuran_params.yml",
            here / "results" / "plots" / "tmaze",
        )
