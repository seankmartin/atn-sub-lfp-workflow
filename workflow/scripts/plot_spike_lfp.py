import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file


def plot_sta(sta_df, out_dir):
    brain_regions = sorted(list(set(sta_df["Region"])))
    is_musc = any(sta_df["Group"].str.startswith("musc"))
    hue = "Treatment" if is_musc else "Spatial"
    style = "Treatment" if is_musc else "Group"
    name_iter = zip(["", "_shuffled"], ["STA", "Shuffled STA"])
    for region, (name, y) in itertools.product(brain_regions, name_iter):
        df_part = sta_df[sta_df["Region"] == region]
        smr.set_plot_style()
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_part,
            x="Time (s)",
            y=y,
            ax=ax,
            style=style,
            hue=hue,
            ci=None,
        )
        smr.despine()
        ax.set_ylabel("Spike triggered average")
        out_name = out_dir / f"{region}_average_sta{name}"
        smr_fig = smr.SimuranFigure(fig=fig, filename=out_name)
        smr_fig.save()


def plot_sfc(sfc_df, out_dir):
    brain_regions = sorted(list(set(sfc_df["Region"])))
    is_musc = any(sfc_df["Group"].str.startswith("musc"))
    hue = "Treatment" if is_musc else "Spatial"
    style = "Treatment" if is_musc else "Group"
    name_iter = zip(["", "_shuffled"], ["SFC", "Shuffled SFC"])
    for region, (name, y) in itertools.product(brain_regions, name_iter):
        df_part = sfc_df[sfc_df["Region"] == region]
        smr.set_plot_style()
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_part,
            x="Frequency (Hz)",
            y=y,
            ax=ax,
            style=style,
            hue=hue,
            ci=None,
        )
        smr.despine()
        ax.set_ylabel("Spike field coherence")
        out_name = out_dir / f"{region}_average_sfc{name}"
        smr_fig = smr.SimuranFigure(fig=fig, filename=out_name)
        smr_fig.save()


def main(input_df_paths, out_dir):
    sta_df, sfc_df = [df_from_file(fpath) for fpath in input_df_paths]
    plot_sta(sta_df, out_dir)
    plot_sfc(sfc_df, out_dir)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input,
        Path(snakemake.output[0]),
    )
