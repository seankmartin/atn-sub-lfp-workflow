import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from skm_pyutils.plot import GridFig
from skm_pyutils.table import df_from_file, list_to_df

module_logger = logging.getLogger("simuran.custom.plot_spectra")


def group_type_from_rat_name(name):
    ctrl = "Control (ATN,   N = 6)"
    lesion = "Lesion  (ATNx, N = 5)"
    return lesion if name.lower().startswith("l") else ctrl


def grab_psds(nwbfile):
    psd_table = nwbfile.processing["lfp_power"]["power_spectra"].to_dataframe()
    electrodes_table = nwbfile.electrodes.to_dataframe()

    return psd_table, electrodes_table


def split_psds(psd_table, electrodes_table):
    normal_psds = psd_table[:-2][electrodes_table["clean"] == "Normal"]
    outlier_psds = psd_table[:-2][electrodes_table["clean"] == "Outlier"]

    return normal_psds, outlier_psds


def add_psds_for_region_to_list(l, normal_psds, outlier_psds, region):
    clean_psds_in_region = normal_psds[normal_psds["region"] == region]
    outlier_psds_in_region = outlier_psds[outlier_psds["region"] == region]
    average_psd_for_clean = np.mean(clean_psds_in_region["power"], axis=0)
    average_psd_for_outlier = np.mean(outlier_psds_in_region["power"], axis=0)
    l.extend(
        [x, y, "Clean", region]
        for (x, y) in zip(average_psd_for_clean, normal_psds.iloc[0]["frequency"])
    )
    if len(outlier_psds_in_region) != 0:
        l.extend(
            [x, y, "Outlier", region]
            for (x, y) in zip(average_psd_for_outlier, normal_psds.iloc[0]["frequency"])
        )


def create_psd_table(nwbfile):
    psd_table, electrodes_table = grab_psds(nwbfile)
    regions = sorted(list(set(electrodes_table["location"])))
    normal_psds, outlier_psds = split_psds(psd_table, electrodes_table)

    l = []
    for region in regions:
        add_psds_for_region_to_list(l, normal_psds, outlier_psds, region)
    headers = ["Power (Db)", "Frequency (Hz)", "Type", "Brain Region"]
    return list_to_df(l, headers=headers)


def convert_df_to_averages(psd_dataframe):
    l = []
    headers = ["Power (Db)", "Frequency (Hz)", "Brain Region"]
    regions = sorted(list(set(psd_dataframe["region"])))
    for r in regions:
        psd = psd_dataframe.loc[psd_dataframe["label"] == f"{r}_avg"]
        l.extend(
            [x, y, r] for x, y in zip(psd["power"].array[0], psd["frequency"].array[0])
        )
    return list_to_df(l, headers=headers)


def plot_split_psd(psd_dataframe, max_frequency=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    smr.set_plot_style()
    sns.lineplot(
        ax=ax,
        x="Frequency (Hz)",
        y="Power (Db)",
        style="Type",
        hue="Brain Region",
        data=psd_dataframe[psd_dataframe["Frequency (Hz)"] < max_frequency],
    )
    ax.set_title("Z-score outlier PSD")
    return ax


def plot_average_signal_psd(psd_dataframe, max_frequency=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    avg_df = convert_df_to_averages(psd_dataframe)
    sns.lineplot(
        ax=ax,
        x="Frequency (Hz)",
        y="Power (Db)",
        hue="Brain Region",
        data=avg_df[avg_df["Frequency (Hz)"] < max_frequency],
    )
    ax.set_title("Average signal PSD")
    return ax


def plot_psd_over_signals(normal_psds, max_frequency=30, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    normal_psds["Brain Region"] = normal_psds["region"]
    regions = sorted(list(set(normal_psds["Brain Region"])))
    data = []
    headers = ["Power (Db)", "Frequency (Hz)", "Brain Region"]
    for _, row in normal_psds.iterrows():
        region = row["region"]
        frequency = row["frequency"]
        power = row["power"]
        data.extend([p, f, region] for (f, p) in zip(frequency, power))
    df = list_to_df(data, headers)
    sns.lineplot(
        ax=ax,
        x="Frequency (Hz)",
        y="Power (Db)",
        hue="Brain Region",
        hue_order=regions,
        data=df[df["Frequency (Hz)"] < max_frequency],
    )
    ax.set_title("Across signals PSD CI")
    return ax


def plot_psds(recording, out_dir, max_frequency):
    psd_dataframe = create_psd_table(recording.data)
    normal_psds, _ = split_psds(*grab_psds(recording.data))

    gf = GridFig(rows=1, cols=3, size_multiplier_x=15, size_multiplier_y=10)
    plot_split_psd(psd_dataframe, max_frequency, ax=gf.get_next())
    plot_average_signal_psd(
        grab_psds(recording.data)[0], max_frequency, ax=gf.get_next()
    )
    plot_psd_over_signals(normal_psds, max_frequency, ax=gf.get_next())
    path = out_dir / f"{recording.get_name_for_save()}--spectra"
    fig = smr.SimuranFigure(gf.fig, path)
    fig.save()
    return psd_dataframe, path


def plot_per_animal_psd(per_animal_df, output_path, max_frequency):
    regions = sorted(list(set(per_animal_df["Brain Region"])))
    paths = []
    for region in regions:
        df = per_animal_df[per_animal_df["Brain Region"] == region]
        fig, ax = plt.subplots()
        sns.lineplot(
            ax=ax,
            x="Frequency (Hz)",
            y="Power (Db)",
            hue="Rat",
            data=df[df["Frequency (Hz)"] < max_frequency],
        )
        smr.despine()
        paths.append(f"{output_path}--{region}")
        fig = smr.SimuranFigure(fig, filename=f"{output_path}--{region}")
        fig.save()

    return paths


def plot_control_vs_lesion_psd(per_animal_df, output_path, max_frequency):
    regions = sorted(list(set(per_animal_df["Brain Region"])))
    paths = []
    for region in regions:
        df = per_animal_df[per_animal_df["Brain Region"] == region]
        fig, ax = plt.subplots()
        sns.lineplot(
            ax=ax,
            x="Frequency (Hz)",
            y="Power (Db)",
            hue="Group",
            style="Group",
            data=df[df["Frequency (Hz)"] < max_frequency],
            estimator=np.median,
        )
        ax.set_title(f"{region} LFP power (median)")
        smr.despine()
        paths.append(f"{output_path}--{region}")
        fig = smr.SimuranFigure(fig, filename=f"{output_path}--{region}")
        fig.save()


def main(df_path, config_path, out_dir):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    max_frequency = config["max_psd_freq"]

    for r in rc.load_iter():
        plot_psds(r, out_dir, max_frequency)


def summary(df_path, config_path, out_dir, order=0):
    """Order 0, average PSDs, order 1, average signals"""
    config = smr.ParamHandler(source_file=config_path)
    max_frequency = config["max_psd_freq"]
    full_df = df_from_file(df_path)

    end_bit = "averaged_psds" if order == 0 else "averaged_signals"
    path = out_dir / f"per_animal_psds--{end_bit}"
    smr.set_plot_style()
    plot_per_animal_psd(full_df, path, max_frequency)
    path = out_dir / f"per_group_psds--{end_bit}"
    plot_control_vs_lesion_psd(full_df, path, max_frequency)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])

    if snakemake.params.get("mode") == "summary":
        for order in (0, 1):
            summary(
                snakemake.input[order],
                snakemake.config["simuran_config"],
                Path(snakemake.output[0]).parent.parent,
                order,
            )
    else:
        main(
            snakemake.input[0],
            snakemake.config["simuran_config"],
            Path(snakemake.output[0]),
        )
