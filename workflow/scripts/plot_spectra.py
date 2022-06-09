import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import simuran as smr
from skm_pyutils.table import df_from_file, list_to_df

module_logger = logging.getLogger("simuran.custom.plot_spectra")


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


def plot_split_psd(psd_dataframe, output_path, max_frequency=30):
    smr.set_plot_style()
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        x="Frequency (Hz)",
        y="Power (Db)",
        style="Type",
        hue="Brain Region",
        data=psd_dataframe[psd_dataframe["Frequency (Hz)"] < max_frequency],
    )
    fig = smr.SimuranFigure(fig, filename=output_path)
    fig.save()


def plot_average_signal_psd(psd_dataframe, output_path, max_frequency=30):
    l = []
    headers = ["Power (Db)", "Frequency (Hz)", "Brain Region"]
    regions = sorted(list(set(psd_dataframe["region"])))
    for r in regions:
        psd = psd_dataframe.loc[psd_dataframe["label"] == f"{r}_avg"]
        l.extend(
            [x, y, r] for x, y in zip(psd["power"].array[0], psd["frequency"].array[0])
        )
    avg_df = list_to_df(l, headers=headers)
    fig, ax = plt.subplots()
    sns.lineplot(
        ax=ax,
        x="Frequency (Hz)",
        y="Power (Db)",
        hue="Brain Region",
        data=avg_df[avg_df["Frequency (Hz)"] < max_frequency],
    )
    fig = smr.SimuranFigure(fig, filename=output_path)
    fig.save()


def plot_psd_over_signals(normal_psds, output_path, max_frequency=30):
    fig, ax = plt.subplots()
    normal_psds["Brain Region"] = normal_psds["region"]
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
        data=df[df["Frequency (Hz)"] < max_frequency],
    )
    fig = smr.SimuranFigure(fig, filename=output_path)
    fig.save()


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


def plot_psds(recording, out_dir, max_frequency):
    psd_dataframe = create_psd_table(recording.data)
    normal_psds, _ = split_psds(*grab_psds(recording.data))

    name_save = recording.get_name_for_save()
    paths = [
        out_dir / "split" / f"{name_save}--split",
        out_dir / "average" / f"{name_save}--average",
        out_dir / "normal_psds" / f"{name_save}--normal_psds",
    ]
    plot_split_psd(psd_dataframe, paths[0], max_frequency)
    plot_average_signal_psd(grab_psds(recording.data)[0], paths[1], max_frequency)
    plot_psd_over_signals(normal_psds, paths[2], max_frequency)
    return psd_dataframe, paths


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


def group_type_from_rat_name(name):
    return "Lesion" if name.lower().startswith("l") else "Control"


def main(df_path, config_path, output_path, out_dir):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    max_frequency = config["max_psd_freq"]

    with open(output_path, "w") as f:
        for r in rc.load_iter():
            psd_df, paths = plot_psds(r, out_dir, max_frequency)
            f.writelines([f"{path.name}\n" for path in paths])


def summary(df_path, config_path, out_dir):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    per_animal_psds = []
    max_frequency = config["max_psd_freq"]

    smr.set_plot_style()
    for r in rc.load_iter():
        rat_name = r.attrs["rat"]
        psd_df = create_psd_table(r.data)
        clean_df = psd_df[psd_df["Type"] == "Clean"]
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group_type_from_rat_name(rat_name))
        per_animal_psds.append(clean_df)
    full_df = pd.concat(per_animal_psds, ignore_index=True)
    path = out_dir / "per_animal_psds"
    plot_per_animal_psd(full_df, path, max_frequency)
    path = out_dir / "per_group_psds"
    plot_control_vs_lesion_psd(full_df, path, max_frequency)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])

    if snakemake.params.get("mode") == "summary":
        summary(
            snakemake.input[0],
            snakemake.config["simuran_config"],
            Path(snakemake.params["output_dir"]),
        )
    else:
        main(
            snakemake.input[0],
            snakemake.config["simuran_config"],
            snakemake.output[0],
            Path(snakemake.params["output_dir"]),
        )
