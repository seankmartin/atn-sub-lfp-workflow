"""Create summary dataframes for statistics"""
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from common import rename_rat


def main(inputs, output_dir, config_path):
    config = smr.config_from_file(config_path)
    datatable = df_from_file(inputs[0])
    loader = smr.loader("nwb")
    datatable.loc[:, "rat"] = datatable["rat"].map(lambda x: rename_rat(x))
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    power_spectra_summary(rc, output_dir)
    openfield_coherence(rc, output_dir)
    openfield_speed(rc, output_dir)

def power_spectra_summary(rc, out_dir):
    def grab_psds(nwbfile):
        psd_table = nwbfile.processing["lfp_power"]["power_spectra"].to_dataframe()
        electrodes_table = nwbfile.electrodes.to_dataframe()
        return psd_table, electrodes_table

    def convert_df_to_averages(psd_dataframe):
        l = []
        headers = ["Power (Db)", "Frequency (Hz)", "Brain Region"]
        regions = sorted(list(set(psd_dataframe["region"])))
        for r in regions:
            psd = psd_dataframe.loc[psd_dataframe["label"] == f"{r}_avg"]
            l.extend(
                [x, y, r]
                for x, y in zip(psd["power"].array[0], psd["frequency"].array[0])
            )
        return list_to_df(l, headers=headers)

    def group_type_from_rat_name(name):
        ctrl = "Control (ATN,   N = 6)"
        lesion = "Lesion  (ATNx, N = 5)"
        return lesion if name.lower().startswith("l") else ctrl

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
                for (x, y) in zip(
                    average_psd_for_outlier, normal_psds.iloc[0]["frequency"]
                )
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

    per_group_dfs = []
    per_animal_dfs = []

    for r in rc.load_iter():
        rat_name = r.attrs["rat"]
        clean_df = convert_df_to_averages(grab_psds(r.data)[0])
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group_type_from_rat_name(rat_name))
        per_group_dfs.append(clean_df)

        psd_df = create_psd_table(r.data)
        clean_df = psd_df[psd_df["Type"] == "Clean"]
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group_type_from_rat_name(rat_name))
        per_animal_dfs.append(clean_df)

    full_df = pd.concat(per_group_dfs, ignore_index=True)
    animal_df = pd.concat(per_animal_dfs, ignore_index=True)
    df_to_file(full_df, out_dir / "averaged_signals_psd.csv")
    df_to_file(animal_df, out_dir / "averaged_psds_psd.csv")


def openfield_coherence(rc, out_dir):
    def create_coherence_df(recording_container):
        l = []
        for recording in recording_container.load_iter():
            nwbfile = recording.data
            coherence_df = nwbfile.processing["lfp_coherence"][
                "coherence_table"
            ].to_dataframe()
            region = coherence_df["label"].values[0]
            group = recording.attrs["treatment"]
            l.extend(
                [group, region, f_val, c_val]
                for f_val, c_val in zip(
                    coherence_df["frequency"].values[0],
                    coherence_df["coherence"].values[0],
                )
            )
        headers = ["Group", "Regions", "Frequency (Hz)", "Coherence"]
        return list_to_df(l, headers)

    coherence_df = create_coherence_df(rc)
    df_to_file(coherence_df, out_dir / "openfield_coherence.csv")


def openfield_speed(rc, out_dir):
    def df_from_rc(recording_container):
        dfs = []
        for recording in recording_container.load_iter():
            nwbfile = recording.data
            speed_df = nwbfile.processing["speed_theta"][
                "speed_lfp_table"
            ].to_dataframe()
            speed_df.loc[:, "Group"] = recording.attrs["treatment"].capitalize()
            dfs.append(speed_df)
        return pd.concat(dfs, ignore_index=True)

    speed_df = df_from_rc(rc)
    df_to_file(speed_df, out_dir / "openfield_speed.csv")


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input,
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
