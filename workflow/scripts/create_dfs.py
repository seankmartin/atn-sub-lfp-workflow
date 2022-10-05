"""Create summary dataframes for statistics"""
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from common import numpy_to_nc, rename_rat


def main(input_, output_dir, config_path):
    config = smr.config_from_file(config_path)
    datatable = df_from_file(input_)
    loader = smr.loader("nwb")
    datatable.loc[:, "rat"] = datatable["rat"].map(lambda x: rename_rat(x))
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    power_spectra_summary(rc, output_dir, config)
    openfield_coherence(rc, output_dir, config)
    openfield_speed(rc, output_dir)


def power_spectra_summary(rc, out_dir, config):
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
        ctrl = "Control (ATN)"
        lesion = "Lesion  (ATNx)"
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

    per_signal_dfs = []
    per_psd_dfs = []
    sum_dfs = []

    delta_min, delta_max = config["delta_min"], config["delta_max"]
    theta_min, theta_max = config["theta_min"], config["theta_max"]
    low_gamma_min, low_gamma_max = config["low_gamma_min"], config["low_gamma_max"]
    high_gamma_min, high_gamma_max = config["high_gamma_min"], config["high_gamma_max"]
    beta_min, beta_max = config["beta_min"], config["beta_max"]
    for r in rc.load_iter():
        rat_name = r.attrs["rat"]
        group = group_type_from_rat_name(rat_name)
        on_target = r.attrs["RSC_on_target"]

        clean_df = convert_df_to_averages(grab_psds(r.data)[0])
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group)
        clean_df = clean_df.assign(RSC_on_target=on_target)
        per_signal_dfs.append(clean_df)
        regions = sorted(list(set(clean_df["Brain Region"])))

        rel_power = []
        for region in regions:
            signal = r.data.processing["average_lfp"][f"{region}_avg"]
            signal = numpy_to_nc(signal.data[:], signal.rate)
            bands = [
                (delta_min, delta_max),
                (theta_min, theta_max),
                (beta_min, beta_max),
                (low_gamma_min, low_gamma_max),
                (high_gamma_min, high_gamma_max),
            ]
            for (l_band, h_band) in bands:
                p = signal.bandpower([l_band, h_band], window_sec=4, unit="milli")
                rel_power.append(p["relative_power"])
        rel_power.extend([group, on_target])
        headers = []
        for region in regions:
            headers.append(f"{region} Delta Rel")
            headers.append(f"{region} Theta Rel")
            headers.append(f"{region} Beta Rel")
            headers.append(f"{region} Low Gamma Rel")
            headers.append(f"{region} High Gamma Rel")
        headers.extend(["Condition, RSC on target"])
        sum_dfs.append(list_to_df([rel_power], headers=headers))

        psd_df = create_psd_table(r.data)
        clean_df = psd_df[psd_df["Type"] == "Clean"]
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group)
        clean_df = clean_df.assign(RSC_on_target=on_target)
        per_psd_dfs.append(clean_df)

    full_df = pd.concat(per_signal_dfs, ignore_index=True)
    animal_df = pd.concat(per_psd_dfs, ignore_index=True)
    sum_df = pd.concat(sum_dfs, ignore_index=True)

    df_to_file(full_df, out_dir / "averaged_signals_psd.csv")
    df_to_file(animal_df, out_dir / "averaged_psds_psd.csv")
    df_to_file(sum_df, out_dir / "signal_bandpowers.csv")


def openfield_coherence(rc, out_dir, config):
    theta_min, theta_max = config["theta_min"], config["theta_max"]
    delta_min, delta_max = config["delta_min"], config["delta_max"]
    beta_min, beta_max = config["beta_min"], config["beta_max"]

    def create_coherence_df(recording_container):
        l = []
        peak_coherences = []
        for recording in recording_container.load_iter():
            nwbfile = recording.data
            coherence_df = nwbfile.processing["lfp_coherence"][
                "coherence_table"
            ].to_dataframe()
            region = coherence_df["label"].values[0]
            group = recording.attrs["treatment"]
            this_bit = [
                [group, region, f_val, c_val]
                for f_val, c_val in zip(
                    coherence_df["frequency"].values[0],
                    coherence_df["coherence"].values[0],
                )
            ]
            this_df = list_to_df(this_bit, headers=["G", "R", "F", "C"])
            l.extend(this_bit)
            theta_coherence = this_df[
                (this_df["F"] >= theta_min) & (this_df["F"] <= theta_max)
            ]
            peak_theta_coherence = max(theta_coherence["C"])
            delta_coherence = this_df[
                (this_df["F"] >= delta_min) & (this_df["F"] <= delta_max)
            ]
            peak_delta_coherence = max(delta_coherence["C"])
            beta_coherence = this_df[
                (this_df["F"] >= beta_min) & (this_df["F"] <= beta_max)
            ]
            peak_beta_coherence = max(beta_coherence["C"])
            peak_coherences.append(
                [
                    peak_theta_coherence,
                    peak_delta_coherence,
                    peak_beta_coherence,
                    group,
                ]
            )
        headers = ["Group", "Regions", "Frequency (Hz)", "Coherence"]
        headers2 = [
            "Peak Theta Coherence",
            "Peak Delta Coherence",
            "Peak Beta Coherence",
            "Condition",
        ]
        return list_to_df(l, headers), list_to_df(peak_coherences, headers2)

    coherence_df, stats_df = create_coherence_df(rc)
    df_to_file(coherence_df, out_dir / "openfield_coherence.csv")
    df_to_file(stats_df, out_dir / "coherence_stats.csv")


def openfield_speed(rc, out_dir):
    def df_from_rc(recording_container):
        dfs = []
        for recording in recording_container.load_iter():
            nwbfile = recording.data
            speed_df = nwbfile.processing["speed_theta"][
                "speed_lfp_table"
            ].to_dataframe()
            speed_df.loc[:, "Group"] = recording.attrs["treatment"].capitalize()
            speed_df.loc[:, "Condition"] = speed_df.loc[:, "Group"]
            dfs.append(speed_df)
        return pd.concat(dfs, ignore_index=True)

    speed_df = df_from_rc(rc)
    df_to_file(speed_df, out_dir / "openfield_speed.csv")


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
