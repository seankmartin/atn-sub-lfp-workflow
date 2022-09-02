"""Create summary dataframes for statistics"""
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
from neurochat.nc_lfp import NLfp
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from common import rename_rat

module_logger = logging.getLogger("simuran.custom.create_dfs")


def numpy_to_nc(data, sample_rate=None, timestamp=None):
    if timestamp is None and sample_rate is None:
        raise ValueError("Must provide either sample_rate or timestamp")
    if timestamp is None:
        timestamp = np.arange(0, len(data), dtype=np.float32) / sample_rate
    elif sample_rate is None:
        sample_rate = 1 / np.mean(np.diff(timestamp))

    lfp = NLfp()
    lfp._set_samples(data)
    lfp._set_sampling_rate(sample_rate)
    lfp._set_timestamp(timestamp)
    lfp._set_total_samples(len(data))
    return lfp


def main(inputs, output_dir, config_path):
    config = smr.config_from_file(config_path)
    datatable = df_from_file(inputs[0])
    loader = smr.loader("nwb")
    datatable.loc[:, "rat"] = datatable["rat"].map(lambda x: rename_rat(x))
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    power_spectra_summary(rc, output_dir, config)
    openfield_coherence(rc, output_dir, config)
    openfield_speed(rc, output_dir)

    n_shuffles = config["num_spike_shuffles"]
    open_df = df_from_file(inputs[1])
    open_df.loc[:, "rat"] = open_df["rat"].map(lambda x: rename_rat(x))
    cells_rc = smr.RecordingContainer.from_table(open_df, loader=loader)
    openfield_spike_lfp(cells_rc, output_dir, n_shuffles, config)

    musc_df = df_from_file(inputs[2])
    musc_df.loc[:, "rat"] = musc_df["rat"].map(lambda x: rename_rat(x))
    musc_rc = smr.RecordingContainer.from_table(musc_df, loader=loader)
    muscimol_spike_lfp(musc_rc, output_dir, n_shuffles, config)


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

    per_signal_dfs = []
    per_psd_dfs = []
    sum_dfs = []

    theta_min, theta_max = config["theta_min"], config["theta_max"]
    for r in rc.load_iter():
        rat_name = r.attrs["rat"]
        clean_df = convert_df_to_averages(grab_psds(r.data)[0])
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group_type_from_rat_name(rat_name))
        per_signal_dfs.append(clean_df)
        regions = sorted(list(set(clean_df["Brain Region"])))

        rel_power = []
        for region in regions:
            signal = r.data.processing["average_lfp"][f"{region}_avg"]
            signal = numpy_to_nc(signal.data[:], signal.rate)
            p = signal.bandpower([theta_min, theta_max], window_sec=4, unit="milli")
            rel_power.extend([p["bandpower"], p["relative_power"]])
        headers = []
        for region in regions:
            headers.append(f"{region} Theta (mV)")
            headers.append(f"{region} Theta Rel")
        sum_dfs.append(list_to_df([rel_power], headers=headers))

        psd_df = create_psd_table(r.data)
        clean_df = psd_df[psd_df["Type"] == "Clean"]
        clean_df = clean_df.assign(Rat=rat_name)
        clean_df = clean_df.assign(Group=group_type_from_rat_name(rat_name))
        per_psd_dfs.append(clean_df)

    full_df = pd.concat(per_signal_dfs, ignore_index=True)
    animal_df = pd.concat(per_psd_dfs, ignore_index=True)
    sum_df = pd.concat(sum_dfs, ignore_index=True)

    df_to_file(full_df, out_dir / "averaged_signals_psd.csv")
    df_to_file(animal_df, out_dir / "averaged_psds_psd.csv")
    df_to_file(sum_df, out_dir / "theta_power.csv")


def openfield_coherence(rc, out_dir, config):
    theta_min, theta_max = config["theta_min"], config["theta_max"]
    delta_min, delta_max = config["delta_min"], config["delta_max"]

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
            l.extend(
                [group, region, f_val, c_val]
                for f_val, c_val in zip(
                    coherence_df["frequency"].values[0],
                    coherence_df["coherence"].values[0],
                )
            )
            theta_coherence = coherence_df[
                (coherence_df["frequency"] >= theta_min)
                & (coherence_df["frequency"] <= theta_max)
            ]
            peak_theta_coherence = max(theta_coherence["coherence"])
            delta_coherence = coherence_df[
                (coherence_df["frequency"] >= delta_min)
                & (coherence_df["frequency"] <= delta_max)
            ]
            peak_delta_coherence = max(delta_coherence["coherence"])
            peak_coherences.append(peak_theta_coherence, peak_delta_coherence)
        headers = ["Group", "Regions", "Frequency (Hz)", "Coherence"]
        headers2 = ["Peak Theta Coherence", "Peak Delta Coherence"]
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
            dfs.append(speed_df)
        return pd.concat(dfs, ignore_index=True)

    speed_df = df_from_rc(rc)
    df_to_file(speed_df, out_dir / "openfield_speed.csv")


def openfield_spike_lfp(rc, output_dir, n_shuffles, config):
    theta_min, theta_max = config["theta_min"], config["theta_max"]
    sta_df, sfc_df, peak_df = convert_spike_lfp(rc, n_shuffles, theta_min, theta_max)
    df_to_file(sta_df, output_dir / "openfield_sta.csv")
    df_to_file(sfc_df, output_dir / "openfield_sfc.csv")
    df_to_file(peak_df, output_dir / "openfield_peak_sfc.csv")


def muscimol_spike_lfp(rc, output_dir, n_shuffles, config):
    theta_min, theta_max = config["theta_min"], config["theta_max"]
    sta_df, sfc_df, peak_df = convert_spike_lfp(rc, n_shuffles, theta_min, theta_max)
    df_to_file(sta_df, output_dir / "muscimol_sta.csv")
    df_to_file(sfc_df, output_dir / "muscimol_sfc.csv")
    df_to_file(peak_df, output_dir / "muscimol_peak_sfc.csv")


def convert_spike_lfp(recording_container, n_shuffles, theta_min, theta_max):
    def add_spike_lfp_info(
        sta_list,
        sfc_list,
        recording,
        animal,
        type_,
        spike_train,
        region,
        n_shuffles,
        theta_min,
        theta_max,
    ):
        signal = recording.data.processing["average_lfp"][f"{region}_avg"]
        lfp = numpy_to_nc(signal.data[:], sample_rate=signal.rate)
        sta, sfc, t, f = compute_spike_lfp(lfp, spike_train, n_shuffles * 4)
        sfc = sfc / 100
        shuffled_sta, shuffled_sfc = compute_shuffled_spike_lfp(
            lfp, spike_train, n_shuffles, len(sfc), len(sta)
        )
        sta_mean = np.mean(shuffled_sta, axis=0)
        sfc_mean = np.mean(shuffled_sfc, axis=0) / 100

        sta_list.extend(
            [
                [region, animal, type_, sta_i, t_i, r]
                for (sta_i, t_i, r) in zip(sta, t, sta_mean)
            ]
        )
        sfc_list.extend(
            [
                [region, animal, type_, sfc_i, f_i, r]
                for (sfc_i, f_i, r) in zip(sfc, f, sfc_mean)
            ]
        )

        theta_part = np.nonzero(np.logical_and(f >= theta_min, f <= theta_max))
        return np.nanmax(theta_part)

    def compute_spike_lfp(lfp, spike_train, nrep=500):
        g_data = lfp.plv(spike_train, mode="bs", fwin=[0, 20], nrep=nrep)
        sta = g_data["STAm"]
        sfc = g_data["SFCm"]
        t = g_data["t"]
        f = g_data["f"]

        return sta, sfc, t, f

    def compute_shuffled_spike_lfp(
        lfp, spike_train, n_shuffles, sfc_length, sta_length
    ):
        shuffled_times = shift_spike_times(spike_train, n_shuffles, None)
        shuffle_sfc = np.zeros(shape=(n_shuffles, sfc_length))
        shuffle_sta = np.zeros(shape=(n_shuffles, sta_length))

        for i in range(n_shuffles):
            sta_sub_rand, sfc_sub_rand, _, _ = compute_spike_lfp(
                lfp, shuffled_times[i], nrep=20
            )
            shuffle_sfc[i] = sfc_sub_rand
            shuffle_sta[i] = sta_sub_rand

        return shuffle_sta, shuffle_sfc

    def shift_spike_times(spike_train, n_shuffles, limit=None):
        """
        Randomly shift the spike times for the currently set unit.

        Parameters
        ----------
        n_shuffles : int
            The number of times to shuffle.
        limit : int
            How much to shuffle by in seconds.
            limit = None implies enirely random shuffle
            limit = 'x' implies shuffles in the range [-x x]

        Returns
        -------
        np.ndarray
            The shifted spike times, shape (n_shuffles, n_spikes)

        """
        dur = spike_train[-1]
        low, high = (-dur, dur) if limit is None else (-limit, limit)
        shift = np.random.uniform(low=low, high=high, size=n_shuffles)
        ftimes = spike_train
        shift_ftimes = np.zeros(shape=(n_shuffles, len(ftimes)), dtype=np.float64)
        for i in np.arange(n_shuffles):
            shift_ftimes[i] = ftimes + shift[i]
            shift_ftimes[i][shift_ftimes[i] > dur] -= dur
            shift_ftimes[i][shift_ftimes[i] < 0] += dur
            shift_ftimes[i] = np.sort(shift_ftimes[i])

        return shift_ftimes

    sta_list = []
    sfc_list = []
    peak_vals = []
    for i in range(len((recording_container))):
        to_log = recording_container[i].attrs["nwb_file"]
        units = recording_container[i].attrs["units"]
        units = ast.literal_eval(units)
        if not isinstance(units, list):
            module_logger.debug(f"No units for spike lfp in {to_log}")
            continue
        module_logger.debug(f"Adding data for {to_log}")
        recording = recording_container.load(i)
        unit_types = ast.literal_eval(recording.attrs["unit_types"])
        animal = recording.attrs["treatment"]
        unit_table = recording.data.units.to_dataframe()
        electrodes = recording.data.electrodes.to_dataframe()
        brain_regions = sorted(list(set(electrodes["location"])))
        for unit, type_ in zip(units, unit_types):
            spike_train = unit_table.loc[unit_table["tname"] == unit].spike_times
            if len(spike_train) == 0:
                raise ValueError(
                    f"unit {unit} not found in table with units {unit_table['tname']}"
                )
            spike_train = spike_train.iloc[0]
            for region in brain_regions:
                module_logger.debug(f"Processing region {region} unit {unit}")
                avg_lfp = recording.data.processing["average_lfp"]
                if f"{region}_avg" not in avg_lfp.data_interfaces:
                    continue
                peak_theta_coh = add_spike_lfp_info(
                    sta_list,
                    sfc_list,
                    recording,
                    animal,
                    type_,
                    spike_train,
                    region,
                    n_shuffles,
                    theta_min,
                    theta_max,
                )
                peak_vals.append(peak_theta_coh, animal, region, type_)

    headers = ["Region", "Group", "Spatial", "STA", "Time (s)", "Shuffled STA"]
    sta_df = list_to_df(sta_list, headers=headers)
    headers = ["Region", "Group", "Spatial", "SFC", "Frequency (Hz)", "Shuffled SFC"]
    sfc_df = list_to_df(sfc_list, headers=headers)
    headers = ["Peak Theta SFC", "Rat", "Region", "Spatial"]
    peak_df = list_to_df(peak_vals, headers=headers)

    if sta_df["Group"].str.startswith("musc").any():
        sta_df["Treatment"] = sta_df["Spatial"]
        sfc_df["Treatment"] = sfc_df["Spatial"]
        peak_df["Treatment"] = peak_df["Spatial"]
    return sta_df, sfc_df, peak_df


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input,
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
