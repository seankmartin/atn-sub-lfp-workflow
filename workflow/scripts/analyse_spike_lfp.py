import ast
import logging
from pathlib import Path

import numpy as np
import simuran as smr
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from common import numpy_to_nc, rename_rat

module_logger = logging.getLogger("simuran.custom.create_dfs")


def main(inputs, output_dir, config_path):
    config = smr.config_from_file(config_path)
    loader = smr.loader("nwb")

    n_shuffles = config["num_spike_shuffles"]
    open_df = df_from_file(inputs[0])
    open_df.loc[:, "rat"] = open_df["rat"].map(lambda x: rename_rat(x))
    cells_rc = smr.RecordingContainer.from_table(open_df, loader=loader)
    openfield_spike_lfp(cells_rc, output_dir, n_shuffles, config)

    musc_df = df_from_file(inputs[1])
    musc_df.loc[:, "rat"] = musc_df["rat"].map(lambda x: rename_rat(x))
    musc_rc = smr.RecordingContainer.from_table(musc_df, loader=loader)
    muscimol_spike_lfp(musc_rc, output_dir, n_shuffles, config)


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

        theta_part = sfc[np.nonzero(np.logical_and(f >= theta_min, f <= theta_max))]
        return np.nanmax(theta_part)

    def compute_spike_lfp(lfp, spike_train, nrep=500):
        g_data = lfp.plv(spike_train, mode="bs", fwin=[0, 120], nrep=nrep)
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
        treatment = recording.attrs["treatment"]
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
                    treatment,
                    type_,
                    spike_train,
                    region,
                    n_shuffles,
                    theta_min,
                    theta_max,
                )
                peak_vals.append([peak_theta_coh, treatment, region, type_])

    headers = ["Region", "Group", "Spatial", "STA", "Time (s)", "Shuffled STA"]
    sta_df = list_to_df(sta_list, headers=headers)
    headers = ["Region", "Group", "Spatial", "SFC", "Frequency (Hz)", "Shuffled SFC"]
    sfc_df = list_to_df(sfc_list, headers=headers)
    headers = ["Peak Theta SFC", "Group", "Region", "Spatial"]
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
