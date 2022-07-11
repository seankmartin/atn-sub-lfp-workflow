import ast
import itertools
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from neurochat.nc_lfp import NLfp
from skm_pyutils.table import df_from_file, list_to_df

module_logger = logging.getLogger("simuran.custom.plot_spike_lfp")


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


def convert_spike_lfp_to_df(recording_container, n_shuffles):
    sta_list = []
    sfc_list = []
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
                add_spike_lfp_info(
                    sta_list,
                    sfc_list,
                    recording,
                    animal,
                    type_,
                    spike_train,
                    region,
                    n_shuffles,
                )

    headers = ["Region", "Group", "Spatial", "STA", "Time (s)", "Shuffled STA"]
    sta_df = list_to_df(sta_list, headers=headers)
    headers = ["Region", "Group", "Spatial", "SFC", "Frequency (Hz)", "Shuffled SFC"]
    sfc_df = list_to_df(sfc_list, headers=headers)

    if sta_df["Group"].str.startswith("musc").any():
        sta_df["Treatment"] = sta_df["Spatial"]
        sfc_df["Treatment"] = sfc_df["Spatial"]
    return sta_df, sfc_df


def add_spike_lfp_info(
    sta_list, sfc_list, recording, animal, type_, spike_train, region, n_shuffles
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


def numpy_to_nc(data, sample_rate=None, timestamp=None):
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


def compute_spike_lfp(lfp, spike_train, nrep=500):
    g_data = lfp.plv(spike_train, mode="bs", fwin=[0, 20], nrep=nrep)
    sta = g_data["STAm"]
    sfc = g_data["SFCm"]
    t = g_data["t"]
    f = g_data["f"]

    return sta, sfc, t, f


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


def compute_shuffled_spike_lfp(lfp, spike_train, n_shuffles, sfc_length, sta_length):
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


def plot_spike_lfp(recording_container, out_dir, n_shuffles):
    sta_df, sfc_df = convert_spike_lfp_to_df(recording_container, n_shuffles)
    plot_sta(sta_df, out_dir)
    plot_sfc(sfc_df, out_dir)


def main(input_df_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    n_shuffles = config.get("num_spike_shuffles")
    plot_spike_lfp(rc, out_dir, n_shuffles)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]),
        snakemake.config["simuran_config"],
    )
