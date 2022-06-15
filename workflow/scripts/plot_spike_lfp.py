import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import simuran as smr
from neurochat.nc_lfp import NLfp
from skm_pyutils.table import df_from_file, df_to_file, list_to_df


def plot_sta(sta_df, out_dir):
    brain_regions = sorted(list(set(sta_df["Region"])))
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
            style="Group",
            hue="Spatial",
            ci=None,
        )
        smr.despine()
        ax.set_ylabel("Spike triggered average")
        out_name = out_dir / f"{region}_average_sta{name}"
        smr_fig = smr.SimuranFigure(fig=fig, filename=out_name)
        smr_fig.save()


def plot_sfc(sta_df, out_dir):
    brain_regions = sorted(list(set(sta_df["Region"])))
    name_iter = zip(["", "_shuffled"], ["SFC", "Shuffled SFC"])
    for region, (name, y) in itertools.product(brain_regions, name_iter):
        df_part = sta_df[sta_df["Region"] == region]
        smr.set_plot_style()
        fig, ax = plt.subplots()
        sns.lineplot(
            data=df_part,
            x="Frequency (Hz)",
            y=y,
            ax=ax,
            style="Group",
            hue="Spatial",
            ci=None,
        )
        smr.despine()
        ax.set_ylabel("Spike field coherence")
        out_name = out_dir / f"{region}_average_sfc{name}"
        smr_fig = smr.SimuranFigure(fig=fig, filename=out_name)
        smr_fig.save()


def convert_spike_lfp_to_df(recording_container):
    sta_list = []
    sfc_list = []
    for i in range(len((recording_container))):
        units = recording_container[i].attrs["units"]
        if np.isnan(units):
            continue
        recording = recording_container.load(i)
        unit_types = recording.attrs["unit_types"]
        animal = recording.attrs[""]
        unit_table = recording.nwbfile.units.to_dataframe()
        electrodes = recording.nwbfile.electrodes.to_dataframe()
        brain_regions = sorted(list(set(electrodes["location"])))
        for unit, type_ in zip(unit, unit_types):
            spike_train = unit_table.loc["unit"].spike_times
            for region in brain_regions:
                if f"{region}_avg" not in recording.nwbfile.processing["average_lfp"]:
                    continue
                add_spike_lfp_info(
                    sta_list, sfc_list, recording, animal, type_, spike_train, region
                )

    headers = ["Region", "Group", "Spatial", "STA", "Time (s)", "Shuffled STA"]
    sta_df = list_to_df(sta_list, headers=headers)
    headers = ["Region", "Group", "Spatial", "SFC", "Time (s)", "Shuffled SFC"]
    sfc_df = list_to_df(sfc_list, headers=headers)
    return sta_df, sfc_df


def add_spike_lfp_info(
    sta_list, sfc_list, recording, animal, type_, spike_train, region
):
    signal = recording.nwbfile.processing["average_lfp"][f"{region}_avg"]
    lfp = numpy_to_nc(signal.data[:], sample_rate=signal.rate)
    sta, sfc, t, f = compute_spike_lfp(lfp, spike_train)
    sfc = sfc / 100
    shuffled_sta, shuffled_sfc = compute_shuffled_spike_lfp(lfp)
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
        timestamp = [float(i) / sample_rate for i in range(len(data))]
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


def get_unit_dict(recording):
    set_units = recording.attrs["units_to_use"]


def compute_spike_lfp(recording):
    pass


def main(input_df_path, input_cell_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    cell_table = df_from_file(input_cell_path)
    merged_df = datatable.merge(
        cell_table,
        how="left",
        on="filename",
        validate="many_to_one",
        suffixes=(None, "_x"),
    )
    print(merged_df)
    df_to_file(merged_df, "test.csv")
    exit(-1)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(merged_df, loader=loader)
    plot_spike_lfp(rc, out_dir)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.input[1],
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
