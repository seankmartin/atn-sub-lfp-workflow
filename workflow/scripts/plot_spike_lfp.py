from pathlib import Path

import numpy as np
import simuran as smr
from neurochat.nc_lfp import NLfp
from skm_pyutils.table import df_from_file, list_to_df


def plot_signals_rc(recording_container, out_dir):
    l = []
    for recording in recording_container.load_iter():
        output_path = out_dir / f"{recording.get_name_for_save()}--lfp"
    headers = []
    return list_to_df(l, headers=headers)


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


def get_unit_dict(recording):
    set_units = recording.attrs["units_to_use"]



def compute_spike_lfp(recording):
    pass

def main(input_df_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    plot_signals_rc(rc, out_dir)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]).parent,
        snakemake.config["simuran_config"],
    )
