import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
import yasa
from ripple_detection import Kay_ripple_detector, filter_ripple_band
from simuran.bridges.mne_bridge import convert_signals_to_mne
from tqdm import tqdm

from sleep_utils import (create_events, ensure_sleeping, mark_rest,
                         spindles_exclude_resting)

module_logger = logging.getLogger("simuran.custom.sleep_analysis")


def main(input_path, out_dir, config):
    config = smr.config_from_file(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    if len(sleep) == 0:
        module_logger.error(f"no sleep recordings found in {input_path}")
        return
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(sleep, loader)
    all_spindles = []
    all_ripples = []
    for r in tqdm(rc.load_iter()):
        electrodes = r.data.electrodes.to_dataframe()
        brain_regions = sorted(list(set(electrodes["location"])))
        if "SUB" not in brain_regions:
            module_logger.info(f"Skipping {r.source_file} due to no SUB signal")
            continue
        if not ensure_sleeping(r):
            module_logger.warning(f"Too much movement in {r.source_file} for sleep")
            continue
        module_logger.info(f"Processing {r.source_file} for spindles and ripples")
        spindles, resting_array = spindle_control(r, config)
        ripple_times = ripple_control(r, resting_array)
        all_spindles.append((r.source_file, spindles))
        all_ripples.append((r.source_file, ripple_times))
    filename = out_dir / "spindles.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_spindles, outfile)

    filename = out_dir / "ripples.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_ripples, outfile)


def spindle_control(r, config):
    resting_array = find_resting(r, config)
    mne_array = convert_to_mne(r, resting_array)
    sp = detect_spindles(mne_array)
    if sp is not None:
        sp = spindles_exclude_resting(sp.summary(), resting_array, mne_array)
    return sp, resting_array


def find_resting(r, config):
    nwbfile = r.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    sub_signal = nwbfile.processing["average_lfp"]["SUB_avg"].data[:]
    sub_rate = nwbfile.processing["average_lfp"]["SUB_avg"].rate
    return mark_rest(speed, sub_signal, sub_rate, speed_rate, **config)


def convert_to_mne(r, events):
    nwbfile = r.data
    lfp = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp.rate
    lfp_data = lfp.data[:].T
    on_target = r.attrs["RSC on target"]
    electrodes = nwbfile.electrodes.to_dataframe()
    signal_array = [
        smr.Eeg.from_numpy(lfp, lfp_rate)
        for i, lfp in enumerate(lfp_data)
        if on_target or (electrodes["location"][i] != "RSC")
    ]

    bad_chans = list(electrodes["clean"])
    ch_names = [
        f"{name}_{i}"
        for i, name in enumerate(electrodes["location"])
        if on_target or (electrodes["location"][i] != "RSC")
    ]
    mne_array = convert_signals_to_mne(signal_array, ch_names, bad_chans)
    return create_events(mne_array, events)


def detect_spindles(mne_data):
    """See https://github.com/raphaelvallat/yasa/tree/master/notebooks

    For demos.
    """
    return yasa.spindles_detect(
        mne_data,
        thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
        freq_sp=(12, 15),
        verbose="error",
    )


def ripple_control(r, resting):
    nwbfile = r.data
    lfp_egf = nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp_egf.rate
    lfp_data = lfp_egf.data[:]
    filtered_lfps = filter_ripple_band(lfp_data)
    time = [i / lfp_rate for i in range(filtered_lfps.shape[0])]
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    speed_long = np.repeat(speed, int(lfp_rate * speed_rate))
    ripple_times = Kay_ripple_detector(
        time,
        filtered_lfps,
        speed_long,
        lfp_rate,
        speed_threshold=2.5,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        smoothing_sigma=0.004,
        close_ripple_threshold=0.1,
    )
    eeg_rate = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    final_times, non_times = [], []
    for _, row in ripple_times.iterrows():
        t_start = int(row["start_time"] * eeg_rate)
        t_end = int(row["end_time"] * eeg_rate)
        if np.all(resting[t_start:t_end]):
            final_times.append((row["start_time"], row["end_time"]))
        else:
            non_times.append((row["start_time"], row["end_time"]))

    return final_times, non_times


if __name__ == "__main__":
    module_logger.setLevel(logging.DEBUG)
    try:
        snakemake
    except NameError:
        using_snakemake = False
    else:
        using_snakemake = True
    if using_snakemake:
        smr.set_only_log_to_file(snakemake.log[0])
        main(
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        input_path = here / "results" / "every_processed_nwb.csv"
        out_dir = here / "sleep"
        config_path = here / "config" / "simuran_params.yml"
        main(input_path, out_dir, config_path)
