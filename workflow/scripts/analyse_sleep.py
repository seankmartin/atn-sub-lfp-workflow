import logging
import pickle
from math import floor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import simuran as smr
import yasa
from ripple_detection import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    filter_ripple_band,
)
from scipy.signal import decimate
from simuran.bridges.mne_bridge import convert_signals_to_mne
from skm_pyutils.table import df_to_file, list_to_df
from tqdm import tqdm

from sleep_utils import (
    create_events,
    ensure_sleeping,
    mark_rest,
    spindles_exclude_resting,
)

module_logger = logging.getLogger("simuran.custom.sleep_analysis")


def main(input_path, out_dir, config, do_spindles=True, do_ripples=True):
    config, rc = setup(input_path, out_dir, config)
    all_spindles = []
    all_ripples = []
    for r in tqdm(rc.load_iter()):
        if not ensure_sleeping(r):
            module_logger.warning(f"Too much movement in {r.source_file} for sleep")
            continue
        module_logger.info(f"Processing {r.source_file} for spindles and ripples")
        resting_array, ratio_resting, resting_groups = find_resting(r, config)
        if do_spindles:
            spindles = spindle_control(r, resting_groups, config)
            all_spindles.append(
                (r.source_file, spindles, ratio_resting, resting_groups)
            )
        if do_ripples:
            ripple_times = ripple_control(r, resting_array, config)
            all_ripples.append(
                (r.source_file, ripple_times, ratio_resting, resting_groups)
            )
            # TODO temp
            break
    if do_spindles:
        do_spindles(out_dir, all_spindles)

    if do_ripples:
        save_ripples(out_dir, all_ripples)


def setup(input_path, out_dir, config):
    config = smr.config_from_file(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    cols = df.columns
    df[cols[2:]].loc[df[cols[2:]].sleep == 1]
    sleep = df.loc[df.sleep == 1]
    if len(sleep) == 0:
        raise FileNotFoundError(f"no sleep recordings found in {input_path}")

    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(sleep, loader)
    return config, rc


def do_spindles(out_dir, all_spindles):
    filename = out_dir / "spindles.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_spindles, outfile)


def save_ripples(out_dir, all_ripples):
    filename = out_dir / "ripples.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_ripples, outfile)

    l = []
    for val in all_ripples:
        fname, ripple_times, ratio_resting, resting_groups = val
        for k, v in ripple_times.iteritems():
            times, n_times = v
            detector, brain_region = k.split("_")
            l.append(
                [
                    fname,
                    times,
                    n_times,
                    ratio_resting,
                    resting_groups,
                    detector,
                    brain_region,
                ]
            )
    headers = [
        "Filename",
        "Ripple Times",
        "Move Ripple Times",
        "Resting Ratio",
        "Detector",
        "Brain Region",
    ]
    df = list_to_df(l, headers=headers)
    df_to_file(df, out_dir / "sleep" / "ripples.csv")


def spindle_control(r, resting_array, config):
    use_avg = config["spindles_use_avg"]
    if use_avg:
        mne_array = convert_to_mne_avg(r, resting_array)
    else:
        mne_array = convert_to_mne(r, resting_array)
    spindles = detect_spindles(mne_array)
    for (br, sp) in spindles.items():
        if sp is not None:
            spindles[br] = spindles_exclude_resting(sp.summary(), resting_array)
    return spindles


def find_resting(r, config):
    nwbfile = r.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    brain_regions = nwbfile.electrodes.to_dataframe()["location"]
    name = "SUB_avg" if "SUB" in brain_regions else "CA1_avg"
    sub_signal = nwbfile.processing["average_lfp"][name].data[:]
    sub_rate = nwbfile.processing["average_lfp"][name].rate

    resting, resting_intervals = mark_rest(
        speed, sub_signal, sub_rate, speed_rate, **config
    )
    ratio_resting = np.sum(resting) / len(resting)
    if ratio_resting > 1:
        raise RuntimeError(f"Incorrect resting amount {ratio_resting}")
    return resting, ratio_resting, resting_intervals


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


def convert_to_mne_avg(r, events):
    nwbfile = r.data
    brain_regions = sorted(list(set(nwbfile.electrodes.to_dataframe()["location"])))
    signal_array = []
    ch_names = []
    for region in brain_regions:
        if region == "RSC" and not r.attrs["RSC on target"]:
            continue
        lfp = nwbfile.processing["average_lfp"][f"{region}_avg"]
        signal_array.append(smr.Eeg.from_numpy(lfp.data[:], lfp.rate))
        ch_names.append(f"{region}_avg")

    mne_array = convert_signals_to_mne(signal_array, ch_names)
    return create_events(mne_array, events)


def detect_spindles(mne_data):
    """See https://github.com/raphaelvallat/yasa/tree/master/notebooks

    For demos.
    """
    ch_names = mne_data.info["ch_names"]
    brain_regions = sorted(list({ch[:3] for ch in ch_names}))
    sp_res = {}
    for brain_region in brain_regions:
        chans = mne.pick_channels_regexp(mne_data.info["ch_names"], f"^{brain_region}")
        ch_to_use = [mne_data.info["ch_names"][ch] for ch in chans]
        mne_data_br = mne_data.pick_channels(ch_to_use, ordered=True)
        sp = yasa.spindles_detect(
            mne_data_br,
            thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
            freq_sp=(12, 15),
            verbose="error",
            multi_only=True,
        )
        sp_res[brain_region] = sp
    return sp_res


def ripple_control(r, resting, config):
    use_first_two = config["use_first_two_for_ripples"]
    ripple_detect = {"Kay": Kay_ripple_detector, "Karlsson": Karlsson_ripple_detector}

    nwbfile = r.data
    lfp_egf = nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp_egf.rate
    lfp_data = lfp_egf.data
    brain_regions = nwbfile.electrodes.to_dataframe()["location"]
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    on_target = r.attrs["RSC on target"]
    eeg_rate = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    desired_rate = config["lfp_ripple_rate"]
    downsampling_factor = floor(lfp_rate / desired_rate)
    new_rate = int(lfp_rate / downsampling_factor)
    speed_long = np.repeat(speed, int(new_rate * speed_rate))
    print(downsampling_factor)

    return extract_lfp_data_and_do_ripples(
        r,
        use_first_two,
        lfp_rate,
        lfp_data,
        brain_regions,
        on_target,
        downsampling_factor,
        resting,
        ripple_detect,
        speed_long,
        eeg_rate,
    )


def extract_lfp_data_and_do_ripples(
    r,
    use_first_two,
    lfp_rate,
    lfp_data,
    brain_regions,
    on_target,
    downsampling_factor,
    resting,
    ripple_detectors,
    speed_long,
    eeg_rate,
):
    time = None
    final_dict = {}
    for ripple_detect_name, ripple_detect in ripple_detectors.iteritems():
        for brain_region in brain_regions:
            if (brain_region == "RSC") and (not on_target):
                continue
            brain_region_indices = [
                i for i in range(len(brain_regions)) if brain_regions[i] == brain_region
            ]
            indices_to_use = (
                brain_region_indices[:2] if use_first_two else brain_region_indices
            )
            lfp_data_sub = lfp_data[:, indices_to_use].T
            print(lfp_data_sub.shape)
            if downsampling_factor != 1:
                lfp_data_sub = decimate(
                    lfp_data_sub, downsampling_factor, zero_phase=True, axis=-1
                )
            print(lfp_data_sub.shape)
            print(r.attrs["duration"])

            filtered_lfps = filter_ripple_band(lfp_data_sub)
            if time is None:
                time = [i / lfp_rate for i in range(filtered_lfps.shape[0])]

            res = ripples(
                resting,
                ripple_detect,
                lfp_rate,
                speed_long,
                eeg_rate,
                filtered_lfps,
                time,
            )
            final_dict[f"{ripple_detect_name}_{brain_region}"] = res
    return final_dict


def ripples(
    resting, ripple_detect, lfp_rate, speed_long, eeg_rate, filtered_lfps, time
):
    ripple_times = ripple_detect(
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
    final_times, non_times = [], []
    for _, row in ripple_times.iterrows():
        t = row["start_time"]
        use_this_time = any(((t >= r[0]) and (t <= r[1]) for r in resting))
        if use_this_time:
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
        out_dir = here / "results" / "sleep"
        config_path = here / "config" / "simuran_params.yml"
        main(input_path, out_dir, config_path)
