import logging
import pickle
import traceback
from math import floor
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import simuran as smr
import yasa
from ripple_detection import Karlsson_ripple_detector, Kay_ripple_detector
from scipy.signal import decimate
from simuran.bridges.mne_bridge import convert_signals_to_mne
from skm_pyutils.table import df_to_file, list_to_df
from tqdm import tqdm

from sleep_utils import (
    create_events,
    ensure_sleeping,
    filter_ripple_band,
    mark_rest,
    spindles_exclude_resting,
)

module_logger = logging.getLogger("simuran.custom.sleep_analysis")


def main(
    input_path, out_dir, config, do_spindles=True, do_ripples=True, overwrite=False
):
    config, rc = setup(input_path, out_dir, config)
    all_spindles = []
    all_ripples = []

    if (out_dir / "spindles_backup.pkl").exists() and not overwrite:
        with open(out_dir / "spindles_backup.pkl", "rb") as f:
            all_spindles = pickle.load(f)
    else:
        all_spindles = []

    if (out_dir / "ripples_backup.pkl").exists() and not overwrite:
        with open(out_dir / "ripples_backup.pkl", "rb") as f:
            all_ripples = pickle.load(f)
    else:
        all_ripples = []

    for r in tqdm(rc.load_iter()):
        try:
            if "awake" in r.source_file:
                module_logger.warning(f"Not processing awake sleep")
                continue
            if not ensure_sleeping(r):
                module_logger.warning(f"Too much movement in {r.source_file} for sleep")
                continue
            module_logger.info(f"Processing {r.source_file} for spindles and ripples")
            ratio_resting, resting_groups, resting_intervals = find_resting(r, config)
            if len(resting_groups) == 0:
                module_logger.warning(
                    f"Not processing {r.source_file} as no resting blocks found"
                )
                continue
            if do_spindles:
                if r.source_file not in [a[0] for a in all_spindles]:
                    spindles = spindle_control(r, resting_groups, config)
                    all_spindles.append(
                        (
                            r.source_file,
                            spindles,
                            ratio_resting,
                            resting_groups,
                            r.attrs["duration"],
                        )
                    )
            if do_ripples:
                if r.source_file not in [a[0] for a in all_ripples]:
                    ripple_times = ripple_control(r, resting_groups, config)
                    all_ripples.append(
                        (
                            r.source_file,
                            ripple_times,
                            ratio_resting,
                            resting_groups,
                            r.attrs["duration"],
                        )
                    )

        except Exception as e:
            print(f"ERROR: sleep execution failed with {e}")
            traceback.print_exc()
            break

    if do_spindles:
        save_spindles(out_dir, all_spindles)

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


def save_spindles(out_dir, all_spindles):
    filename = out_dir / "spindles.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_spindles, outfile)
    filename = out_dir / "spindles_backup.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_spindles, outfile)
    l = []
    for spindles in all_spindles:
        source_file, sp_dict, ratio_resting, resting_group, duration = spindles
        for k, spindle_times in sp_dict.items():
            num_spindles = len(spindle_times)
            l.append(
                [
                    source_file,
                    ratio_resting,
                    resting_group,
                    k,
                    spindle_times,
                    num_spindles,
                    60 * num_spindles / (ratio_resting * duration),
                ]
            )
    headers = [
        "Filename",
        "Resting Ratio",
        "Resting Times",
        "Brain Region",
        "Spindle Times",
        "Number of Spindles",
        "Spindles per Minute",
    ]
    df = list_to_df(l, headers=headers)
    df_to_file(df, out_dir / "spindles.csv")


def save_ripples(out_dir, all_ripples):
    filename = out_dir / "ripples.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_ripples, outfile)
    filename = out_dir / "ripples_backup.pkl"
    with open(filename, "wb") as outfile:
        pickle.dump(all_ripples, outfile)
    l = []
    for val in all_ripples:
        fname, ripple_times, ratio_resting, resting_groups, duration = val
        for k, v in ripple_times.items():
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
                    len(times),
                    len(n_times),
                    60 * len(times) / (ratio_resting * duration),
                ]
            )
    headers = [
        "Filename",
        "Ripple Times",
        "Move Ripple Times",
        "Resting Ratio",
        "Resting Times",
        "Detector",
        "Brain Region",
        "Number of Ripples",
        "Number of Non-rest Ripples",
        "Ripples per Minute",
    ]
    df = list_to_df(l, headers=headers)
    df_to_file(df, out_dir / "ripples.csv")


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
        else:
            spindles[br] = []
    return spindles


def find_resting(r, config):
    nwbfile = r.data
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    brain_regions = list(nwbfile.electrodes.to_dataframe()["location"])
    num_sub, num_ca1 = 0, 0
    for br in brain_regions:
        if br == "SUB":
            num_sub += 1
        elif br == "CA1":
            num_ca1 += 1
    name = "SUB_avg" if num_sub >= num_ca1 else "CA1_avg"
    sub_signal = nwbfile.processing["average_lfp"][name].data[:]
    sub_rate = nwbfile.processing["average_lfp"][name].rate

    resting_intervals, intervaled = mark_rest(
        speed, sub_signal, sub_rate, speed_rate, **config
    )
    if (len(resting_intervals) > 0) and (
        resting_intervals[-1][-1] > r.attrs["duration"]
    ):
        resting_intervals[-1][-1] = r.attrs["duration"]
    duration = r.attrs["duration"]
    if abs(timestamps[-1] - r.attrs["duration"]) > 0.2:
        raise RuntimeError("Mismatched duration and data")
    resting_bits = sum((r[1] - r[0] for r in resting_intervals))
    ratio_resting = resting_bits / duration
    if ratio_resting > 1:
        raise RuntimeError(
            f"Incorrect resting amount {ratio_resting}, duration was {duration}, resting was {resting_bits}, intervals {resting_intervals}"
        )
    return ratio_resting, resting_intervals, intervaled


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
        mne_data_br = mne_data.copy().pick(chans)
        sp = yasa.spindles_detect(
            mne_data_br,
            thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 2.5},
            freq_sp=(12, 15),
            verbose=False,
            multi_only=True,
        )
        sp_res[brain_region] = sp
    return sp_res


def ripple_control(r, resting, config):
    use_first_two = config["use_first_two_for_ripples"]
    only_kay_detect = config["only_kay_detect"]
    if only_kay_detect:
        ripple_detect = {"Kay": Kay_ripple_detector}
    else:
        ripple_detect = {
            "Kay": Kay_ripple_detector,
            "Karlsson": Karlsson_ripple_detector,
        }

    nwbfile = r.data
    lfp_egf = nwbfile.processing["high_rate_ecephys"]["LFP"]["ElectricalSeries"]
    lfp_rate = lfp_egf.rate
    lfp_data = lfp_egf.data
    brain_regions = list(nwbfile.electrodes.to_dataframe()["location"])
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    timestamps = nwbfile.processing["behavior"]["running_speed"].timestamps[:]
    speed_rate = np.mean(np.diff(timestamps))
    on_target = r.attrs["RSC on target"]
    desired_rate = config["lfp_ripple_rate"]
    downsampling_factor = floor(lfp_rate / desired_rate)
    new_rate = int(lfp_rate / downsampling_factor)
    speed_long = np.repeat(speed, int(new_rate * speed_rate))

    if abs(len(speed_long) - r.attrs["duration"] * desired_rate) > 5:
        raise ValueError("Non matching speed and duration in sleep analysis")

    return extract_lfp_data_and_do_ripples(
        use_first_two,
        lfp_rate,
        new_rate,
        lfp_data,
        brain_regions,
        on_target,
        downsampling_factor,
        resting,
        ripple_detect,
        speed_long,
    )


def extract_lfp_data_and_do_ripples(
    use_first_two,
    lfp_rate,
    new_rate,
    lfp_data,
    brain_regions,
    on_target,
    downsampling_factor,
    resting,
    ripple_detectors,
    speed_long,
):
    time = None
    final_dict = {}
    for ripple_detect_name, ripple_detect in ripple_detectors.items():
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
            if np.sum(np.abs(lfp_data_sub)) <= 1.0:
                if len(brain_region_indices) == 2:
                    logging.warning(f"{brain_region} has no data")
                    continue
                indices_to_use = brain_region_indices[2:4]
                lfp_data_sub = lfp_data[:, indices_to_use].T
            filtered_lfps = filter_ripple_band(lfp_data_sub, lfp_rate).T

            if downsampling_factor != 1:
                filtered_lfps = decimate(
                    filtered_lfps, downsampling_factor, zero_phase=True, axis=0
                )
            if time is None:
                time = [i / new_rate for i in range(filtered_lfps.shape[0])]

            res = ripples(
                resting,
                ripple_detect,
                new_rate,
                speed_long,
                filtered_lfps,
                time,
            )
            final_dict[f"{ripple_detect_name}_{brain_region}"] = res
    return final_dict


def ripples(resting, ripple_detect, lfp_rate, speed_long, filtered_lfps, time):
    ripple_times = ripple_detect(
        time,
        filtered_lfps,
        speed_long,
        lfp_rate,
        speed_threshold=2.5,
        minimum_duration=0.015,
        zscore_threshold=2.0,
        smoothing_sigma=0.004,
        close_ripple_threshold=0.05,
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
    try:
        snakemake
    except NameError:
        using_snakemake = False
    else:
        using_snakemake = True
    if using_snakemake:
        smr.set_only_log_to_file(snakemake.log[0])
        module_logger.setLevel(logging.DEBUG)
        logging.getLogger("simuran.custom.sleep_utils").setLevel(logging.DEBUG)
        main(
            snakemake.input[0],
            Path(snakemake.output[0]).parent,
            snakemake.config["simuran_config"],
            True,
            True,
            snakemake.config["overwrite_sleep"],
        )
    else:
        here = Path(__file__).parent.parent.parent
        input_path = here / "results" / "every_processed_nwb.csv"
        out_dir = here / "results" / "sleep"
        config_path = here / "config" / "simuran_params.yml"
        do_spindles = True
        do_ripples = True
        main(input_path, out_dir, config_path, do_spindles, do_ripples)
