"""Process openfield LFP into power spectra etc. saved to NWB"""
import itertools
import logging
from math import ceil, floor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import simuran as smr
from hdmf.common import DynamicTable
from pynwb import TimeSeries
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from convert_to_nwb import add_lfp_array_to_nwb, export_nwbfile
from frequency_analysis import calculate_psd
from lfp_clean import LFPAverageCombiner, NWBSignalSeries

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.process_lfp")


def describe_columns():
    return [
        {"name": "label", "type": str, "doc": "label of electrode"},
        {"name": "region", "type": str, "doc": "brain region of electrode"},
        {"name": "frequency", "type": np.ndarray, "doc": "frequency values in Hz"},
        {"name": "power", "type": np.ndarray, "doc": "power values in dB"},
        {"name": "max_psd", "type": float, "doc": "maximum power value (uV)"},
    ]


def describe_coherence_columns():
    return [
        {"name": "label", "type": str, "doc": "label of coherence pair"},
        {"name": "frequency", "type": np.ndarray, "doc": "frequency values in Hz"},
        {"name": "coherence", "type": np.ndarray, "doc": "coherence values unitless"},
    ]


def describe_speed_columns():
    return [
        {"name": "region", "type": str, "doc": "The brain region of the lfp"},
        {"name": "speed", "type": np.ndarray, "doc": "The binned average speed"},
        {
            "name": "power",
            "type": np.ndarray,
            "doc": "The average lfp power in the same bins",
        },
    ]


def bin_speeds_and_lfp(
    speed, lfp_signal, samples_per_second, speed_sr, lfp_sr, low_f=None, high_f=None
):
    time_to_use = get_sample_times(len(speed), samples_per_second, speed_sr)
    avg_speed = np.zeros_like(time_to_use, dtype=np.float32)
    lfp_amplitudes = np.zeros_like(time_to_use, dtype=np.float32)
    diff = 1 / (2 * samples_per_second)

    for i, t in enumerate(time_to_use):
        low_sample = floor((t - diff) * speed_sr)
        high_sample = ceil((t + diff) * speed_sr)

        avg_speed[i] = np.mean(speed[low_sample:high_sample])

        low_sample = floor((t - diff) * lfp_sr)
        high_sample = ceil((t + diff) * lfp_sr)
        if high_sample >= len(lfp_signal):
            module_logger.warning(
                f"Position data ({time_to_use[-1]}s) is longer than EEG data ({len(lfp_signal) / lfp_sr}s)"
            )
            avg_speed = avg_speed[: len(lfp_signal)]
            lfp_amplitudes = lfp_amplitudes[: len(lfp_signal)]
            break
        lfp_sample_200ms = lfp_signal[low_sample:high_sample]
        abs_power, total_power = power_of_signal(
            lfp_sample_200ms, lfp_sr, low_f, high_f
        )
        lfp_amplitudes[i] = abs_power / total_power

    pd_df = list_to_df(
        [avg_speed, lfp_amplitudes], transpose=True, headers=["Speed_", "power"]
    )
    pd_df["speed"] = np.around(pd_df["Speed_"])
    pd_df.drop("Speed_", axis=1, inplace=True)

    return pd_df


def get_sample_times(n_samples, samples_per_second, sr):
    skip_rate = int(sr / samples_per_second)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    return [i / sr for i in range(n_samples)][slicer]


def power_of_signal(lfp_signal, lfp_sr, low_f, high_f):
    slep_win = scipy.signal.hann(lfp_signal.size, False)
    f, psd = scipy.signal.welch(
        lfp_signal,
        fs=lfp_sr,
        window=slep_win,
        nperseg=len(lfp_signal),
        nfft=256,
        noverlap=0,
    )
    idx_band = np.logical_and(f >= low_f, f <= high_f)
    abs_power = scipy.integrate.simps(psd[idx_band], x=f[idx_band])
    total_power = scipy.integrate.simps(psd, x=f)

    return abs_power, total_power


def process_lfp(ss, config, type_):
    combiner = LFPAverageCombiner(
        z_threshold=config["z_score_threshold"],
        remove_outliers=True,
    )
    results_dict = combiner.combine(ss)

    clean_kwargs = config[type_]
    sub_ss = ss.select_electrodes(
        clean_kwargs["pick_property"], clean_kwargs["options"]
    )
    selected_res = combiner.combine(sub_ss)
    return results_dict, selected_res


def add_lfp_info(recording, config):
    ss = NWBSignalSeries(recording)
    ss.filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
    canulated = recording.data.subject.fields["subject_id"].startswith("CanCsCa")
    type_ = "can_clean_kwargs" if canulated else "clean_kwargs"
    results_all, results_picked = process_lfp(ss, config, type_)

    nwbfile = recording.data
    # nwb_proc = nwbfile.copy()
    nwb_proc = nwbfile
    did_anything = [store_normalised_lfp(ss, results_all, nwb_proc)]
    did_anything.append(store_average_lfp(results_picked, nwb_proc))
    did_anything.append(calculate_and_store_lfp_power(config, nwb_proc))
    did_anything.append(
        store_coherence(nwb_proc, flims=(config["fmin"], config["fmax"]))
    )
    did_anything.append(
        store_speed_theta(
            nwb_proc,
            config["speed_theta_samples_per_second"],
            config["theta_min"],
            config["theta_max"],
            recording.source_file,
        )
    )

    for d in did_anything:
        if d is not False:
            return nwb_proc, True

    return nwb_proc, False


def calculate_and_store_lfp_power(config, nwb_proc):
    if "lfp_power" in nwb_proc.processing:
        return False
    signals = nwb_proc.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
    brain_regions = sorted(list(set(nwb_proc.electrodes.to_dataframe()["location"])))
    br_avg = [f"{br}_avg" for br in brain_regions]
    average_signals = np.array(
        [nwb_proc.processing["average_lfp"][br].data[:] for br in br_avg]
    )
    all_sigs = np.concatenate((signals, average_signals), axis=0)
    regions = list(nwb_proc.electrodes.to_dataframe()["location"])
    regions.extend(brain_regions)
    labels = list(nwb_proc.electrodes.to_dataframe()["label"])
    labels.extend(br_avg)
    results_list = []
    for (sig, region, label) in zip(all_sigs, regions, labels):
        warn = True if label.endswith("_avg") else False
        f, Pxx, max_psd = calculate_psd(
            sig,
            scale="decibels",
            fmin=config["fmin"],
            fmax=config["fmax"],
            warn=warn,
        )
        if max_psd == 0:
            breakpoint()
        results_list.append([label, region, f, Pxx, max_psd])
    results_df = list_to_df(
        results_list, headers=["label", "region", "frequency", "power", "max_psd"]
    )
    results_df.index.name = "Index"
    hdmf_table = DynamicTable.from_dataframe(
        df=results_df, name="power_spectra", columns=describe_columns()
    )
    mod = nwb_proc.create_processing_module(
        "lfp_power", "Store power spectra and spectograms"
    )
    mod.add(hdmf_table)


def store_average_lfp(results_picked, nwb_proc):
    if "average_lfp" in nwb_proc.processing:
        return False
    mod = nwb_proc.create_processing_module(
        "average_lfp", "A single averaged LFP signal per brain region"
    )

    for brain_region, result in results_picked.items():
        ts = TimeSeries(
            name=f"{brain_region}_avg",
            data=result["average_signal"],
            unit="V",
            conversion=0.001,
            rate=250.0,
            starting_time=0.0,
            description="A single averaged LFP signal per brain region",
        )
        mod.add(ts)


def store_normalised_lfp(ss, results_all, nwb_proc):
    if "normalised_lfp" in nwb_proc.processing:
        return False
    mod = nwb_proc.create_processing_module(
        "normalised_lfp",
        "Store filtered and z-score normalised LFP, with outlier information",
    )
    lfp_array = np.zeros_like(ss.data)
    electrode_type = np.zeros(shape=(lfp_array.shape[0]), dtype=object)
    region_to_idx_dict = ss.group_by_brain_region(index=True)

    for brain_region, result in results_all.items():
        indices = region_to_idx_dict[brain_region]
        signals = result["signals"]
        good_idx = result["good_idx"]
        outliers = result["outliers"]
        outliers_idx = result["outliers_idx"]
        for sig, idx in zip(signals, good_idx):
            lfp_array[indices[idx]] = sig
            electrode_type[indices[idx]] = "Normal"
        for sig, idx in zip(outliers, outliers_idx):
            lfp_array[indices[outliers_idx]] = sig
            electrode_type[indices[outliers_idx]] = "Outlier"

    nwb_proc.add_electrode_column(
        name="clean",
        description="The LFP signal matches others from this brain region or is an outlier",
        data=list(electrode_type),
    )
    add_lfp_array_to_nwb(nwb_proc, len(ss.data), lfp_array.T, mod)


def store_coherence(nwb_proc, flims=None):
    if "lfp_coherence" in nwb_proc.processing:
        return False
    average_signals = nwb_proc.processing["average_lfp"]
    fields = average_signals.data_interfaces.keys()
    if len(fields) < 2:
        return False
    coherence_list = []
    for fd in sorted(itertools.combinations(fields, 2)):
        x = average_signals[fd[0]].data[:]
        y = average_signals[fd[1]].data[:]
        fs = average_signals[fd[0]].rate
        f, Cxy = scipy.signal.coherence(x, y, fs, nperseg=2 * fs)

        if flims is not None:
            fmin, fmax = flims
            f = f[np.nonzero((f >= fmin) & (f <= fmax))]
            Cxy = Cxy[np.nonzero((f >= fmin) & (f <= fmax))]

        key = f"{fd[0][:-4]}_{fd[1][:-4]}"
        coherence_list.append([key, f, Cxy])

    headers = ["label", "frequency", "coherence"]
    results_df = list_to_df(coherence_list, headers=headers)
    hdmf_table = DynamicTable.from_dataframe(
        df=results_df, name="coherence_table", columns=describe_coherence_columns()
    )
    mod = nwb_proc.create_processing_module("lfp_coherence", "Store coherence")
    mod.add(hdmf_table)


def store_speed_theta(nwbfile, samples_per_second, low_f, high_f, sf):
    if "speed_theta" in nwbfile.processing:
        return False
    electrodes = nwbfile.electrodes.to_dataframe()
    brain_regions = sorted(list(set(electrodes["location"])))
    speed = nwbfile.processing["behavior"]["running_speed"].data[:]
    speed_diff = np.mean(
        np.diff(nwbfile.processing["behavior"]["running_speed"].timestamps)
    )
    if speed_diff == 0:
        module_logger.error(f"No running speed detected in {sf}")
    speed_sr = int(1.0 / speed_diff)
    dfs = []
    for region in brain_regions:
        lfp_signal = nwbfile.processing["average_lfp"][f"{region}_avg"].data[:]
        lfp_sr = nwbfile.processing["average_lfp"][f"{region}_avg"].rate

        df = bin_speeds_and_lfp(
            speed, lfp_signal, samples_per_second, speed_sr, lfp_sr, low_f, high_f
        )
        df.loc[:, "region"] = region
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)

    hdmf_table = DynamicTable.from_dataframe(
        df=final_df, name="speed_lfp_table", columns=describe_speed_columns()
    )
    mod = nwbfile.create_processing_module(
        "speed_theta", "Store speed theta relationship"
    )
    mod.add(hdmf_table)


def main(table_paths, config_path, output_paths, num_cpus, overwrite=False):
    output_dfs = []
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus
    for table_path, output_path in zip(table_paths, output_paths[:-1]):
        datatable = df_from_file(table_path)
        loader = NWBLoader(mode="a") if overwrite else NWBLoader(mode="r")
        rc = smr.RecordingContainer.from_table(datatable, loader)
        out_df = datatable.copy()

        for i, r in enumerate(rc.load_iter()):
            fname = Path(r.source_file)
            fname = fname.parent.parent / "processed" / fname.name
            if not fname.is_file() or overwrite:
                module_logger.debug(f"Processing {r.source_file}")
                nwbfile, _ = add_lfp_info(r, config)
                export_nwbfile(fname, r, nwbfile, r._nwb_io, debug=True)
            else:
                module_logger.debug(f"Already processed {r.source_file}")
            row_idx = datatable.index[i]
            out_df.at[row_idx, "nwb_file"] = str(fname)
        output_dfs.append(out_df)
        df_to_file(out_df, output_path)
    final_df = pd.concat(output_dfs, ignore_index=True)
    final_df.drop_duplicates("nwb_file", inplace=True, ignore_index=True)
    df_to_file(final_df, output_paths[-1])


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input,
        snakemake.config["simuran_config"],
        snakemake.output,
        snakemake.threads,
    )
