"""Process openfield LFP into power spectra etc. saved to NWB"""
import itertools
import logging
from pathlib import Path

import numpy as np
import simuran as smr
from hdmf.common import DynamicTable
from pynwb import TimeSeries
from scipy.signal import coherence
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from convert_to_nwb import add_lfp_array_to_nwb, export_nwbfile
from scripts.frequency_analysis import calculate_psd
from scripts.lfp_clean import LFPAverageCombiner, NWBSignalSeries

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


def process_lfp(ss, config):
    combiner = LFPAverageCombiner(
        z_threshold=config["z_score_threshold"],
        remove_outliers=True,
    )
    results_dict = combiner.combine(ss)

    clean_kwargs = config["clean_kwargs"]
    sub_ss = ss.select_electrodes(
        clean_kwargs["pick_property"], clean_kwargs["options"]
    )
    selected_res = combiner.combine(sub_ss)
    return results_dict, selected_res


def add_lfp_info(recording, config):
    ss = NWBSignalSeries(recording)
    ss.filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
    results_all, results_picked = process_lfp(ss, config)

    nwbfile = recording.data
    # nwb_proc = nwbfile.copy()
    nwb_proc = nwbfile
    did_anything = [store_normalised_lfp(ss, results_all, nwb_proc)]
    did_anything.append(store_average_lfp(results_picked, nwb_proc))
    did_anything.append(calculate_and_store_lfp_power(config, nwb_proc))
    did_anything.append(
        store_coherence(nwb_proc, flims=(config["fmin"], config["fmax"]))
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
        f, Pxx, max_psd = calculate_psd(
            sig,
            scale="decibels",
            fmin=config["fmin"],
            fmax=config["fmax"],
        )
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
    coherence_list = []
    for fd in sorted(itertools.combinations(fields, 2)):
        x = average_signals[fd[0]].data[:]
        y = average_signals[fd[1]].data[:]
        fs = average_signals[fd[0]].rate
        f, Cxy = coherence(x, y, fs, nperseg=2 * fs)

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


def main(table_path, config_path, output_path, num_cpus, overwrite=False):
    datatable = df_from_file(table_path)
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus

    loader = NWBLoader(mode="a") if overwrite else NWBLoader(mode="r")

    rc = smr.RecordingContainer.from_table(datatable, loader)

    out_df = datatable.copy()

    for i, r in enumerate(rc.load_iter()):
        row_idx = datatable.index[i]
        module_logger.debug(f"Processing {r.source_file}")
        nwbfile, did_anything = add_lfp_info(r, config)
        if not did_anything:
            continue
        try:
            fname = Path(r.source_file)
            if overwrite:
                r._nwb_io.write(nwbfile)
            else:
                fname = fname.parent.parent / "processed" / fname.name
                export_nwbfile(fname, r, nwbfile, r._nwb_io)
        except Exception as e:
            module_logger.error(f"Failed to process {r.source_file}")
            raise (e)
        out_df.at[row_idx, "nwb_file"] = str(fname)
    df_to_file(out_df, output_path)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.output[0],
        snakemake.threads,
    )
