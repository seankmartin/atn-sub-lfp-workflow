"""Process openfield LFP into power spectra etc. saved to NWB"""

from pathlib import Path

import numpy as np
import simuran as smr
from hdmf.container import Table
from pynwb import TimeSeries
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file, list_to_df

from .convert_to_nwb import add_lfp_array_to_nwb, write_nwbfile
from .scripts.frequency_analysis import calculate_psd
from .scripts.lfp_clean import LFPAverageCombiner, NWBSignalSeries

here = Path(__file__).resolve().parent


def process_lfp(ss, config):
    combiner = LFPAverageCombiner(
        z_threshold=config["z_score_threshold"],
        remove_outliers=True,
    )
    results_dict = combiner.combine(ss)

    ss.select_electrodes(config["group_name"], config["options"])
    selected_res = combiner.combine(ss)
    return results_dict, selected_res


def add_lfp_info(recording, config):
    ss = NWBSignalSeries(recording)
    ss.filter(config["fmin"], config["fmax"], **config["filter_kwargs"])
    results_all, results_picked = process_lfp(ss, config)

    nwbfile = recording.data
    nwb_proc = nwbfile.copy()
    store_normalised_lfp(ss, results_all, nwb_proc)
    store_average_lfp(results_picked, nwb_proc)
    calculate_and_store_lfp_power(config, nwb_proc)

    return nwb_proc


def calculate_and_store_lfp_power(config, nwb_proc):
    signals = nwb_proc.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
    average_signals = np.array(
        [
            nwb_proc.processing["average_lfp"]["SUB_avg"].data[:],
            nwb_proc.processing["average_lfp"]["RSC_avg"].data[:],
        ]
    )
    all_sigs = np.concatenate(signals, average_signals, axis=0)
    regions = list(nwb_proc.electrodes.to_dataframe()["location"])
    regions.extend(("SUB", "RSC"))
    labels = list(nwb_proc.electrodes.to_dataframe()["label"])
    labels.extend("SUB_avg", "RSC_avg")
    results_list = []
    for (sig, region, label) in zip(all_sigs, regions, labels):
        f, Pxx, max_psd = calculate_psd(
            sig,
            scale="decibels",
            fmin=config["fmin"],
            fmax=config["fmax"],
        )
        results_list.append(label, region, f, Pxx, max_psd)
    results_df = list_to_df(
        results_list, headers=["label", "region", "frequency", "power", "max_psd"]
    )
    hdmf_table = Table.from_dataframe(results_df, name="power_spectra")
    mod = nwb_proc.create_processing_module(
        "lfp_power", "Store power spectra and spectograms"
    )
    mod.add(hdmf_table)


def store_average_lfp(results_picked, nwb_proc):
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
        )
        mod.add(ts)


def store_normalised_lfp(ss, results_all, nwb_proc):
    mod = nwb_proc.create_processing_module(
        "normalised_lfp",
        "Store filtered and z-score normalised LFP, with outlier information",
    )
    lfp_array = np.zeroes_like(ss.data)
    electrode_type = []
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
        data=electrode_type,
    )
    add_lfp_array_to_nwb(nwb_proc, len(ss.data), lfp_array.T, mod)


def main(
    table_path,
    config_path,
    out_dir,
    num_cpus,
):
    datatable = df_from_file(table_path)
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus
    loader = NWBLoader()

    rc = smr.RecordingContainer.from_table(datatable, loader)

    for r in rc.load_iter():
        process_lfp(r, config)
        nwbfile = add_lfp_info(r, config)
        filename = out_dir / "processed_nwbfiles" / r.source_file.name
        write_nwbfile(filename, r, nwbfile)
        exit(-1)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]).parent,
        snakemake.threads,
    )
