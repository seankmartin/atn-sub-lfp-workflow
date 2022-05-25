"""Process openfield LFP into power spectra etc. saved to NWB"""

from pathlib import Path

import numpy as np
import simuran as smr
from pynwb import TimeSeries
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file

from scripts.convert_to_nwb import add_lfp_array_to_nwb, add_nwb_electrode
from scripts.lfp_clean import LFPAverageCombiner, NWBSignalSeries

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

    mod = nwb_proc.create_processing_module(
        "lfp_power", "Store power spectra and spectograms"
    )
    mod.add_container(spectra)


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
        add_lfp_info(r, config)
        exit(-1)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        Path(snakemake.output[0]).parent,
        snakemake.threads,
    )
