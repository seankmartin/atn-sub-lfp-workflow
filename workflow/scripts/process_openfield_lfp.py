"""Process openfield LFP into power spectra etc. saved to NWB"""

import logging
from pathlib import Path

import numpy as np
import simuran as smr
from hdmf.common import DynamicTable
from pynwb import TimeSeries
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file, df_to_file, list_to_df

from convert_to_nwb import add_lfp_array_to_nwb, write_nwbfile
from scripts.frequency_analysis import calculate_psd
from scripts.lfp_clean import LFPAverageCombiner, NWBSignalSeries

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.process_lfp")


def describe_columns():
    return [
        {"name": "label", "type": str, "doc": "label of electrode"},
        {"name": "region", "type": str, "doc": "brain region of electrode"},
        {"name": "frequency", "type": np.ndarray, "doc": "frequency values"},
        {"name": "power", "type": np.ndarray, "doc": "power values in dB"},
        {"name": "max_psd", "type": float, "doc": "maximum power value (uV)"},
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
    all_sigs = np.concatenate((signals, average_signals), axis=0)
    regions = list(nwb_proc.electrodes.to_dataframe()["location"])
    regions.extend(("SUB", "RSC"))
    labels = list(nwb_proc.electrodes.to_dataframe()["label"])
    labels.extend(("SUB_avg", "RSC_avg"))
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


def main(
    table_path,
    config_path,
    output_path,
    num_cpus,
):
    datatable = df_from_file(table_path)
    config = smr.ParamHandler(source_file=config_path)
    config["num_cpus"] = num_cpus
    loader = NWBLoader(mode="a")
    out_dir = Path(output_path).parent

    rc = smr.RecordingContainer.from_table(datatable, loader)

    out_df = datatable.copy()

    for i, r in enumerate(rc.load_iter()):
        module_logger.debug(f"Processing {r.source_file}")
        nwbfile = add_lfp_info(r, config)
        try:
            r._nwb_io.write(nwbfile)
            fname = r.source_file
        except Exception as e:
            fname = None
            module_logger.error(f"Failed to process {r.source_file}")
            raise (e)
        row_idx = datatable.index[i]
        out_df.at[row_idx, "nwb_file"] = fname
    df_to_file(out_df, output_path)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.output[0],
        snakemake.threads,
    )
