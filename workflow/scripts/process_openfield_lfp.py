"""Process openfield LFP into power spectra etc. saved to NWB"""

from pathlib import Path

import simuran as smr
from simuran.loaders.nwb_loader import NWBLoader
from skm_pyutils.table import df_from_file

from scripts.lfp_clean import LFPAverageCombiner, NWBSignalSeries

here = Path(__file__).resolve().parent


def process_lfp(recording, config):
    ss = NWBSignalSeries(recording)
    print(ss.description)
    ss.select_electrodes("group_name", ["BE0", "BE1"])
    print(ss.description)
    combiner = LFPAverageCombiner(
        z_threshold=config["z_score_threshold"],
        remove_outliers=True,
    )
    combiner.combine(ss)


def add_lfp_info(recording, config):
    lfp_combiner_interface_obj = BLAH
    lfp_combiner = BLAH(lfp_combiner_interface_obj)
    extra_signals = lfp_combiner.combine()

    nwbfile = recording.data
    nwb_proc = nwbfile.copy()
    mod = nwb_proc.create_processing_module(
        "lfp_filtering", "Store filtered and average LFP signals"
    )

    mod.add_container(lfp)

    mod = nwb_proc.create_processing_module(
        "lfp_power", "Store power spectra and spectograms"
    )
    mod.add_container(spectra)


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
