"""Process openfield LFP into power spectra etc. saved to NWB"""

from pathlib import Path

import simuran as smr
from simuran.main import process_config

here = Path(__file__).resolve().parent


def main(
    table_path,
    config_path,
    function_path,
    output_directory,
    num_cpus,
):
    kwargs = process_config(
        datatable_filepath=table_path,
        config_filepath=config_path,
        function_filepath=function_path,
        data_filter=None,
        output_directory=output_directory,
    )
    kwargs["num_cpus"] = num_cpus
    kwargs["config_params"]["loader"] = "nwb"
    kwargs["config_params"]["loader_kwargs"] = {}
    results, rc = smr.main_with_data(**kwargs)


if __name__ == "__main__":
    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.params["function_path"],
        Path(snakemake.output[0]).parent,
        snakemake.threads,
    )
