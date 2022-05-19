"""Process openfield LFP into power spectra etc. saved to NWB"""

from datetime import datetime
from pathlib import Path

import simuran as smr
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.ecephys import LFP
from pynwb.file import Subject

here = Path(__file__).resolve().parent


def main(
    table_path,
    config_path,
    function_path,
    data_fpath,
    output_directory,
    num_cpus,
    data_dir,
):
    results, rc = smr.main_with_files(
        datatable_filepath=table_path,
        config_filepath=config_path,
        function_filepath=function_path,
        data_filter=data_fpath,
        output_directory=output_directory,
        num_cpus=num_cpus,
    )

    nwbfile = NWBFile(
        session_description="Processed openfield LFP data from ATNx SUB O'Mara lab",
        identifier="ATNx_SUB_LFP",
        session_start_time=datetime.now(),
        experimenter="Bethany Frost",
        lab="O'Mara lab",
        institution="TCD",
        related_publications="DOI:10.1523/JNEUROSCI.2868-20.2021",
    )

    for i in range(len(rc)):
        r = rc.load(i)
        results = r.results
        lfp_ts = LFP(name=r.get_name_for_save(rel_dir=data_dir) + "--LFP")
        lfp_ts.create_electrical_series(
            name="ALL-LFP",
            data=r.data["signals"],
            electrodes=
            )


if __name__ == "__main__":
    nwbfile = NWBFile(
        session_description="Processed openfield LFP data from ATNx SUB O'Mara lab",
        identifier="ATNx_SUB_LFP",
        session_start_time=datetime.now(),
        experimenter="Bethany Frost",
        lab="O'Mara lab",
        institution="TCD",
        related_publications="DOI:10.1523/JNEUROSCI.2868-20.2021",
    )
    nwbfile.subject = Subject(species="Lister Hooded rat", sex="M")
    device = nwbfile.create_device(
        name="Platinum-iridium wires",
        description="Bundles of 4 connected to 32-channel Axona microdrive",
        manufacturer="California Fine Wire"
    )
    nwbfile.add_electrode_column()
    print(nwbfile)

    main(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.params["function_path"],
        snakemake.config["openfield_filter"],
        Path(snakemake.output[0]).parent,
        snakemake.threads,
        snakemake.config["data_directory"],
    )
