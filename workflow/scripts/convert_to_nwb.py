"""Process openfield LFP into power spectra etc. saved to NWB"""

from pathlib import Path

import numpy as np
import simuran as smr
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from skm_pyutils.table import df_from_file, filter_table

here = Path(__file__).resolve().parent


def main(table_path, config_path, data_fpath, output_directory : Path, overwrite=False):
    table = df_from_file(table_path)
    config = smr.ParamHandler(source_file=config_path, name="params")
    filter_ = smr.ParamHandler(source_file=data_fpath, name="filter")
    filtered_table = filter_table(table, filter_)
    loader = smr.loader(config["loader"])(**config["loader_kwargs"])

    rc = smr.RecordingContainer.from_table(filtered_table, loader)
    
    for i in range(len(rc)):
        save_name = rc[i].get_name_for_save(config["cfg_base_dir"])
        filename = output_directory / "nwbfiles"/ f"{save_name}.nwb"
        
        if not overwrite and filename.is_file():
            continue

        r = rc.load(i)
        nwbfile = convert_recording_to_nwb(r, config["cfg_base_dir"])

        filename.parent.mkdir(parents=True, exist_ok=True)
        with NWBHDF5IO(filename, "w") as io:
            io.write(nwbfile)


def convert_recording_to_nwb(recording, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)

    nwbfile = NWBFile(
        session_description=f"Openfield recording for {name}",
        identifier=f"ATNx_SUB_LFP--{name}",
        session_start_time=recording.datetime,
        experiment_description="Relationship between ATN, SUB, RSC, and CA1",
        experimenter="Bethany Frost",
        lab="O'Mara lab",
        institution="TCD",
        related_publications="doi:10.1523/JNEUROSCI.2868-20.2021",
    )
    nwbfile.subject = Subject(
        species="Lister Hooded rat", sex="M", subject_id=recording.attrs["rat"]
    )
    piw_device = nwbfile.create_device(
        name="Platinum-iridium wires 25um thick",
        description="Bundles of 4 connected to 32-channel Axona microdrive",
        manufacturer="California Fine Wire",
    )
    be_device = nwbfile.create_device(
        name="Bipolar electrodes 75um thick",
        description="Bundles of 2 connected to 32-channel Axona microdrive -- only LFP",
        manufacturer="California Fine Wire",
    )
    nwbfile.add_electrode_column(name="label", description="label of electrode")
    num_electrodes = 4
    for i in range(2):
        brain_region = recording.data["signals"][i * 2].region
        electrode_group = nwbfile.create_electrode_group(
            name=f"BE{i}",
            device=be_device,
            location=brain_region,
            description=f"Bipolar electrodes {i} placed in {brain_region}",
        )
        for j in range(2):
            nwbfile.add_electrode(
                x=np.nan,
                y=np.nan,
                z=np.nan,
                imp=np.nan,
                location=brain_region,
                filtering="Notch filter at 50Hz",
                group=electrode_group,
                label=f"BE{i}_E{j}",
            )
    if len(recording.data["signals"]) == 32:
        num_electrodes = 32

    for i in range(7):
        if len(recording.data["signals"]) == 32:
            brain_region = recording.data["signals"][(i + 1) * 4].region
        else:
            brain_region = recording.data["units"][i+1].region
        electrode_group = nwbfile.create_electrode_group(
            name=f"TT{i}",
            device=piw_device,
            location=brain_region,
            description=f"Tetrode {i} electrodes placed in {brain_region}",
        )
        if len(recording.data["signals"]) == 32:
            for j in range(4):
                nwbfile.add_electrode(
                    x=np.nan,
                    y=np.nan,
                    z=np.nan,
                    imp=np.nan,
                    location=brain_region,
                    filtering="Notch filter at 50Hz",
                    group=electrode_group,
                    label=f"TT{i}_E{j}",
                )

    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(num_electrodes)), description="all electrodes"
    )

    lfp_data = np.transpose(
        np.array([s.samples.value for s in recording.data["signals"]])
    )
    compressed_data = H5DataIO(data=lfp_data, compression="gzip", compression_opts=4)
    lfp_electrical_series = ElectricalSeries(
        name="LFP",
        data=compressed_data,
        electrodes=all_table_region,
        starting_time=0.0,
        rate=250.0,
    )
    lfp = LFP(electrical_series=lfp_electrical_series)

    ecephys_module = nwbfile.create_processing_module(
        name="ecephys", description="Processed extracellular electrophysiology data"
    )
    ecephys_module.add(lfp)

    nwbfile.add_unit_column(name="tname", description="Tetrode and unit number")

    for i, unit_info in enumerate(recording.data["units"]):
        if unit_info.available_units is None:
            continue
        if i == 0:
            group = f"BE{i}"
        else:
            group = f"TT{i-1}"
        all_units = unit_info.available_units
        nunit = unit_info.data
        for unit_no in all_units:
            nunit.set_unit_no(unit_no)
            timestamps = nunit.get_unit_stamp()
            mean_wave_res = nunit.wave_property()
            mean_wave = mean_wave_res["Mean wave"][:, mean_wave_res["Max channel"]]
            sd_wave = mean_wave_res["Std wave"][:, mean_wave_res["Max channel"]]
            nwbfile.add_unit(
                spike_times=timestamps,
                tname=f"TT{unit_info.tag}_U{unit_no}",
                waveform_mean=mean_wave,
                waveform_sd=sd_wave,
                electrode_group=nwbfile.get_electrode_group(group),
            )

    position_data = np.transpose(
        np.array(
            [
                recording.data["spatial"].position[0].value,
                recording.data["spatial"].position[1].value,
            ]
        )
    )
    position_timestamps = recording.data["spatial"].time
    time_rate = np.mean(np.diff(position_timestamps))

    spatial_series = SpatialSeries(
        name="PositionSeries",
        description="(x,y) position in open field",
        data=position_data,
        starting_time=0.,
        rate=time_rate,
        reference_frame="(0,0) is top left corner",
        unit="centimeters",
    )
    position_obj = Position(spatial_series=spatial_series)

    hd_series = SpatialSeries(
        name="HDSeries",
        description="head direction",
        data=recording.data["spatial"].direction,
        starting_time=0.,
        rate=time_rate,
        reference_frame="0 degrees is west, rotation is anti-clockwise",
        unit="degrees",
    )
    compass_obj = CompassDirection(spatial_series=hd_series)

    speed_ts = TimeSeries(
        name="running_speed",
        description="Running speed in openfield",
        data=recording.data["spatial"].speed.value,
        starting_time=0.,
        rate=time_rate,
        unit="cm/s",
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavior data"
    )
    behavior_module.add(position_obj)
    behavior_module.add(compass_obj)
    behavior_module.add(speed_ts)

    return nwbfile


if __name__ == "__main__":
    main(
        "results/subret_recordings.csv",
        "config/simuran_params.yaml",
        "config/openfield_recordings.yaml",
        Path("results/"),
    )

    # main(
    #     snakemake.input[0],
    #     snakemake.config["simuran_config"],
    #     snakemake.config["openfield_filter"],
    #     Path(snakemake.output[0]).parent,
    #     snakemake.threads,
    #     snakemake.config["data_directory"],
    # )
