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


def main(table_path, config_path, data_fpath, output_directory: Path, overwrite=False):
    table = df_from_file(table_path)
    config = smr.ParamHandler(source_file=config_path, name="params")
    filter_ = smr.ParamHandler(source_file=data_fpath, name="filter")
    filtered_table = filter_table(table, filter_)
    loader = smr.loader(config["loader"])(**config["loader_kwargs"])

    rc = smr.RecordingContainer.from_table(filtered_table, loader)

    for i in range(len(rc)):
        convert_to_nwb_and_save(
            rc, i, output_directory, config["cfg_base_dir"], overwrite
        )


def convert_to_nwb_and_save(rc, i, output_directory, rel_dir=None, overwrite=False):
    save_name = rc[i].get_name_for_save(rel_dir)
    filename = output_directory / "nwbfiles" / f"{save_name}.nwb"

    if not overwrite and filename.is_file():
        return

    r = rc.load(i)
    nwbfile = convert_recording_to_nwb(r, rel_dir)
    filename.parent.mkdir(parents=True, exist_ok=True)

    try:
        with NWBHDF5IO(filename, "w") as io:
            io.write(nwbfile)
    except Exception:
        print(f"Could not write {nwbfile} from {r} out to {filename}")
        if filename.is_file():
            filename.unlink()


def convert_recording_to_nwb(recording, rel_dir=None):
    name = recording.get_name_for_save(rel_dir=rel_dir)
    nwbfile = create_nwbfile_with_metadata(recording, name)
    piw_device, be_device = add_devices_to_nwb(nwbfile)
    num_electrodes = add_electrodes_to_nwb(recording, nwbfile, piw_device, be_device)

    add_lfp_data_to_nwb(recording, nwbfile, num_electrodes)
    add_unit_data_to_nwb(recording, nwbfile)
    add_position_data_to_nwb(recording, nwbfile)

    return nwbfile


def add_position_data_to_nwb(recording, nwbfile):
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
        starting_time=0.0,
        rate=time_rate,
        reference_frame="(0,0) is top left corner",
        unit="centimeters",
    )
    position_obj = Position(spatial_series=spatial_series)

    hd_series = SpatialSeries(
        name="HDSeries",
        description="head direction",
        data=recording.data["spatial"].direction,
        starting_time=0.0,
        rate=time_rate,
        reference_frame="0 degrees is west, rotation is anti-clockwise",
        unit="degrees",
    )
    compass_obj = CompassDirection(spatial_series=hd_series)

    speed_ts = TimeSeries(
        name="running_speed",
        description="Running speed in openfield",
        data=recording.data["spatial"].speed.value,
        starting_time=0.0,
        rate=time_rate,
        unit="cm/s",
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavior data"
    )
    behavior_module.add(position_obj)
    behavior_module.add(compass_obj)
    behavior_module.add(speed_ts)


def add_unit_data_to_nwb(recording, nwbfile):
    nwbfile.add_unit_column(name="tname", description="Tetrode and unit number")

    if recording.attrs.get("units", "default") is not None:
        for i, unit_info in enumerate(recording.data["units"]):
            if unit_info.available_units is None:
                continue
            group = f"BE{i}" if i == 0 else f"TT{i-1}"
            all_units = unit_info.available_units
            nunit = unit_info.data
            for unit_no in all_units:
                nunit.set_unit_no(unit_no)
                timestamps = nunit.get_unit_stamp()
                if len(timestamps) < 10:
                    continue
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


def add_lfp_data_to_nwb(recording, nwbfile, num_electrodes):
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


def add_electrodes_to_nwb(recording, nwbfile, piw_device, be_device):
    nwbfile.add_electrode_column(name="label", description="electrode label")
    for i in range(2):
        brain_region = recording.data["signals"][i * 2].region
        electrode_group = nwbfile.create_electrode_group(
            name=f"BE{i}",
            device=be_device,
            location=brain_region,
            description=f"Bipolar electrodes {i} placed in {brain_region}",
        )
        for j in range(2):
            add_nwb_electrode(nwbfile, brain_region, electrode_group, f"BE{i}_E{j}")
    num_electrodes = 32 if len(recording.data["signals"]) == 32 else 4
    if (
        len(recording.data["signals"]) == 32
        or recording.attrs.get("units", "d") is not None
    ):
        for i in range(7):
            brain_region = get_brain_region_for_tetrode(recording, i)
            electrode_group = nwbfile.create_electrode_group(
                name=f"TT{i}",
                device=piw_device,
                location=brain_region,
                description=f"Tetrode {i} electrodes placed in {brain_region}",
            )
            if len(recording.data["signals"]) == 32:
                for j in range(4):
                    add_nwb_electrode(
                        nwbfile, brain_region, electrode_group, f"TT{i}_E{j}"
                    )

    return num_electrodes


def add_nwb_electrode(nwbfile, brain_region, electrode_group, label):
    nwbfile.add_electrode(
        x=np.nan,
        y=np.nan,
        z=np.nan,
        imp=np.nan,
        location=brain_region,
        filtering="Notch filter at 50Hz",
        group=electrode_group,
        label=label,
    )


def get_brain_region_for_tetrode(recording, i):
    brain_region = None
    if len(recording.data["signals"]) == 32:
        if hasattr(recording.data["signals"][(i + 1) * 4], "region"):
            brain_region = recording.data["signals"][(i + 1) * 4].region
    if (len(recording.data["units"]) == 7) and (brain_region is None):
        if hasattr(recording.data["units"][i], "region"):
            brain_region = recording.data["units"][i].region
    if len(recording.data["units"]) == 8 and (brain_region is None):
        if hasattr(recording.data["units"][i + 1], "region"):
            brain_region = recording.data["units"][i + 1].region
    if brain_region is None:
        brain_region = "recording_setup_error"
        print(
            f"Electrode group {i+1} has unknown "
            f"brain region for recording {recording.source_file}"
        )

    return brain_region


def add_devices_to_nwb(nwbfile):
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

    return piw_device, be_device


def create_nwbfile_with_metadata(recording, name):
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
