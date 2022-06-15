"""Process openfield LFP into power spectra etc. saved to NWB"""

import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from skm_pyutils.table import df_from_file, df_to_file, filter_table

pd.options.mode.chained_assignment = None  # default='warn'

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.convert_to_nwb")


def main(table, config, filter_, output_directory, out_name, overwrite=False):
    filtered_table = filter_table(table, filter_) if filter_ is not None else table
    loader = smr.loader(config["loader"], **config["loader_kwargs"])
    rc = smr.RecordingContainer.from_table(filtered_table, loader)
    filenames = []

    for i in range(len(rc)):
        module_logger.info(f"Converting {rc[i].source_file} to NWB")
        fname = convert_to_nwb_and_save(
            rc, i, output_directory, config["cfg_base_dir"], overwrite
        )
        filenames.append(fname)

    filtered_table["nwb_file"] = filenames
    df_to_file(filtered_table, output_directory / out_name)


def convert_to_nwb_and_save(rc, i, output_directory, rel_dir=None, overwrite=False):
    save_name = rc[i].get_name_for_save(rel_dir)
    filename = output_directory / "nwbfiles" / f"{save_name}.nwb"

    if not overwrite and filename.is_file():
        return filename

    r = rc.load(i)
    nwbfile = convert_recording_to_nwb(r, rel_dir)
    return write_nwbfile(filename, r, nwbfile)


def write_nwbfile(filename, r, nwbfile, manager=None):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w", manager=manager) as io:
            io.write(nwbfile)
        return filename
    except Exception:
        module_logger.error(f"Could not write {nwbfile} from {r} out to {filename}")
        if filename.is_file():
            filename.unlink()
        traceback.print_exc()
        return None


def export_nwbfile(filename, r, nwbfile, src_io):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w") as io:
            io.export(src_io=src_io, nwbfile=nwbfile)
        return filename
    except Exception:
        module_logger.error(f"Could not write {nwbfile} from {r} out to {filename}")
        if filename.is_file():
            filename.unlink()
        traceback.print_exc()
        return None


def access_nwb(nwbfile):
    # lfp_data = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].data[:]
    # unit_data = nwbfile.units
    # behavior = nwbfile.processing["behavior"]
    # position = behavior["Position"]["SpatialSeries"].data[:]
    # head_direction = behavior["CompassDirection"]["SpatialSeries"].data[:]
    # running_speed = behavior["running_speed"].data[:]
    pass


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
    spatial_series = SpatialSeries(
        name="SpatialSeries",
        description="(x,y) position in open field",
        data=position_data,
        timestamps=position_timestamps,
        reference_frame="(0,0) is top left corner",
        unit="centimeters",
    )
    position_obj = Position(spatial_series=spatial_series)

    hd_series = SpatialSeries(
        name="SpatialSeries",
        description="head direction",
        data=recording.data["spatial"].direction,
        timestamps=position_timestamps,
        reference_frame="0 degrees is west, rotation is anti-clockwise",
        unit="degrees",
    )
    compass_obj = CompassDirection(spatial_series=hd_series)

    speed_ts = TimeSeries(
        name="running_speed",
        description="Running speed in openfield",
        data=recording.data["spatial"].speed.value,
        timestamps=position_timestamps,
        unit="cm/s",
    )

    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavior data"
    )
    behavior_module.add(position_obj)
    behavior_module.add(compass_obj)
    behavior_module.add(speed_ts)


def add_unit_data_to_nwb(recording, nwbfile):
    if recording.attrs.get("units", "default") is None:
        return
    added = False
    for i, unit_info in enumerate(recording.data["units"]):
        if unit_info.available_units is None:
            continue
        if not added:
            nwbfile.add_unit_column(name="tname", description="Tetrode and unit number")
            added = True
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
    lfp_data = np.transpose(
        np.array([s.samples.value for s in recording.data["signals"]])
    )
    add_lfp_array_to_nwb(nwbfile, num_electrodes, lfp_data)


def add_lfp_array_to_nwb(nwbfile, num_electrodes, lfp_data, module=None):
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(num_electrodes)), description="all electrodes"
    )

    compressed_data = H5DataIO(data=lfp_data, compression="gzip", compression_opts=4)
    lfp_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=compressed_data,
        electrodes=all_table_region,
        starting_time=0.0,
        rate=250.0,
        conversion=0.001,
        filtering="Notch filter at 50Hz",
    )
    lfp = LFP(electrical_series=lfp_electrical_series)

    if module is None:
        module = nwbfile.create_processing_module(
            name="ecephys", description="Processed extracellular electrophysiology data"
        )
    module.add(lfp)


def add_electrodes_to_nwb(recording, nwbfile, piw_device, be_device):
    nwbfile.add_electrode_column(name="label", description="electrode label")
    if len(recording.data["signals"]) == 1:
        brain_region = recording.data["signals"][0].region
        electrode_group = nwbfile.create_electrode_group(
            name="BE0",
            device=be_device,
            location=brain_region,
            description=f"Bipolar electrode 0 placed in {brain_region}",
        )
        add_nwb_electrode(nwbfile, brain_region, electrode_group, "BE0_E0")
        num_electrodes = 1
    else:
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
    if len(recording.data["signals"]) == 32 and hasattr(
        recording.data["signals"][(i + 1) * 4], "region"
    ):
        brain_region = recording.data["signals"][(i + 1) * 4].region
    if (
        (len(recording.data["units"]) == 7)
        and (brain_region is None)
        and hasattr(recording.data["units"][i], "region")
    ):
        brain_region = recording.data["units"][i].region
    if (
        len(recording.data["units"]) == 8
        and (brain_region is None)
        and hasattr(recording.data["units"][i + 1], "region")
    ):
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


def convert_listed_data_to_nwb(
    overall_datatable,
    config_path,
    data_fpath,
    output_directory,
    individual_tables,
    overwrite=False,
):
    """These are processed in order of individual_tables"""
    table = df_from_file(overall_datatable)
    config = smr.ParamHandler(source_file=config_path, name="params")
    filter_ = smr.ParamHandler(source_file=data_fpath, name="filter")
    out_name = "openfield_nwb.csv"
    # main(table, config, filter_, output_directory, out_name, overwrite=overwrite)
    for id_table_name in individual_tables:
        id_table = df_from_file(id_table_name)
        out_name = f"{Path(id_table_name).stem}_nwb.csv"
        filter_ = {"filename": id_table["filename"].values}
        filtered_table = filter_table(table, filter_)
        filtered_table.merge(
            id_table,
            how="left",
            on="filename",
            validate="one_to_one",
            suffixes=(None, "_x"),
        )
        if "directory_x" in filtered_table.columns:
            filtered_table.drop("directory_x", inplace=True)
        main(filtered_table, config, None, output_directory, out_name, overwrite)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    convert_listed_data_to_nwb(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.config["openfield_filter"],
        Path(snakemake.output[0]).parent,
        snakemake.input[1:],
        snakemake.config["overwrite_nwb"],
    )
