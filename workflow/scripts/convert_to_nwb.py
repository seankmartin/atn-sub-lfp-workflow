"""Process openfield LFP into power spectra etc. saved to NWB"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import simuran as smr
from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
from neurochat.nc_lfp import NLfp
from neurochat.nc_spike import NSpike
from neurochat.nc_utils import RecPos
from pynwb import NWBHDF5IO, NWBFile, TimeSeries
from pynwb.behavior import CompassDirection, Position, SpatialSeries
from pynwb.ecephys import LFP, ElectricalSeries
from pynwb.file import Subject
from skm_pyutils.table import (df_from_file, df_to_file, filter_table,
                               list_to_df)


def describe_columns():
    return [
        {
            "name": "tetrode_chan_id",
            "type": str,
            "doc": "label of tetrode_label of channel",
        },
        {
            "name": "num_spikes",
            "type": int,
            "doc": "the number of spikes identified",
        },
        {
            "name": "timestamps",
            "type": np.ndarray,
            "doc": "identified spike times in seconds",
        },
    ]


def describe_columns_waves():
    return [
        {
            "name": "tetrode_chan_id",
            "type": str,
            "doc": "label of tetrode_label of channel",
        },
        {
            "name": "num_spikes",
            "type": int,
            "doc": "the number of spikes identified",
        },
        {
            "name": "waveforms",
            "type": np.ndarray,
            "doc": "Flattened sample values around the spike time on each of the four channels, unflattened has shape (X, 50)",
        },
    ]


pd.options.mode.chained_assignment = None  # default='warn'

here = Path(__file__).resolve().parent
module_logger = logging.getLogger("simuran.custom.convert_to_nwb")


def main(
    table,
    config,
    filter_,
    output_directory,
    out_name,
    overwrite=False,
    except_errors=False,
):
    filtered_table = filter_table(table, filter_) if filter_ is not None else table
    loader = smr.loader(config["loader"], **config["loader_kwargs"])
    rc = smr.RecordingContainer.from_table(filtered_table, loader)
    used = []
    filenames = []

    for i in range(len(rc)):
        if str(rc[i].attrs["mapping"].source_file) == "no_mapping":
            module_logger.warning(
                f"Provide a mapping in index_axona_files.py"
                f" before converting {rc[i].source_file}"
            )
            continue
        fname, e = convert_to_nwb_and_save(
            rc, i, output_directory, config["cfg_base_dir"], overwrite
        )
        if fname is not None:
            filenames.append(fname)
            used.append(i)
        elif not except_errors:
            print(f"Error with recording {rc[i].source_file}")
            raise e
        else:
            print(f"Error with recording {rc[i].source_file}, check logs")

    if len(used) != len(filtered_table):
        missed = len(filtered_table) - len(used)
        print(f"WARNING: unable to convert all files, missed {missed}")
    filtered_table = filtered_table.iloc[used, :]
    filtered_table["nwb_file"] = filenames
    df_to_file(filtered_table, output_directory / out_name)
    return filenames


def convert_to_nwb_and_save(rc, i, output_directory, rel_dir=None, overwrite=False):
    save_name = rc[i].get_name_for_save(rel_dir)
    filename = output_directory / "nwbfiles" / f"{save_name}.nwb"

    if not overwrite and filename.is_file():
        module_logger.debug(f"Already converted {rc[i].source_file}")
        return filename, None

    module_logger.info(f"Converting {rc[i].source_file} to NWB")
    try:
        r = rc.load(i)
    except Exception as e:
        module_logger.error(f"Could not load {rc[i].source_file} due to {e}")
        return None, e
    nwbfile = convert_recording_to_nwb(r, rel_dir)
    return write_nwbfile(filename, r, nwbfile)


def write_nwbfile(filename, r, nwbfile, manager=None):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w", manager=manager) as io:
            io.write(nwbfile)
        return filename, None
    except Exception as e:
        module_logger.error(
            f"Could not write nwbfile from {r.source_file} out to {filename}"
        )
        if filename.is_file():
            filename.unlink()
        return None, e


def export_nwbfile(filename, r, nwbfile, src_io, debug=False):
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with NWBHDF5IO(filename, "w") as io:
            io.export(src_io=src_io, nwbfile=nwbfile)
        return filename, None
    except Exception as e:
        module_logger.error(
            f"Could not write nwbfile from {r.source_file} out to {filename}"
        )
        if debug:
            breakpoint()
        if filename.is_file():
            filename.unlink()
        return None, e


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
    filename = os.path.join(recording.attrs["directory"], recording.attrs["filename"])
    rec_pos = RecPos(filename, load=True)
    position_data = np.transpose(
        np.array(
            [
                recording.data["spatial"].position[0],
                recording.data["spatial"].position[1],
            ]
        )
    )
    position_timestamps = recording.data["spatial"].timestamps
    spatial_series = SpatialSeries(
        name="SpatialSeries",
        description="(x,y) position in camera",
        data=position_data,
        timestamps=position_timestamps,
        reference_frame="(0,0) is top left corner",
        unit="centimeters",
    )
    position_obj = Position(spatial_series=spatial_series)
    recording.data["spatial"].direction
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
        description="Smoothed running speed calculated from position",
        data=recording.data["spatial"].speed,
        timestamps=position_timestamps,
        unit="cm/s",
    )

    raw_pos = rec_pos.get_raw_pos()
    behavior_module = nwbfile.create_processing_module(
        name="behavior", description="processed behavior data"
    )
    pos = np.transpose(np.array(raw_pos, dtype=np.uint16))
    big_led_ts = TimeSeries(
        name="led_pixel_positions",
        description="LED positions, note 1023 indicates untracked data. Order is Big LED x, Big LED y, Small LED x, Small LED y",
        data=pos,
        rate=50.0,
        unit="centimeters",
        conversion=(1 / rec_pos.pixels_per_cm),
    )
    behavior_module.add(big_led_ts)

    behavior_module.add(position_obj)
    behavior_module.add(speed_ts)

    if filename.endswith(".pos"):
        behavior_module.add(compass_obj)


def add_unit_data_to_nwb(recording, nwbfile):
    add_waveforms_and_times_to_nwb(recording, nwbfile)
    if recording.attrs.get("units", "default") == "default":
        module_logger.info(f"{recording.source_file} has no unit information")
        return
    added = False
    for i, unit_info in enumerate(recording.data["units"]):
        if unit_info.available_units is None:
            continue
        if not added:
            nwbfile.add_unit_column(name="tname", description="Tetrode and unit number")
            added = True
        electrodes = nwbfile.electrodes.to_dataframe()
        if electrodes.iloc[0]["group_name"].startswith("BE"):
            group = f"BE{i}" if i == 0 else f"TT{i-1}"
        else:
            group = f"TT{i}"
        all_units = unit_info.available_units
        nunit = unit_info.data
        for unit_no in all_units:
            nunit.set_unit_no(unit_no)
            timestamps = nunit.get_unit_stamp()
            tname = f"TT{unit_info.tag}_U{unit_no}"
            if len(timestamps) < 3:
                module_logger.warning("Low firing rate spike {tname} excluded")
                continue
            mean_wave_res = nunit.wave_property()
            mean_wave = mean_wave_res["Mean wave"][:, mean_wave_res["Max channel"]]
            sd_wave = mean_wave_res["Std wave"][:, mean_wave_res["Max channel"]]
            nwbfile.add_unit(
                spike_times=timestamps,
                tname=tname,
                waveform_mean=mean_wave,
                waveform_sd=sd_wave,
                electrode_group=nwbfile.get_electrode_group(group),
            )


def add_waveforms_and_times_to_nwb(recording, nwbfile):
    try:
        spike_files = recording.attrs["source_files"]["Spike"]
    except KeyError:
        module_logger.warning(f"No spike files for {recording.source_file}")
        return
    nc_spike = NSpike()
    df_list = []
    df_list_waves = []
    for sf in spike_files:
        if not os.path.exists(sf):
            continue
        times, waves = nc_spike.load_spike_Axona(sf, return_raw=True)
        ext = os.path.splitext(sf)[-1][1:]
        for chan, val in waves.items():
            name = f"{ext}_{chan}"
            num_spikes = len(times)
            df_list.append([name, num_spikes, np.array(times)])
            df_list_waves.append([name, num_spikes, val.flatten()])
    max_spikes = max(d[1] for d in df_list)
    for df_ in df_list:
        df_[2] = np.pad(df_[2], (0, max_spikes - df_[1]), mode="empty")
    for df_wave in df_list_waves:
        df_wave[2] = np.pad(df_wave[2], (0, (max_spikes * 50) - (df_wave[1] * 50)))

    final_df = list_to_df(df_list, ["tetrode_chan_id", "num_spikes", "timestamps"])
    hdmf_table = DynamicTable.from_dataframe(
        df=final_df, name="times", columns=describe_columns()
    )
    mod = nwbfile.create_processing_module("spikes", "Store unsorted spike times")
    mod.add(hdmf_table)

    final_df = list_to_df(df_list_waves, ["tetrode_chan_id", "num_spikes", "waveforms"])
    hdmf_table = DynamicTable.from_dataframe(
        df=final_df, name="waveforms", columns=describe_columns_waves()
    )
    mod.add(hdmf_table)


def convert_eeg_path_to_egf(p):
    p = Path(p)
    p = p.with_suffix(f".egf{p.suffix[4:]}")
    if p.is_file():
        return p
    else:
        None


def add_lfp_data_to_nwb(recording, nwbfile, num_electrodes):
    egf_files = [
        convert_eeg_path_to_egf(f) for f in recording.attrs["source_files"]["Signal"]
    ]
    if egf_files := [f for f in egf_files if f is not None]:
        data = []
        for f in egf_files:
            lfp = NLfp()
            lfp.load(f, system="Axona")
            data.append(lfp.get_samples())
        rate = float(lfp.get_sampling_rate())
        lfp_data = np.transpose(np.array(data))
        module = nwbfile.create_processing_module(
            name="high_rate_ecephys",
            description="High sampling rate extracellular electrophysiology data",
        )
        add_lfp_array_to_nwb(
            nwbfile, num_electrodes, lfp_data, rate=rate, module=module
        )
    else:
        module_logger.warning(f"No egf files found for {recording.source_file}")
    lfp_data = np.transpose(np.array([s.samples for s in recording.data["signals"]]))
    add_lfp_array_to_nwb(nwbfile, num_electrodes, lfp_data, rate=250.0)


def add_lfp_array_to_nwb(
    nwbfile,
    num_electrodes,
    lfp_data,
    module=None,
    conversion=0.001,
    rate=250.0,
):
    all_table_region = nwbfile.create_electrode_table_region(
        region=list(range(num_electrodes)), description="all electrodes"
    )

    compressed_data = H5DataIO(data=lfp_data, compression="gzip", compression_opts=4)
    lfp_electrical_series = ElectricalSeries(
        name="ElectricalSeries",
        data=compressed_data,
        electrodes=all_table_region,
        starting_time=0.0,
        rate=rate,
        conversion=conversion,
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
    if recording.attrs["rat"].startswith("CanCsCa"):
        num_electrodes = add_tetrodes_no_bipolar(recording, nwbfile, piw_device)
    else:
        num_electrodes = add_bipolar_electrodes(recording, nwbfile, be_device)
        add_tetrodes_for_bipolar(recording, nwbfile, piw_device)

    return num_electrodes


def add_tetrodes_no_bipolar(recording, nwbfile, piw_device):
    if (
        len(recording.data["signals"]) == 32
        or recording.attrs.get("units", "d") is not None
    ):
        for i in range(8):
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
        return 32
    return 0


def add_tetrodes_for_bipolar(recording, nwbfile, piw_device):
    if (
        len(recording.data["signals"]) == 32
        or recording.attrs.get("units", "d") is not None
    ):
        for i in range(7):
            brain_region = get_brain_region_for_tetrode_bipolar(recording, i)
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


def add_bipolar_electrodes(recording, nwbfile, be_device):
    if len(recording.data["signals"]) == 1:
        brain_region = recording.data["signals"][0].region
        if brain_region is None:
            regions = [s.region for s in recording.data["signals"]]
            raise ValueError(
                f"Brain region was none for single chan, regions available are {regions} in {recording.source_file} with mapping {recording.attrs['mapping']}"
            )
        electrode_group = nwbfile.create_electrode_group(
            name="BE0",
            device=be_device,
            location=brain_region,
            description=f"Bipolar electrode 0 placed in {brain_region}",
        )
        add_nwb_electrode(nwbfile, brain_region, electrode_group, "BE0_E0")
        return 1
    else:
        for i in range(2):
            brain_region = recording.data["signals"][i * 2].region
            if brain_region is None:
                regions = [s.region for s in recording.data["signals"]]
                raise ValueError(
                    f"Brain region was none for four chan, regions available are {regions} in {recording.source_file} with mapping {recording.attrs['mapping']}"
                )
            electrode_group = nwbfile.create_electrode_group(
                name=f"BE{i}",
                device=be_device,
                location=brain_region,
                description=f"Bipolar electrodes {i} placed in {brain_region}",
            )
            for j in range(2):
                add_nwb_electrode(nwbfile, brain_region, electrode_group, f"BE{i}_E{j}")
        return 32 if len(recording.data["signals"]) == 32 else 4


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
        recording.data["signals"][i * 4], "region"
    ):
        brain_region = recording.data["signals"][i * 4].region
    if (
        (len(recording.data["units"]) == 8)
        and (brain_region is None)
        and hasattr(recording.data["units"][i], "region")
    ):
        brain_region = recording.data["units"][i].region
    if brain_region is None:
        brain_region = "recording_setup_error"
        print(
            f"Electrode group {i+1} has unknown "
            f"brain region for recording {recording.source_file}"
        )

    return brain_region


def get_brain_region_for_tetrode_bipolar(recording, i):
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
        session_description=f"Recording {name}",
        identifier=f"ATNx_SUB_LFP--{name}",
        session_start_time=recording.datetime,
        experiment_description="Relationship between ATN, SUB, RSC, and CA1",
        experimenter="Bethany Frost",
        lab="O'Mara lab",
        institution="TCD",
        related_publications="doi:10.1523/JNEUROSCI.2868-20.2021",
    )
    nwbfile.subject = Subject(
        species="Lister Hooded rat",
        sex="M",
        subject_id=recording.attrs["rat"],
        weight=0.330,
    )

    return nwbfile


def convert_listed_data_to_nwb(
    overall_datatable,
    config_path,
    data_fpath,
    output_directory,
    individual_tables,
    overwrite=False,
    except_errors=False,
):
    """These are processed in order of individual_tables"""
    config = smr.ParamHandler(source_file=config_path, name="params")
    table = df_from_file(overall_datatable)
    for id_table_name in individual_tables:
        id_table = df_from_file(id_table_name)
        out_name = f"{Path(id_table_name).stem}_nwb.csv"
        filter_ = {"filename": id_table["filename"]}
        filtered_table = filter_table(table, filter_)
        merged_df = filtered_table.merge(
            id_table,
            how="left",
            on="filename",
            validate="one_to_one",
            suffixes=(None, "_x"),
        )
        if "directory_x" in merged_df.columns:
            merged_df.drop("directory_x", axis=1, inplace=True)
        main(merged_df, config, None, output_directory, out_name, overwrite=overwrite, except_errors=except_errors)

    filter_ = smr.ParamHandler(source_file=data_fpath, name="filter")
    out_name = "openfield_nwb.csv"
    main(table, config, filter_, output_directory, out_name, overwrite=overwrite)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    module_logger.setLevel(logging.DEBUG)
    convert_listed_data_to_nwb(
        snakemake.input[0],
        snakemake.config["simuran_config"],
        snakemake.config["openfield_filter"],
        Path(snakemake.output[0]).parent,
        snakemake.input[1:],
        snakemake.config["overwrite_nwb"],
        except_errors=snakemake.config["except_nwb_errors"],
    )
