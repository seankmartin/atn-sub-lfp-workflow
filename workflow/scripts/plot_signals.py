from pathlib import Path

import matplotlib.pyplot as plt
import simuran as smr
from simuran.bridges.mne_bridge import plot_signals
from skm_pyutils.table import df_from_file


def plot_all_signals(recording, output_path):
    eeg_array = []
    nwbfile = recording.data
    electrodes_table = nwbfile.electrodes.to_dataframe()
    bad_chans = electrodes_table[electrodes_table["clean"] == "Outlier"].index
    locations = electrodes_table["location"]
    ch_names = [f"{locations[i]}_{i}" for i in range(len(locations))]
    bad_chans = [ch_names[i] for i in bad_chans]
    sr = nwbfile.processing["ecephys"]["LFP"]["ElectricalSeries"].rate
    lfp_data = nwbfile.processing["normalised_lfp"]["LFP"]["ElectricalSeries"].data[:].T
    for sig in lfp_data:
        eeg = smr.BaseSignal.from_numpy(sig, sr)
        eeg.conversion = 0.0000001
        eeg_array.append(eeg)
    average_signal = nwbfile.processing["average_lfp"]
    names = []
    for k in average_signal.data_interfaces:
        sig = average_signal[k].data[:]
        eeg = smr.Eeg.from_numpy(sig, sr)
        eeg.conversion = 0.0000001
        eeg_array.append(eeg)
        names.append(k)

    ch_names.extend(names)
    fig = plot_signals(
        eeg_array,
        ch_names=ch_names,
        bad_chans=bad_chans,
        title=recording.get_name_for_save(),
        show=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def plot_signals_rc(recording_container, out_dir):
    for recording in recording_container.load_iter():
        output_path = out_dir / f"{recording.get_name_for_save()}--lfp.png"
        plot_all_signals(recording, output_path)


def main(input_df_path, out_dir, config_path):
    config = smr.ParamHandler(source_file=config_path)
    datatable = df_from_file(input_df_path)
    loader = smr.loader("nwb")
    rc = smr.RecordingContainer.from_table(datatable, loader=loader)
    plot_signals_rc(rc, out_dir)


if __name__ == "__main__":
    smr.set_only_log_to_file(snakemake.log[0])
    main(
        snakemake.input[0],
        Path(snakemake.output[0]),
        snakemake.config["simuran_config"],
    )
