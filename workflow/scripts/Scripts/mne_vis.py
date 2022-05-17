import datetime
import os

import mne
from mne.preprocessing import ICA


def create_mne_array(recording, ch_names=None):
    """
    Populate a full mne raw array object with information.

    Parameters
    ----------
    lfp_odict : bvmpc.lfp_odict.LfpODict
        The lfp_odict object to convert to numpy data.
    ch_names : List of str, Default None
        Optional. What to name the mne eeg channels, default: region+chan_idx.

    Returns
    -------
    mne.io.RawArray

    """
    raw_data = recording.get_np_signals() / 1000

    if ch_names is None:
        try:
            ch_names = [
                "{}-{}".format(x, y)
                for x, y in zip(
                    recording.get_signals().get_property("region"),
                    recording.get_signal_channels(as_idx=True),
                )
            ]
        except BaseException:
            ch_names = [str(i) for i in range(len(recording.get_signals()))]

    # Convert LFP data into mne format
    example_lfp = recording.get_signals()[0]
    sfreq = example_lfp.get_sampling_rate()
    ch_types = ["eeg"] * len(recording.get_signals())
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(raw_data, info)

    return raw


def ICA_pipeline(
    mne_array,
    regions,
    chans_to_plot=20,
    base_name="",
    exclude=None,
    skip_plots=False,
):
    """This is example code using mne."""
    raw = mne_array
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1.0, h_freq=90)

    if not skip_plots:
        # Plot raw signal
        filt_raw.plot(
            n_channels=chans_to_plot,
            block=True,
            duration=50,
            show=True,
            clipping="transparent",
            title="Raw LFP Data from {}".format(base_name),
            remove_dc=False,
            scalings=dict(eeg=350e-6),
        )
    ica = ICA(method="fastica", random_state=42)
    ica.fit(filt_raw)

    # ica.exclude = [4, 6, 12]
    raw.load_data()
    if exclude is None:
        # Plot raw ICAs
        ica.plot_sources(filt_raw)

        # Overlay ICA cleaned signal over raw. Seperate plot for each region.
        cont = input("Plot region overlay? (y|n) \n")
        if cont.strip().lower() == "y":
            reg_grps = []
            for reg in set(regions):
                temp_grp = []
                for ch in raw.info.ch_names:
                    if reg in ch:
                        temp_grp.append(ch)
                reg_grps.append(temp_grp)
            for grps in reg_grps:
                ica.plot_overlay(
                    raw, stop=int(30 * 250), title="{}".format(grps[0][:3]), picks=grps
                )
    else:
        # ICAs to exclude
        ica.exclude = exclude
        if not skip_plots:
            ica.plot_sources(filt_raw)
    # Apply ICA exclusion
    reconst_raw = filt_raw.copy()
    exclude_raw = filt_raw.copy()
    print("ICAs excluded: ", ica.exclude)
    ica.apply(reconst_raw)

    if not skip_plots:
        # change exclude to all except chosen ICs
        all_ICs = list(range(ica.n_components_))
        for i in ica.exclude:
            all_ICs.remove(i)
        ica.exclude = all_ICs
        ica.apply(exclude_raw)

        # Plot excluded ICAs
        exclude_raw.plot(
            block=True,
            show=True,
            clipping="transparent",
            duration=50,
            title="Excluded ICs from {}".format(base_name),
            remove_dc=False,
            scalings=dict(eeg=350e-6),
        )

        # Plot reconstructed signals w/o excluded ICAs
        reconst_raw.plot(
            block=True,
            show=True,
            clipping="transparent",
            duration=50,
            title="Reconstructed LFP Data from {}".format(base_name),
            remove_dc=False,
            scalings=dict(eeg=350e-6),
        )

    return reconst_raw


def plot_mne_signal(recording, do_ica=True):
    mne_array = create_mne_array(recording)
    mne_config = recording.get("mne_config", None)
    s_date = recording.datetime

    date_found = False
    if mne_config is not None:
        for key, val in mne_config.items():
            config_date = datetime.datetime.strptime(key, "%Y-%m-%d").date()
            if config_date == s_date:
                date_found = True
                badchans = val["Bad Chs"]
                exclude = val["Bad ICs"]

    if date_found == False:
        print("No session specific mne config found.")
        badchans = []
        exclude = None

    mne_array.info["bads"] = badchans

    if do_ica:
        ica_txt = "ICA"  # Used for file naming
        recon_raw = ICA_pipeline(
            mne_array,
            recording.get_signals().get_property("region"),
            chans_to_plot=len(recording.get_signals()),
            base_name=os.path.splitext(os.path.basename(recording.source_file))[0],
            exclude=exclude,
            skip_plots=False,
        )
    else:
        ica_txt = "Raw"  # Used for file naming
        recon_raw = mne_array
