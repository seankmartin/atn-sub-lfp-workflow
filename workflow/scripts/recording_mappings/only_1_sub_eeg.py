"""
The simuran_base_params is used to initialize the
the simuran_params file in each recording data session
that is a child directory of starting directory.

This should contain information about the recording,
such as the number electrodes in the recording.
"""


def setup_signals():
    """Set up the signals (such as eeg or lfp)."""

    # The total number of signals in the recording
    num_signals = 1

    # What brain region each signal was recorded from
    regions = ["SUB"]

    # If the wires were bundled, or any other kind of grouping existed
    # If no grouping, grouping = [i for in range(num_signals)]
    groups = ["LFP"]

    # The sampling rate in Hz of each signal
    sampling_rate = [250]

    return {
        "num_signals": num_signals,
        "region": regions,
        "group": groups,
        "sampling_rate": sampling_rate,
    }


def setup_units():
    """Set up the single unit data."""
    # The number of tetrodes, probes, etc - any kind of grouping
    num_groups = 7

    # The region that each group belongs to
    regions = ["SUB"] * num_groups

    # A group number for each group, for example the tetrode number
    groups = [i + 2 for i in range(num_groups)]

    return {
        "num_groups": num_groups,
        "region": regions,
        "group": groups,
    }


def setup_spatial():
    """Set up the spatial data."""
    arena_size = "default"

    return {
        "arena_size": arena_size,
    }


def setup_loader():
    """
    Set up the loader and keyword arguments for the loader.

    See also
    --------
    simuran.loaders.loader_list.py

    """
    # The type of loader to use, see simuran.loaders.loader_list.py for options
    loader = "nc_loader"

    # Keyword arguments to pass to the loader.
    loader_kwargs = {
        "system": "Axona",
        "pos_extension": ".pos",
    }

    return {
        "loader": loader,
        "loader_kwargs": loader_kwargs,
    }


load_params = setup_loader()
mapping = {
    "signals": setup_signals(),
    "units": setup_units(),
    "spatial": setup_spatial(),
    "loader": load_params["loader"],
    "loader_kwargs": load_params["loader_kwargs"],
}
