"""
simuran_fn_params controls the functions that will be performed.

These functions are performed on each recording in a loaded container.
"""


def setup_functions():
    """Establish the functions to run and arguments to pass."""
    # The list of functions to run, in order
    # Each function should take as its first argument a recording object
    # This should be an actual function, as opposed to a string name

    functions = []

    def argument_handler(recording_container, idx, figures):
        """
        Set up what arguments should be passed to the functions.

        Given recording_container, idx, and figures
        this should return all arguments for this run.

        This can be used to run the same function many times
        with different parameters, by providing an argument. E.g.
        def add(recording, num1, num2):
            return num1 + num2
        functions = ["add"]
        arguments["add"] = {"0": 1, 2, "1": 2, 3}
        would add 1 and 2, and then separately add 2 and 3

        Parameters
        ----------
        recording_container : simuran.recordingcontainer.RecordingContainer
            The recording container object that is in use
        idx : int
            The index of the current recording_container item in use
        figures : list of matplotlib figure or simuran SimuranFigure objects
            The list of figures in use.
            This can be used to plot into axes of specific figures,
            or can be appended to in order to store figures.

        Returns
        -------
        dict
            The arguments to use for each function in functions

        """
        return {}

    return functions, argument_handler


def setup_figures():
    """
    Establish the figures that will be used.

    This is mostly used if you need to make a big
    grid plot and plot into specific axes.

    See also
    --------
    skm_pyutils.py_plot.GridFig

    """
    # The list of figures to use
    figs = []

    # The name for each figure provided
    fig_names = []

    return figs, fig_names


def setup_output():
    """Establish what results of the functions will be saved."""
    # This should list the results to save to a csv
    save_list = []

    # You can name each of these outputs
    output_names = []

    return save_list, output_names


def setup_sorting():
    """If you don't need to do sorting, simply return None."""

    def sort_fn(x):
        """
        Establish a sorting function for recordings in a container.

        Note
        ----
        "__dirname__" is a magic string that can be used to obtain
        the absolute path to the directory this file is in
        so you don't have to hard code it.

        Returns
        -------
        object
            any object with a defined ordering function.
            for example, an integer

        """
        in_dir = "__dirname__"
        comp = x.source_file[len(in_dir) + 1 :]
        return comp

    # Use return None to do no sorting
    # return None
    return None


def setup_loading():
    """Establish how recordings are loaded."""
    # Indicates if all information on a recording is loaded in bulk, or as needed
    load_all = False

    # If load_all is True, indicates what is loaded in bulk
    # Should be a subset of ["signals", "spatial", "units"]
    to_load = []

    # Whether a subset of recordings should be considered
    # True opens a console to help choose, but a list of indices can be passed
    select_recordings = False

    return load_all, to_load, select_recordings


functions, args_func = setup_functions()
save_list, output_names = setup_output()
figs, fig_names = setup_figures()
sort_fn = setup_sorting()
load_all, to_load, select_recordings = setup_loading()
fn_params = {
    "run": functions,
    "args": args_func,
    "save": save_list,
    "names": output_names,
    "figs": figs,
    "fignames": fig_names,
    "sorting": sort_fn,
    "load_all": load_all,
    "to_load": to_load,
    "select_recordings": select_recordings,
}
