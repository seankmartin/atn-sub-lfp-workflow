"""simuran_batch_params.py describes behaviour for recursing through directories."""

import os

# Where to start running batch analysis from
start_dir = os.path.abspath(
    os.path.join("__dirname__", "muscimol_data", "CanCSR8_muscimol", "05102018")
)

# regex_filters should be a list of regex patterns to match against.
regex_filters = [
    "^s",
    "(?<!nc_plots)$",
    "(?<!nc_results)$",
    "(?<!final_plots_asSVG)$",
    "(?<!data)$",
    "(?<!_tmaze)$"
]

# Overwrites existing parameter files (simuran_params.py) if they exist.
overwrite = True

# If True, only prints where the parameter files would be written to.
only_check = False

# If True, opens an interactive console to help design your regex_filters.
interactive = False

# Please include a path to mapping_file, or setup the mapping dictionary.
# This will determine which parameters are used in proceeding analysis.
# mapping directly specifies parameters, so mapping_file is preferred
mapping = {}  # see default_params.py to set this up

# The magic string __dirname__, is replaced by the directory this file is in
directory = "__thisdirname__"
# Absolute path to a file that contains the mapping.
mapping_file = os.path.abspath(
    os.path.join(directory, "..", "recording_mappings", "CanCSR.py")
)

# The basename of the output parameter files.
out_basename = "simuran_params.py"

# Should all parameter files be deleted that match out_basename
# And exist in a child directory of the starting directory
delete_old_files = True

# NB overwrites the function selection if present
def setup_sorting():
    """If you don't need to do sorting, simply return None."""
    import os
    start_dir = os.path.abspath(
        os.path.join("__dirname__", "muscimol_data", "CanCSR8_muscimol",  "05102018")
    )

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
        comp = x.source_file[len(start_dir + os.sep) + 1 :]
        comp = int(comp.split("_")[0])
        return comp

    # Use return None to do no sorting
    # return None
    return sort_fn


params = {
    "start_dir": start_dir,
    "regex_filters": regex_filters,
    "overwrite": overwrite,
    "only_check": only_check,
    "interactive": interactive,
    "mapping": mapping,
    "mapping_file": mapping_file,
    "out_basename": out_basename,
    "delete_old_files": delete_old_files,
    "sorting": setup_sorting(),
}
