from pathlib import Path
from copy import copy

here = Path(__file__).parent
here = here.parent / "scripts" / "recording_mappings"


def t_maze_dict():

    td = {}

    # The key is used to select the t-maze to analyse
    # (e.g LSR1_t1 would select that session)
    # An entry to analyse needs these keys
    # 1. t_maze_dir : the directory to the trial
    # 2. base_dir : base dir of beths data
    # 3. mapping_location : parameter file of wires to region
    # -- The next 3 are used to lookup results/maze.xlsx
    # 4. animal : name of animal (note try to name as in this sheet - )
    # 5. date : date from sheet
    # 6. session_no : the session number

    # Lesion
    base_lrs1 = {
        "base_dir": r"H:\SubRet_recordings_imaging",
        "mapping_location": here / "CL-RS.py",
        "animal": "LRetSub1",
    }

    def add_new_lrs1(td, date, session):
        new_dict = copy(base_lrs1)
        dict_date = date[1:] if date.startswith("0") else date
        new_dict["date"] = dict_date
        new_dict["session_no"] = session
        new_dict["t_maze_dir"] = (
            r"H:\SubRet_recordings_imaging\LRS1\t_maze" + f"\{date}_t{session}"
        )
        td[f"LRS1_t{session}"] = new_dict

    add_new_lrs1(td, "22032018", 1)
    add_new_lrs1(td, "23032018", 2)
    add_new_lrs1(td, "27032018", 3)
    add_new_lrs1(td, "28032018", 4)
    add_new_lrs1(td, "29032018", 5)
    add_new_lrs1(td, "04042018", 6)
    add_new_lrs1(td, "09042018", 7)
    add_new_lrs1(td, "10042018", 8)

    base_crs1 = {
        "base_dir": r"H:\SubRet_recordings_imaging",
        "mapping_location": here / "CL-RS.py",
        "animal": "CRetSub1",
    }

    def add_new_crs1(td, date, session):
        new_dict = copy(base_crs1)
        dict_date = date[1:] if date.startswith("0") else date
        new_dict["date"] = dict_date
        new_dict["session_no"] = session
        new_dict["t_maze_dir"] = (
            r"H:\SubRet_recordings_imaging\CRS1\+ maze" + f"\{date}_t{session}"
        )
        td[f"CRS1_t{session}"] = new_dict

    add_new_crs1(td, "22032018", 1)
    add_new_crs1(td, "23032018", 2)
    add_new_crs1(td, "27032018", 3)
    add_new_crs1(td, "28032018", 4)
    add_new_crs1(td, "29032018", 5)
    add_new_crs1(td, "04042018", 6)
    add_new_crs1(td, "09042018", 7)
    add_new_crs1(td, "10042018", 8)

    base_crs2 = {
        "base_dir": r"H:\SubRet_recordings_imaging",
        "mapping_location": here / "CL-RS.py",
        "animal": "CRetSub2",
    }

    def add_new_crs2(td, date, session):
        new_dict = copy(base_crs2)
        dict_date = date[1:] if date.startswith("0") else date
        new_dict["date"] = dict_date
        new_dict["session_no"] = session
        new_dict["t_maze_dir"] = (
            r"H:\SubRet_recordings_imaging\CRS2\+ maze" + f"\{date}_t{session}"
        )
        td[f"CRS2_t{session}"] = new_dict

    add_new_crs2(td, "22032018", 1)
    add_new_crs2(td, "23032018", 2)
    add_new_crs2(td, "27032018", 3)
    add_new_crs2(td, "28032018", 4)
    add_new_crs2(td, "29032018", 5)

    return td


if __name__ == "__main__":
    print(t_maze_dict())
