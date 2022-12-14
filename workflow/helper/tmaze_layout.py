from pathlib import Path

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
    td["LRS1_t1"] = {
        "t_maze_dir": r"H:\SubRet_recordings_imaging\LRS1\t_maze\22032018_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": here / "CL-RS.py",
        "animal": "LRetSub1",
        "date": "22032018",
        "session_no": 1,
    }

    return td
