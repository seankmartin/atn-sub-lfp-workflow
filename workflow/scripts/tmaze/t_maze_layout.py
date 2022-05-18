import os

here = os.path.dirname(os.path.abspath(__file__))

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
    td["LSR1_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet1\recording\+maze\04092017_first trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "LSubRet1",
        "date": "4092017",
        "session_no": 1,
    }

    td["LSR1_t9"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet1\recording\+maze\15092017_9th",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "LSubRet1",
        "date": "15092017",
        "session_no": 9,
    }

    td["LSR1_t2"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet1\recording\+maze\05092017_2nd trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "LSubRet1",
        "date": "5092017",
        "session_no": 2,
    }

    td["LSR3_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet3\recording\+maze\04092017_1st trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "LSubRet3",
        "date": "4092017",
        "session_no": 1,
    }

    td["LSR3_t9"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet3\recording\+maze\15092017_9th",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "LSubRet3",
        "date": "15092017",
        "session_no": 9,
    }

    td["LSR4_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet4\recording\+maze\29112017_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet4",
        "date": "29112017",
        "session_no": 1,
    }

    td["LSR5_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\29112017_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "29112017",
        "session_no": 1,
    }

    td["LSR5_t3"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\01122017_t3",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "1122017",
        "session_no": 3,
    }

    td["LSR5_t4"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\04122017_t4",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "4122017",
        "session_no": 4,
    }

    td["LSR5_t5"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\05122017_t5",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "5122017",
        "session_no": 5,
    }

    td["LSR5_t6"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\06122017_t6",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "6122017",
        "session_no": 6,
    }

    td["LSR5_t7"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\07122017_t7",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "7122017",
        "session_no": 7,
    }

    td["LSR5_t8"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\LSubRet5\recording\plus maze\08122017_t8",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "LSubRet5",
        "date": "8122017",
        "session_no": 8,
    }

    # Control
    td["CSR1_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet1\CSubRet1_recording\+maze\04092017_first trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet1",
        "date": "4092017",
        "session_no": 1,
    }

    td["CSR1_t9"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet1\CSubRet1_recording\+maze\15092017_9th",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet1",
        "date": "15092017",
        "session_no": 9,
    }

    td["CSR2_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet2_sham\CSubRet2_recording\+maze\04092017_first trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet2",
        "date": "4092017",
        "session_no": 1,
    }

    td["CSR2_t9"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet2_sham\CSubRet2_recording\+maze\04092017_first trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet2",
        "date": "15092017",
        "session_no": 9,
    }

    td["CSR3_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet3_sham\recording\+maze\04092017_first trial",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet3",
        "date": "4092017",
        "session_no": 1,
    }

    td["CSR3_t3"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet3_sham\recording\+maze\06092017_3rd",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_1-3-no-cells.py"
        ),
        "animal": "CSubRet3",
        "date": "6092017",
        "session_no": 3,
    }

    td["CSR4_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet4\recording\+maze\29112017_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet4",
        "date": "29112017",
        "session_no": 1,
    }

    td["CSR5_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSubRet5_sham\recording\+ maze\29112017_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet5",
        "date": "29112017",
        "session_no": 1,
    }

    td["CSR6_t1"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSR6\+ maze\22032018_t1",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet6",
        "date": "22032018",
        "session_no": 1,
    }

    td["CSR6_t6"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSR6\+ maze\09042018_t6",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet6",
        "date": "9042018",
        "session_no": 6,
    }

    td["CSR6_t7"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSR6\+ maze\09042018_t7",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet6",
        "date": "9042018",
        "session_no": 7,
    }

    td["CSR6_t8"] = {
        "t_maze_dir": r"D:\SubRet_recordings_imaging\CSR6\+ maze\10042018_t8",
        "base_dir": r"D:\SubRet_recordings_imaging",
        "mapping_location": os.path.join(
            here, "..", "recording_mappings", "CL-SR_4-6-no-cells.py"
        ),
        "animal": "CSubRet6",
        "date": "10042018",
        "session_no": 8,
    }

    return td