# See the __main__ for parameters
import os
import shutil
from site import addsitedir
import argparse

import simuran
import pandas as pd
from skm_pyutils.py_path import get_all_files_in_dir

lib_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
addsitedir(lib_folder)
from lib.plots import plot_pos_over_time

addsitedir(os.path.join(os.path.dirname(__file__)))
from t_maze_layout import t_maze_dict


def analyse_recording(recording):

    spatial = recording.spatial.underlying

    # NOTE can change the rate here if going too fast or too slow
    # NOTE skip_rate can be used to skip less time.
    times = plot_pos_over_time(
        spatial.get_pos_x(), spatial.get_pos_y(), rate=1.5, skip_rate=27)
    return times


def main(
    t_maze_dir,
    output_location,
    base_dir,
    mapping_file,
    xls_location=None,
    animal="",
    date="",
    session_no="",
):
    """Create a single recording for analysis."""
    # loop over the sessions in each t-maze folder
    xls_location_ = os.path.join(output_location, "maze.xlsx")
    ref_df = pd.read_excel(
        xls_location_,
        sheet_name="all",
        usecols=[0, 1, 3, 4, 9],
        dtype={
            "Rat": str,
            "Date": str,
            "SesNo": int,
            "TrialNo": int,
            "pass/fail": str,
        },
    )
    matching_rat_date = ref_df[(ref_df["Rat"] == animal) & (ref_df["Date"] == date)]

    columns = [
        "location",
        "start",
        "choice",
        "end",
        "session",
        "trial",
        "animal",
        "date",
        "test",
        "passed",
        "mapping",
    ]
    data_list = []
    df = None
    if xls_location is not None:
        if os.path.exists(xls_location):
            df = pd.read_excel(xls_location)
    if df is None:
        df = pd.DataFrame(columns=columns)

    else:
        os.makedirs(output_location, exist_ok=True)

    for folder in os.listdir(t_maze_dir):
        dir_loc = os.path.join(t_maze_dir, folder)
        set_file_locations = get_all_files_in_dir(dir_loc, ext=".set")
        if len(set_file_locations) == 0:
            raise ValueError(f"No set files were found in {dir_loc}")
        set_file_location = set_file_locations[0]
        main_file_name = set_file_location[len(base_dir + os.sep) :].replace(
            os.sep, "--"
        )
        recording = simuran.Recording(
            param_file=mapping_file, base_file=set_file_location
        )
        if main_file_name not in df["location"].values:
            print(f"Analysing {set_file_location}")
            trial_number = int(folder[-1])

            passed_bit = matching_rat_date[
                (matching_rat_date["SesNo"] == session_no)
                & (matching_rat_date["TrialNo"] == trial_number)
            ]
            if len(passed_bit) == 1:
                passed = passed_bit["pass/fail"].values.flatten()[0]
                print(f"The rat passed (Y) or failed (N)? {passed}")
            else:
                print("WARNING unable to get pass/fail for this trial")
                print("Please set manually after, currently FAILED_TO_FIND")
                passed = "FAILED_TO_FIND"
            done = False
            while not done:
                times = analyse_recording(recording)
                if times == "QUIT":
                    print("Saving results to {}".format(xls_location))
                    process_list_to_df(df, data_list, columns, xls_location)
                    return
                if len(times) != 6:
                    print("Incorrect number of times, retrying")
                    print("Times should be (start, choice_made, end) for two trials")
                else:
                    done = True
            data = [
                main_file_name,
                times[0],
                times[1],
                times[2],
                session_no,
                trial_number,
                animal,
                date,
                "first",
                passed.strip().upper(),
                os.path.basename(mapping_file),
            ]
            data_list.append(data)
            data = [
                main_file_name,
                times[3],
                times[4],
                times[5],
                session_no,
                trial_number,
                animal,
                date,
                "second",
                passed.strip().upper(),
                os.path.basename(mapping_file),
            ]
            data_list.append(data)

    print("Saving results to {}".format(xls_location))
    process_list_to_df(df, data_list, columns, xls_location)


def process_list_to_df(orig_df, list_, columns, out_name):
    df = pd.DataFrame(list_, columns=columns)
    df = pd.concat((orig_df, df))
    if os.path.exists(out_name):
        split = os.path.splitext(out_name)
        new_name = split[0] + "__copy" + split[1]
        shutil.copy(out_name, new_name)

    try:
        df.to_excel(out_name, index=False)
    except PermissionError as e:
        print(f"Please close {out_name}.")
        inp = input("When closed, press y to continue\n")
        if inp.lower().strip() == "y":
            df.to_excel(out_name, index=False)


if __name__ == "__main__":
    # Mark (start, decision_point, end) using space
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="t-maze cli")
    parser.add_argument("trial_id", type=str, help="trial id as animal_tX")
    parsed = parser.parse_args()
    name_to_get = parsed.trial_id

    td = t_maze_dict()
    to_analyse = td[name_to_get]
    main_t_maze_dir = to_analyse["t_maze_dir"]
    main_base_dir = to_analyse["base_dir"]
    main_mapping_location = to_analyse["mapping_location"]
    main_animal = to_analyse["animal"]
    main_date = to_analyse["date"]
    main_session_no = to_analyse["session_no"]

    # Determined locations for loading and saving
    main_output_location = os.path.join(here, "results")
    os.makedirs(main_output_location, exist_ok=True)
    xls_location = os.path.join(main_output_location, "tmaze-times.xlsx")

    main(
        main_t_maze_dir,
        main_output_location,
        main_base_dir,
        main_mapping_location,
        xls_location,
        animal=main_animal,
        date=main_date,
        session_no=main_session_no,
    )
