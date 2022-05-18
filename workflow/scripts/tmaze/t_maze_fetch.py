# See the __main__ for parameters
import os
import shutil
from site import addsitedir
import argparse

import simuran
import pandas as pd
from skm_pyutils.py_path import get_all_files_in_dir

addsitedir(os.path.join(os.path.dirname(__file__)))
from t_maze_layout import t_maze_dict


def main(
    t_maze_dir,
    output_location,
    xls_location,
    base_dir,
    animal="",
    date="",
    session_no="",
):
    """Create a single recording for analysis."""
    # loop over the sessions in each t-maze folder
    columns = ["fname", "passed"]

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

    list_ = []
    for folder in os.listdir(t_maze_dir):
        dir_loc = os.path.join(t_maze_dir, folder)
        set_file_locations = get_all_files_in_dir(dir_loc, ext=".set")
        if len(set_file_locations) == 0:
            raise ValueError(f"No set files were found in {dir_loc}")
        set_file_location = set_file_locations[0]
        main_file_name = set_file_location[len(base_dir + os.sep) :].replace(
            os.sep, "--"
        )
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
        list_.append([main_file_name, passed])
        list_.append([main_file_name, passed])

    df = pd.DataFrame(list_, columns=columns)
    df.to_excel(xls_location, index=False)

    print(df)


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
    main_animal = to_analyse["animal"]
    main_date = to_analyse["date"]
    main_session_no = to_analyse["session_no"]

    # Determined locations for loading and saving
    main_output_location = os.path.join(here, "results")
    os.makedirs(main_output_location, exist_ok=True)
    xls_location = os.path.join(main_output_location, "tmaze-times_fetched.xlsx")

    main(
        main_t_maze_dir,
        main_output_location,
        xls_location,
        main_base_dir,
        animal=main_animal,
        date=main_date,
        session_no=main_session_no,
    )
