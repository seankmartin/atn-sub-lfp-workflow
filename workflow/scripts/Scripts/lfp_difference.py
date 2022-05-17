import os

import numpy as np

from skm_pyutils.py_path import make_path_if_not_exists
import simuran.plot.custom.lfp_plot
from simuran.plot.figure import SimuranFigure


def get_normalised_diff(s1, s2, s1_sq=None, s2_sq=None):
    # MSE of one divided by MSE of main - Normalized squared difference
    # Symmetric
    s1_sq = np.square(s1) if s1_sq is None else s1_sq
    s2_sq = np.square(s2) if s2_sq is None else s2_sq
    return np.sum(np.square(s1 - s2)) / (np.sum(s1_sq + s2_sq) / 2)
    # return np.sum(np.square(s1 - s2)) / s1_sq)  # Non-symmetric


def compare_lfp(
    recording,
    figures,
    out_base_dir=None,
    rel_dir=None,
    ch_to_use="all",
    save_result=True,
    plot=False,
):
    """
    Compare LFP channels for differences.

    Parameters
    ----------
    recording : simuran.recording.Recording
        Recording holding data
    out_base_dir : str, None
        Path for desired output location.
        Default - Saves output to folder named !LFP in base directory.
    ch: int
        Number of LFP channels in session

    """
    # Do the actual calculation
    if ch_to_use == "all":
        ch_labels = recording.get_signal_channels()
    else:
        ch_labels = ch_to_use
    ch = [i for i in range(len(ch_labels))]

    grid = np.meshgrid(ch, ch, indexing="ij")
    stacked = np.stack(grid, 2)
    pairs = stacked.reshape(-1, 2)
    result_a = np.zeros(shape=pairs.shape[0], dtype=np.float32)

    cached_sum_squares = [
        np.sum(np.square(signal.samples)) for signal in recording.signals
    ]
    for i, pair in enumerate(pairs):
        signal1 = recording.signals[pair[0]]
        signal2 = recording.signals[pair[1]]
        sq1 = cached_sum_squares[pair[0]]
        sq2 = cached_sum_squares[pair[1]]
        res = get_normalised_diff(
            signal1.samples, signal2.samples, s1_sq=sq1, s2_sq=sq2
        )
        result_a[i] = res

    # Save out a csv and do plotting
    if out_base_dir is None:
        out_base_dir = os.path.dirname(recording.source_file)
    if rel_dir is None:
        rel_dir = out_base_dir

    base_name_part = recording.get_name_for_save(rel_dir=rel_dir)

    if save_result:
        out_name = base_name_part + "_LFP_Comp.csv"
        out_loc = os.path.join(out_base_dir, out_name)
        make_path_if_not_exists(out_loc)
        print("Saving csv to {}".format(out_loc))
        with open(out_loc, "w") as f:
            headers = [str(i) for i in ch]
            out_str = ",".join(headers)
            f.write(out_str)
            out_str = ""
            for i, (pair, val) in enumerate(zip(pairs, result_a)):
                if i % len(ch) == 0:
                    f.write(out_str + "\n")
                    out_str = ""

                out_str += "{:.2f},".format(val)
            f.write(out_str + "\n")

    if plot:
        out_name = base_name_part + "_LFP_Comp.png"
        fig = simuran.plot.custom.lfp_plot.plot_compare_lfp(result_a, ch, save=False)
        figures.append(SimuranFigure(fig, out_name, dpi=400, done=True, format="png"))

    regions = recording.signals.get_property("region")
    results = np.reshape(result_a, newshape=[len(ch), len(ch)])
    results_dict = {"sub_diff": float(0), "rsc_diff": float(0)}
    sub_count, rsc_count = 0, 0
    current = "START"
    for i in range(len(regions)):
        if regions[i] == "RSC":
            rsc_count += 1
            if current == "MIDDLE":
                raise ValueError("Regions must be in continuous blocks")
        elif regions[i] == "SUB":
            sub_count += 1
            if current == "START":
                current = "MIDDLE"

    results_dict["sub_diff"] = float(np.mean(results[:rsc_count, :rsc_count]))
    results_dict["rsc_diff"] = float(np.mean(results[rsc_count:, rsc_count:]))

    return {"full": result_a, "summary": results_dict}


def average_difference(recording_container, figures, plot=False, fmt="png"):
    results_list = [r["compare_lfp"]["full"] for r in recording_container.get_results()]
    save_name = os.path.basename(recording_container.base_dir)
    ch = [i for i in range(len(recording_container[0].get_signal_channels()))]

    results = np.zeros(results_list[0].shape)
    for result_a in results_list:
        results = results + result_a
    results = results / len(results_list)
    results = np.reshape(results, newshape=[len(ch), len(ch)])

    if plot:
        print("should save to {}".format(save_name))
        save_name = save_name + "_LFP_AVG_Comp.png"
        fig = simuran.plot.custom.lfp_plot.plot_compare_lfp(results, ch, save=False)
        figures.append(SimuranFigure(fig, save_name, dpi=400, done=True, format=fmt))

    return results
