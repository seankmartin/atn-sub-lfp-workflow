"""Find the rate map of LFP in open-field."""
import logging
from copy import deepcopy

import numpy as np
from collections import OrderedDict as oDict
from neurochat.nc_utils import chop_edges, histogram2d, smooth_2d
from neurochat.nc_plot import _make_ax_if_none
from neurochat.nc_utils import butter_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from math import floor, ceil

from lfp_atn_simuran.Scripts.lfp_clean import LFPClean
import simuran


def lfp_rate_recording(recording, base_dir, figures, **kwargs):
    min_f = kwargs.get("fmin", 1.0)
    max_f = kwargs.get("fmax", 100)

    # Delta and Theta
    low_f = [1.5, 6]
    high_f = [4, 10]
    band_name = ["Delta", "Theta"]

    save_format = kwargs.get("image_format", "png")
    clean_method = kwargs.get("clean_method", "avg")
    clean_kwargs = kwargs.get("clean_kwargs", {})

    lc = LFPClean(method=clean_method, visualise=False)
    region_sigs = lc.clean(recording, min_f, max_f, method_kwargs=clean_kwargs)[
        "signals"
    ]
    save_name = recording.get_name_for_save(rel_dir=base_dir)
    results = {}
    fig, ax = plt.subplots(2, 2)
    spatial = recording.spatial.underlying

    for j in range(len(low_f)):
        low = low_f[j]
        high = high_f[j]
        band = band_name[j]

        for i, (region, signal) in enumerate(region_sigs.items()):
            data = lfp_rate(
                spatial,
                signal,
                low_f=low,
                high_f=high,
                filter_kwargs={"verbose": "WARNING"},
            )
            lfp_rate_plot(data, ax=ax[j][i])
            ax[j][i].set_title(f"{region}: {low}-{high} Hz")
            nc_results = spatial.get_results()
            for name, value in nc_results.items():
                results[f"{region}--{band}--{name}"] = value
            spatial.reset_results()

    fig.tight_layout()
    out_name = f"{save_name}--lfp_rate"
    figures.append(
        simuran.SimuranFigure(fig, out_name, dpi=300, format=save_format, done=True)
    )

    return results


def lfp_rate(self, lfp_signal, low_f=None, high_f=None, filter_kwargs=None, **kwargs):
    """Calculate LFP rate map."""
    _results = oDict()
    graph_data = {}
    update = kwargs.get("update", True)
    pixel = kwargs.get("pixel", 3)
    chop_bound = kwargs.get("chop_bound", 5)
    filttype, filtsize = kwargs.get("filter", ["b", 5])
    lim = kwargs.get("range", [0, self.get_duration()])
    brAdjust = kwargs.get("brAdjust", False)
    # Can pass another NData object to estimate the border from
    # Can be useful in some cases, such as when the animal
    # only explores a subset of the arena.
    separate_border_data = kwargs.get("separateBorderData", None)
    samples_per_sec = kwargs.get("samplesPerSec", 10)

    if low_f is not None:
        if filter_kwargs is None:
            filter_kwargs = {}
        try:
            lfp_signal = lfp_signal.filter(low_f, high_f, **filter_kwargs)
        except BaseException:
            lfp_signal = deepcopy(lfp_signal)
            _filt = [10, low_f, high_f, "bandpass"]
            lfp_signal.set_samples(
                butter_filter(
                    lfp_signal.get_samples(), lfp_signal.get_sampling_rate(), _filt
                )
            )

    if separate_border_data is not None:
        self.set_border(separate_border_data.calc_border(**kwargs))
        times = self._time
        lower, upper = (times.min(), times.max())
        new_times = separate_border_data._time
        sample_spatial_idx = ((new_times <= upper) & (new_times >= lower)).nonzero()
        self._border_dist = self._border_dist[sample_spatial_idx]
    else:
        self.set_border(self.calc_border(**kwargs))

    xedges = self._xbound
    yedges = self._ybound

    posX = self._pos_x[
        np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
    ]
    posY = self._pos_y[
        np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
    ]
    time_to_use = self.get_time()[
        np.logical_and(self.get_time() >= lim[0], self.get_time() <= lim[1])
    ]
    skip_rate = int(self.get_sampling_rate() / samples_per_sec)
    slicer = slice(skip_rate, -skip_rate, skip_rate)
    posX = posX[slicer]
    posY = posY[slicer]
    time_to_use = time_to_use[slicer]

    tmap, yedges, xedges = histogram2d(posY, posX, yedges, xedges)

    if (
        tmap.shape[0]
        != tmap.shape[1] & np.abs(tmap.shape[0] - tmap.shape[1])
        <= chop_bound
    ):
        tmap = chop_edges(tmap, min(tmap.shape), min(tmap.shape))[2]

    ybin, xbin = tmap.shape
    xedges = np.arange(xbin) * pixel
    yedges = np.arange(ybin) * pixel

    binned_lfp = np.zeros_like(tmap)

    lfp_amplitudes = np.zeros_like(time_to_use)
    lfp_samples = lfp_signal.get_samples()
    do_once = True

    # Want value in mv
    if hasattr(lfp_samples, "unit"):
        import astropy.units as u

        lfp_samples = lfp_samples.to(u.uV).value
    else:
        lfp_samples = lfp_samples * 1000

    for i, t in enumerate(time_to_use):
        low_sample = floor((t - 0.05) * lfp_signal.get_sampling_rate())
        high_sample = ceil((t + 0.05) * lfp_signal.get_sampling_rate())
        if high_sample < len(lfp_samples):
            lfp_amplitudes[i] = np.mean(
                np.abs(lfp_samples[low_sample : high_sample + 1])
            )
        elif do_once:
            simuran.log.warning(
                "Position data ({}s) is longer than EEG data ({}s)".format(
                    time_to_use[-1], len(lfp_samples) / lfp_signal.get_sampling_rate()
                )
            )
            do_once = False

    x_idx = np.digitize(posX, xedges) - 1
    y_idx = np.digitize(posY, yedges) - 1
    for k, (i, j) in enumerate(zip(x_idx, y_idx)):
        binned_lfp[j, i] += lfp_amplitudes[k]

    fmap = np.divide(binned_lfp, tmap, out=np.zeros_like(binned_lfp), where=tmap != 0)
    rate_tmap = tmap / samples_per_sec
    ps_fmap = np.divide(
        binned_lfp, rate_tmap, out=np.zeros_like(binned_lfp), where=rate_tmap != 0
    )

    if brAdjust:
        nfmap = fmap / fmap.max()
        if (
            np.sum(np.logical_and(nfmap >= 0.2, tmap != 0))
            >= 0.8 * nfmap[tmap != 0].flatten().shape[0]
        ):
            back_rate = np.mean(fmap[np.logical_and(nfmap >= 0.2, nfmap < 0.4)])
            fmap -= back_rate
            fmap[fmap < 0] = 0

    if filttype is not None:
        smoothMap = smooth_2d(fmap, filttype, filtsize)
        smoothMap2 = smooth_2d(ps_fmap, filttype, filtsize)
    else:
        smoothMap = fmap
        smoothMap2 = fmap

    if update:
        _results["Spatial Skaggs"] = self.skaggs_info(np.abs(ps_fmap), rate_tmap)
        _results["Spatial Sparsity"] = self.spatial_sparsity(np.abs(ps_fmap), rate_tmap)
        _results["Spatial Coherence"] = np.corrcoef(
            ps_fmap[rate_tmap != 0].flatten(), smoothMap2[rate_tmap != 0].flatten()
        )[0, 1]
        _results["Peak Firing Rate"] = fmap.max()
        self.update_result(_results)

    smoothMap[tmap == 0] = None

    graph_data["posX"] = posX
    graph_data["posY"] = posY
    graph_data["fmap"] = fmap
    graph_data["smoothMap"] = smoothMap
    graph_data["firingMap"] = fmap
    graph_data["tmap"] = tmap
    graph_data["xedges"] = xedges
    graph_data["yedges"] = yedges
    graph_data["lfpMap"] = binned_lfp

    return graph_data


def lfp_rate_plot(place_data, ax=None, smooth=True, **kwargs):
    """
    Plot location vs spike rate.

    By default, colormap="viridis", style="contour".
    However, the old NC style was colormap="default", style="digitized".
    The old style produces very nice maps, but not colorblind friendly.

    Parameters
    ----------
    place_data : dict
        Graphical data from the unit firing to locational correlation
    ax : matplotlib.pyplot.axis
        Axis object. If specified, the figure is plotted in this axis.
    kwargs :
        colormap : str
            viridis is used if not specified
            "default" uses the standard red green intensity colours
            but these are bad for colorblindness.
        style : str
            What kind of map to plot - can be
            "contour", "digitized" or "interpolated"
        levels : int
            Number of contour regions.

    Returns
    -------
    ax : matplotlib.pyplot.Axis
        Axis of the firing rate map

    """
    colormap = kwargs.get("colormap", "viridis")
    style = kwargs.get("style", "contour")
    levels = kwargs.get("levels", 5)
    raster = kwargs.get("raster", True)
    splits = None

    if colormap == "default":
        clist = [
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.5),
            (0.9, 1.0, 0.0),
            (1.0, 0.75, 0.0),
            (0.9, 0.0, 0.0),
        ]
        colormap = mcol.ListedColormap(clist)

    ax, fig = _make_ax_if_none(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    if smooth:
        fmap = place_data["smoothMap"]
    else:
        fmap = place_data["firingMap"]

    if style == "digitized":
        res = ax.pcolormesh(
            place_data["xedges"],
            place_data["yedges"],
            np.ma.array(fmap, mask=np.isnan(fmap)),
            cmap=colormap,
            rasterized=raster,
        )

    elif style == "interpolated":
        extent = (0, place_data["xedges"].max(), 0, place_data["yedges"].max())
        tp = fmap[:-1, :-1]
        res = ax.imshow(
            tp, cmap=colormap, extent=extent, interpolation="bicubic", origin="lower"
        )

    elif style == "contour":
        dx = np.mean(np.diff(place_data["xedges"]))
        dy = np.mean(np.diff(place_data["yedges"]))
        pad_map = np.pad(fmap[:-1, :-1], ((1, 1), (1, 1)), "edge")
        vmin, vmax = np.nanmin(pad_map), np.nanmax(pad_map)
        if vmax - vmin > 0.1:
            splits = np.linspace(vmin, vmax, levels + 1)
        else:
            splits = np.linspace(vmin, vmin + 0.1 * levels, levels + 1)
        splits = np.around(splits, decimals=1)
        to_delete = []
        for i in range(len(splits) - 1):
            if splits[i] >= splits[i + 1]:
                to_delete.append(i)
        splits = np.delete(splits, to_delete)
        x_edges = np.append(
            place_data["xedges"] - dx / 2, place_data["xedges"][-1] + dx / 2
        )
        y_edges = np.append(
            place_data["yedges"] - dy / 2, place_data["yedges"][-1] + dy / 2
        )
        res = ax.contourf(
            x_edges,
            y_edges,
            np.ma.array(pad_map, mask=np.isnan(pad_map)),
            levels=splits,
            cmap=colormap,
            corner_mask=True,
        )

        # This produces it with no padding
        # res = ax.contourf(
        #     place_data['xedges'][:-1] + dx / 2.,
        #     place_data['yedges'][:-1] + dy / 2.,
        #     np.ma.array(fmap[:-1, :-1], mask=np.isnan(fmap[:-1, :-1])),
        #     levels=15, cmap=colormap, corner_mask=True)

    else:
        logging.error("Unrecognised style passed to loc_rate")
        return

    ax.set_ylim([0, place_data["yedges"].max()])
    ax.set_xlim([0, place_data["xedges"].max()])
    ax.set_aspect("equal")
    ax.invert_yaxis()
    cbar = plt.colorbar(res, cax=cax, orientation="vertical", use_gridspec=True)
    ax.set_title("Amplitude in uV")

    return ax
