import json
import sys
import sheets as s
import pandas as pd
import paths as p
from pathlib import Path
import oscfar as ocf
import glob
import scipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from chime_frb_api.backends.frb_master import FRBMaster
from ui import Color, Style

RESET = Style.RESET_ALL

sys.path.append("fitburst")
from fitburstmodel import FitBurstModel

try:
    master = FRBMaster()
except Exception as e:
    print("FRB Master problemo")
    master = None

np = ocf.np

# _, sheet = s.get_spreadsheet("Chloé's fitburst (G)")

# df = pd.DataFrame(sheet[2:])
# df.columns = sheet[1]

# df["Struct-max DM (pc cm−3)"] = df["Struct-max DM (pc cm−3)"].astype(float)


def find_file(event_id: int | str):
    if isinstance(event_id, str) and Path(event_id).exists():
        return event_id
    a = glob.glob(f"{p.NPZ_SAVE_PATH}/{event_id}.npz")
    return a[0] if len(a) >= 1 else None


def boxcar_kernel(width):
    """Returns the boxcar kernel of given width normalized by
    sqrt(width) for S/N reasons.
    Parameters
    ----------
    width : int
            Width of the boxcar.
    Returns
    -------
    boxcar : array_like
            Boxcar of width `width` normalized by sqrt(width).
    """
    width = int(round(width, 0))
    return np.ones(width, dtype="float32") / np.sqrt(width)


def find_burst(ts, min_width=1, max_width=128):
    """Find burst peak and width using boxcar convolution.
    Parameters
    ----------
    ts : array_like
            Time-series.
    min_width : int, optional
            Minimum width to search from, in number of time samples.
            1 by default.
    max_width : int, optional
            Maximum width to search up to, in number of time samples.
            128 by default.  plot : bool, optional
            If True, show figure to summarize burst finding results.
    Returns
    -------
    peak : int
            Index of the peak of the burst in the time-series.
    width : int
            Width of the burst in number of samples.
    snr : float
            S/N of the burst.

    """
    min_width = int(min_width)
    max_width = int(max_width)

    # do not search widths bigger than timeseries
    widths = list(range(min_width, min(max_width + 1, len(ts) - 2)))

    # envelope finding
    snrs = np.empty_like(widths, dtype=float)
    peaks = np.empty_like(widths, dtype=int)

    for i in range(len(widths)):
        convolved = scipy.signal.convolve(ts, boxcar_kernel(widths[i]), mode="same")
        peaks[i] = np.nanargmax(convolved)
        snrs[i] = convolved[peaks[i]]

    best_idx = np.nanargmax(snrs)

    return peaks[best_idx], widths[best_idx], snrs[best_idx]


def get_tns(event_id: int):
    if not isinstance(event_id, int):
        try:
            event_id = int(event_id)
        except:
            print(
                "Cannot find TNS name for {}. Invalid INT conversion".format(event_id)
            )
            return

    if master is None:
        return ""
    res = master.tns.search(event_id)
    return res.get("tns_name")


def get_model(npz: ocf.npz_reader, fitburst_path: str):
    if not Path(fitburst_path).exists():
        print("Invalid fitburst model path")
        print(fitburst_path)
        return

    with open(fitburst_path) as f:
        data = json.load(f)
    model_parameters = data.get("model_parameters")
    scint = data["fit_statistics"]["bestfit_parameters"].get("amplitude") is None
    print(scint)

    modeler = FitBurstModel(
        npz.freqs,
        npz.times,
        num_components=len(model_parameters["arrival_time"]),
        scintillation=scint,
    )
    modeler.update_parameters(model_parameters)
    computed_model = modeler.compute_model(npz.data_full)
    return computed_model, computed_model.mean(0), computed_model.mean(1)


def _plot_event_helper(
    fname: str,
    event_id: str,
    outer,
    xlabel: bool,
    ylabel: bool,
    textsize: int = 8,
    titlesize: int = 14,
    fitburst_path: str = None,
    tns_name_override="",
    plot_2d_model=False,
    boxcar=False,
    x_range=None,
) -> ocf.utils.NpzReader:
    print("Plotting {}..".format(fname))
    print(f"Using {titlesize} for title and {textsize} for labels.")

    npz_file = ocf.npz_reader(fname)
    if tns_name_override:
        tns_name = tns_name_override
    else:
        tns_name = get_tns(event_id)

    if tns_name is None:
        tns_name = tns_name_override

    if fitburst_path is not None:
        model, modeled_profile, modeled_spectrum = get_model(npz_file, fitburst_path)

    all_freqs = npz_file.freqs[::-1]
    dm = float(npz_file.metadata.get("dm", 0))
    good_channels = np.sum(npz_file.data_full, axis=1) != 0.0
    empty_channels = ~good_channels

    full_wfall = npz_file.data_full
    full_wfall[empty_channels] = np.nan

    ts = np.nanmean(full_wfall, axis=0)

    if fitburst_path is not None:
        model[np.isnan(full_wfall)] = np.nan

    peak, width, snr = find_burst(ts if not plot_2d_model else np.nanmean(model, 0))

    dt = npz_file.res_time
    df = npz_file.res_freq

    gs = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=outer,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        hspace=0,
        wspace=0,
    )

    data_im = plt.subplot(gs[2])
    data_ts = plt.subplot(gs[0], sharex=data_im)
    data_spec = plt.subplot(gs[3], sharey=data_im)

    # plt.subplot(gs[1]).axis("off")

    vmin = np.nanpercentile(full_wfall, 5)
    vmax = np.nanpercentile(full_wfall, 95)

    # tick
    window = width * 30
    limits = [peak - window, peak + window]

    # if limits[0] < 0:
    #     limits[1] = limits[1] + abs(limits[0])
    #     if limits[1] > len(ts):
    #         limits[1] = len(ts)
    #     limits[0] = 0
    # if limits[1] > len(ts):
    #     limits[0] = limits[0] - abs(limits[1] - len(ts))
    #     if limits[0] < 0:
    #         limits[0] = 0
    #     limits[1] = len(ts)

    limits = [0, len(ts) - 1]

    on_spec = np.nanmean(
        full_wfall[:, max(0, peak - width) : min(len(ts), peak + width)], axis=1
    )

    # Fix masking:
    if fitburst_path is not None:
        residuals = full_wfall - model
        res_med = np.nanmedian(residuals)
        res_std = np.nanstd(residuals)
        model[np.isnan(full_wfall)] = np.nanmedian(full_wfall)

        vmin = res_med - res_std * 4
        vmax = res_med + res_std * 4
    full_wfall[np.isnan(full_wfall)] = np.nanmedian(full_wfall)

    extent = (
        (limits[0] - peak) * dt * 1e3,
        (limits[1] - peak) * dt * 1e3,
        all_freqs[-1] - df / 2.0,
        all_freqs[0] + df / 2.0,
    )

    print(full_wfall[:, limits[0] : limits[1]].shape)

    # Plot the data:
    data_im.imshow(
        full_wfall[:, limits[0] : limits[1]] if not plot_2d_model else model,
        extent=extent,
        interpolation="none",
        cmap=cm.viridis,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        origin="lower",
    )

    if boxcar:
        data_ts.axvspan(
            -width * dt * 1e3,
            width * dt * 1e3,
            facecolor="tab:blue",
            edgecolor="none",
            alpha=0.4,
        )

    # x_values = np.arange(limits[0] - peak, limits[1] - peak + 1) * dt * 1e3
    x_values = (np.array(npz_file.times) - peak * dt) * 1e3
    data_ts.plot(
        # np.linspace(extent[0], extent[1], limits[1] - limits[0] + 1),
        x_values,
        np.append(ts[limits[0] : limits[1]], ts[limits[0] : limits[1]][-1])
        if not plot_2d_model
        else np.nanmean(model, 0),
        color="tab:blue",
        drawstyle="steps-post",
        lw=0.5,
    )

    if fitburst_path is not None and not plot_2d_model:
        data_ts.plot(x_values, modeled_profile, lw=1, color="tab:orange")

    y_values = all_freqs[::-1]
    data_spec.plot(
        on_spec if not plot_2d_model else np.nanmean(model, 1),
        y_values,
        color="tab:blue",
        lw=0.5,
    )

    if fitburst_path is not None and not plot_2d_model:
        data_spec.plot(modeled_spectrum, y_values, lw=1, color="tab:orange")

    plt.setp(data_im.get_xticklabels(), fontsize=textsize)
    plt.setp(data_im.get_yticklabels(), fontsize=textsize)

    # remove some labels and ticks
    plt.setp(data_ts.get_xticklabels(), visible=False)
    data_ts.set_yticklabels([], visible=False)
    # data_ts.set_yticks([])
    data_ts.set_xlim(extent[0], extent[1])
    plt.setp(data_spec.get_yticklabels(), visible=False)
    data_spec.set_xticklabels([], visible=True)
    data_spec.set_xticks([])
    data_spec.set_ylim(extent[2], extent[3])

    if x_range:
        data_im.set_xlim(*x_range)

    # add event ID and DM
    xlim = data_ts.get_xlim()
    ylim = data_ts.get_ylim()
    x_offset = extent[0]

    if x_range:
        xlim = x_range
        x_offset = x_range[0]

    # add 20% extra white space at the top
    span = np.abs(ylim[1]) + np.abs(ylim[0])
    data_ts.set_ylim(ylim[0], ylim[1] + 0.2 * span)
    ylim = data_ts.get_ylim()

    ypos = (ylim[1] - ylim[0]) * 0.9 + ylim[0]
    xpos = (xlim[1] - xlim[0]) * 0.98 + x_offset

    print(xpos, ypos)
    data_ts.text(
        xpos,
        ypos,
        "{}\n{:.1f} pc/cc\n{:.2f} us".format(tns_name, dm, dt * 1e6),
        ha="right",
        va="top",
        fontsize=titlesize,
    )
    # data_ts.set_title(tns_name, fontsize=titlesize)

    data_im.locator_params(axis="x", min_n_ticks=3)

    if ylabel:
        data_im.set_yticks([400, 500, 600, 700, 800])
        data_im.set_ylabel("Frequency (MHz)", fontsize=textsize)
    else:
        data_im.set_yticklabels([], visible=False)
        data_im.set_yticks([])
    if xlabel:
        data_im.set_xlabel("Time (ms)", fontsize=textsize)

    return npz_file, extent


def plot_event(
    event_id: str,
    plot_fitburst=False,
    figsize=(10, 10),
    tns_override="",
    file_override="",
    plot_2d=False,
    boxcar=False,
    x_range=None,
    fontsize=(16, 14),
) -> ocf.utils.NpzReader:
    # print(fontsize)
    fig = plt.figure(figsize=figsize)
    path = find_file(event_id)
    fit_file = None
    if plot_fitburst:
        info = s.get_info(event_id)
        fit_file = info.get("Path").replace("/arc", p.CANFAR_ROOT_PATH)
    if file_override:
        path = file_override

    outer_grid = gridspec.GridSpec(1, 1, wspace=0, hspace=0.25, figure=fig)

    for i, outer in enumerate(outer_grid):
        npz, range = _plot_event_helper(
            path,
            event_id,
            outer,
            True,
            True,
            fontsize[0],
            fontsize[1],
            fit_file,
            tns_name_override=tns_override,
            plot_2d_model=plot_2d,
            boxcar=boxcar,
            x_range=x_range,
        )
    print(Color.fg(2) + "2D Spectrum range" + RESET)
    print(f"Time range: {range[0]} to {range[1]}")
    print(f"Frequency range: {range[2]} to {range[3]}")

    return npz
