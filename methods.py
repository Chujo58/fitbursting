import sys, os, logging

#####################################################
##### Configure paths for python and for data saving.
#####################################################

other_paths = [
    "/usr/local/lib/python38.zip",
    "/usr/local/lib/python3.8",
    "/usr/local/lib/python3.8/lib-dynload",
    "/opt/pysetup/.venv/lib/python3.8/site-packages",
    "/opt/pysetup",
]
sys.path += other_paths

from paths import BASEBAND_ANALYSIS_PATH, DM_PHASE_PATH

sys.path.insert(0, DM_PHASE_PATH)
sys.path.append(BASEBAND_ANALYSIS_PATH)

# Library imports

from baseband_analysis.core import bbdata as BBData
import numpy as np
import matplotlib.pyplot as plt
from chime_frb_constants import K_DM, FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ
from scipy.optimize import least_squares, curve_fit
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import chime_frb_api
from time import sleep

from chime_frb_api import frb_master
import datetime as dt

from baseband_analysis.core import BBData
from baseband_analysis.core.signal import (
    get_profile,
    get_floor,
    get_main_peak_lim,
    tiedbeam_baseband_to_power,
    get_power,
)
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.dedispersion import coherent_dedisp
from baseband_analysis.dev.Morphology_utils import *
from baseband_analysis.pipelines.run_fitburst import run_fitburst
from baseband_analysis.dev.Morphology_dev import *

import fitburst

logger = logging.getLogger(__name__)

#####################
##### Useful methods.
#####################

##### Implement peak finding and initial guess next ####
#### Finally make it into a fitburst input #####

####### The general format of the function #######
"""
1. It loads the singlebeam file - Has option for coherent dedispersion
2. Has all the options needed by get_snr() - See all that are optional and fixed
3. Perfoms structure maximizng DM  - Optional
4. Finds all the peaks - Optional, Provide the number of peaks or timestamp of peaks (manually give position)
5. Add an option here to make a fitburst input file with all the default value and just the peak information
6. Provide the initial guess using MCMC. Make it run exactly for the number of peaks defined above
7. Add an option to allow it to run for multiple peak options, i.e. it finds the best number of peak by calculating chi-sq
"""


# Create a function to show differences from two dictionaries, showing an arrow from old to new values for each key that has changed. This is useful to show the changes in the burst parameters after the MCMC fit.
def show_dict_diff(old_dict, new_dict):
    for key in old_dict.keys():
        if key in new_dict:
            if old_dict[key] != new_dict[key]:
                print(f"{key}: {old_dict[key]} --> {new_dict[key]}")
        else:
            print(f"{key}: {old_dict[key]} --> (removed)")
    for key in new_dict.keys():
        if key not in old_dict:
            print(f"{key}: (added) --> {new_dict[key]}")


def get_signal_time_range(
    power, downsampling_factor, time_range: tuple = None, full_time: bool = False
) -> tuple:
    """
    This function gets the start and end time of the signal in the power.

    Parameters
    ----------

    power : np.ndarray
        The power of the event_id generated using get_snr()
    downsampling_factor: int
        The dowsampling factor used to generate the power
    time_range : tuple, optional
        Bin numbers where to start and end the baseband data at downsampling 'downsample'. e.g [1200, 1400]
        This will only extract data from time bins 1200 to 1400
    full_time : bool, optional
        Set this true and it will just extract the entire power and give the start and end of the power

    Returns
    -------

    profile : np.ndarray
        The timeseries of the profile between the given start and end time
    start : int
        The bin where profile starts
    end : int
        The bin where profile ends
    """

    profile = get_profile(power)

    if full_time:
        logger.info("Reading full time")
        start, end = 0, power.shape[1]

    else:
        if time_range is None:
            logger.info("Getting the signal")
            start, end = get_signal(profile, ds=downsampling_factor)
        else:
            start, end = time_range[0], time_range[1]

    profile = profile[start:end]

    return profile, start, end


def get_best_downsamp(
    prof: np.ndarray, downsamp_fact: int = 1, snr_cutoff: int = 20
) -> int:
    """
    This function estimates the best downsampling factor for
    the burst using its profile at the given snr cutoff, upto max of 256 downsamp

    Parameters
    ----------
    prof : np.ndarray
        The timeseries profile of the burst
    downsamp_fact : int
        The intial dowsample of the profile, default is 1 i.e. the signal is assumed to be at the native resolution
    snr_cutoff: int
        The signal to noise cutoff. The higher it is the the higher downsample it will go to. S/N 20 was selected
        as default to be same as the polarization pipeline

    Returns
    -------
    downsamp_fact : int
        The best downsample for the burst

    """

    ##### Checking whether the input downsample if already the best ######
    # profile already comes in downsampled to factor downsamp_fact
    logger.info(f"Profile already came in downsampled to: {downsamp_fact}")
    SNr = prof.copy()

    downsamp_fact = downsamp_fact

    init_run = int(
        np.log2(downsamp_fact)
    )  #### To get from where to start downsample ###

    logger.info(
        f"SNR for initial downsampling factor {downsamp_fact}: {np.nanmax(SNr)}"
    )

    # further downsamples the already downsampled profile
    if np.nanmax(SNr) > snr_cutoff:
        logger.info(
            "The burst is already bright enough given the snr_cutoff at this downsample"
        )
        return downsamp_fact
    else:
        logger.warning(
            "The input dowsample was not bright enough. Finding the best downsample ....."
        )
        downsamp_fact_orig = downsamp_fact
        max_fact = 9 - init_run
        for downsamp_fact in 2 ** np.arange(1, max_fact):
            SNr = prof.copy()
            SNr = SNr[: SNr.size // downsamp_fact * downsamp_fact]
            SNr = np.nansum(
                SNr.reshape([int(SNr.size / downsamp_fact), downsamp_fact]), axis=-1
            ) / np.sqrt(downsamp_fact)
            ds = downsamp_fact * downsamp_fact_orig

            x = np.arange(len(SNr))
            plt.plot(x, SNr)
            plt.title(f"Profile for downsamp of {ds}")
            plt.show()
            logger.info(
                f"SNR for downsampling factor {downsamp_fact * downsamp_fact_orig}: {np.nanmax(SNr)}"
            )
            if np.nanmax(SNr) > snr_cutoff:
                downsamp_fact = downsamp_fact * downsamp_fact_orig
                break
            if downsamp_fact == 2 ** (max_fact - 1):
                logger.error(
                    f"SNR of {snr_cutoff} never reached. Stopping at dsamp of {downsamp_fact * downsamp_fact_orig}"
                )
                downsamp_fact = downsamp_fact * downsamp_fact_orig
    logger.info(f"The final downsample is {int(downsamp_fact)}")
    return int(downsamp_fact)


def run_singlebeam(
    path: str,
    DM: float = None,
    downsample: int = None,
    time_range: tuple = None,
    fit_DM: bool = True,
    dm_range: int = 5,
    spectrum_lim: bool = False,
    fill_missing_time: bool = None,
    diagnostic_plots: bool = False,
    raise_missing_signal: bool = False,
    full_time: bool = False,
    snr_cutoff: int = 20,
    co_dedisp: bool = False,
    return_full: bool = True,
):
    results = singlebeam_to_input(
        path,
        DM,
        downsample,
        time_range,
        fit_DM,
        dm_range,
        spectrum_lim,
        fill_missing_time,
        diagnostic_plots,
        raise_missing_signal,
        full_time,
        snr_cutoff,
        co_dedisp,
        return_full,
    )
    if return_full:
        if len(results) == 14:
            res = {
                "data": results[0],
                "freq_id": results[1],
                "freq": results[2],
                "power": results[3],
                "valid_channels": results[4],
                "DM": results[5],
                "DM_err": results[6],
                "downsampling_factor": results[7],
                "profile": results[8],
                "t_res": results[9],
                "start * t_res": results[10],
                "full_power": results[11],
                "start": results[12],
                "end": results[13],
            }
        if len(results) == 12:
            res = {
                "data": results[0],
                "freq_id": results[1],
                "freq": results[2],
                "power": results[3],
                "valid_channels": results[4],
                "DM": results[5],
                "DM_err": results[6],
                "downsampling_factor": results[7],
                "profile": results[8],
                "t_res": results[9],
                "start * t_res": results[10],
                "full_power": results[11],
                "start": int(results[10] // results[9]),
                "end": int(results[10] // results[9] + len(results[8])),
            }
    else:
        res = {
            "profile": results[0],
            "t_res": results[1],
        }

    return res


def singlebeam_to_input(
    path: str,
    DM: float = None,
    downsample: int = None,
    time_range: tuple = None,
    fit_DM: bool = True,
    dm_range: int = 5,
    spectrum_lim: bool = False,
    fill_missing_time: bool = None,
    diagnostic_plots: bool = False,
    raise_missing_signal: bool = False,
    full_time: bool = False,
    snr_cutoff: int = 20,
    co_dedisp: bool = False,
    return_full: bool = True,
) -> tuple:
    '''
    Converts the singlebeam file to numpy format (can be used to extract power), cleans the data and calculates structure maximizing DM.

    Parameters
    ----------
    path : str
        Path to the singlebeam h5 file that contains baseband data.
    DM : float, optional
        If you already have the best DM and just need the power provide a DM value.
        IMPORTANT!! - Leave it as NONE and set fit_DM = True if you need a structure Maximizing DM.
    downsample : int, optional
        To downsample the data in addition to the downsampling in the singlebeam file.
        This downsampling factor is used in the DM correction, though if not enough a best downsample will be calculated.
    time_range : tuple, optional
        Bin numbers where to start and end the baseband data at downsampling 'downsample'.
    fit_DM : bool, optional
        If True, run DM_phase's structure maximising DM correction and apply it.
        If False and no DM provided will give S/N maximized DM
    dm_range: int, optional
        The dm_range upto which you want to search structure maximizing DM
    spectrum_lim : bool, optional
        Whether to cut out the frequency channels without strong signal (True) or not.
        Default is True.
    fill_missing_time : bool, optional
        If you want to force get_snr to apply the procedue to fill the triangular artefact with noise.
        Default is None (i.e. get_snr decides whether to fill the triangle or not).
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    raise_missing_signal: bool, optional
        If True, will show error if signal not found in data
        If False, will show warning if signal not found in data
    full_time: bool, optional
        If true, takes the entire time range of the file for fitting, useful when many sub-bursts distance apart
        also when you need more noise in the dataset for acf analysis.
    snr_cutoff: int, optional
        The signal to noise cutoff upto which it should search the best downsampling time
    co_dedisp: bool, optionak
        If True coherently dedisperse burst at the given DM or the structure maximizing DM
    return_full : bool, optional
        If True (default), returns data, freq_id, freq, power[...,start:end], valid_channels,
        DM, DM_err, downsampling_factor, profile, t_res, peak_times, start * t_res, power
        See in baseband_analysis.analysis.snr, the get_snr() function for more info
        If False, returns only pulse profile and time resolution
    Returns
    -------
    Tuple
        See description for return_full

    """

    '''
    data = BBData.from_file(path)
    event_id = path.split("/")[-2].split("_")[-1]
    fname = pathlib.Path(path)
    assert fname.exists(), f"No such file: {fname}"  # check that the file exists

    ##### Checking if baseband data was already de-smeared #######

    logger.info("Checking if baseband data was already de-smeared..")
    if "DM" in data["tiedbeam_baseband"].attrs.keys():
        logger.info("Data is de-smeared need to perform coherent dedispersion")
        if DM is not None:
            DM_coh = DM
            coherent_dedisp(data, DM_coh, write=True)
            tiedbeam_baseband_to_power(
                data, time_downsample_factor=1, dm=DM_coh, dedisperse=True
            )
        else:
            logger.info("Looking if there is already a DM_coherent in tiedbeam_power")
            try:
                DM_coh = data["tiedbeam_power"].attrs["DM_coherent"]
                logger.info("Found a DM value of {}".format(DM_coh))
                coherent_dedisp(data, DM_coh, write=True)
                tiedbeam_baseband_to_power(
                    data, time_downsample_factor=1, dm=DM_coh, dedisperse=True
                )
            except KeyError:
                logger.error(
                    "[singlebeam_to_input] No DM found please supply a DM value"
                )
    else:
        logger.warning("No DM in tiedbeam_baseband, so continuing on")

    ####### Check complete ###############

    ####### Finding the best downsample #######
    logger.info("---------Downsampling---------")
    logger.info(f"Finding the best downsample. The SNr cutoff used is: {snr_cutoff}")

    # first, getting the power at the inputted dsamp
    freq_id, freq, power, _, _, valid_channels, _, DM_dsamp, downsampling_factor = (
        get_snr(
            data,  ### Change here to DM_dsamp as it changes if DM was none
            DM=DM,
            downsample=downsample,
            fill_missing_time=fill_missing_time,
            # valid_channels = manual_valid_channels,
            diagnostic_plots=diagnostic_plots,
            spectrum_lim=spectrum_lim,
            refine_RFI=True,
            return_full=return_full,
            DM_range=None,
            raise_missing_signal=raise_missing_signal,
        )
    )

    # initial profile at initial inputted dsamp

    logger.info(f"Power shape: {power.shape}")
    profile, start, end = get_signal_time_range(
        power, downsampling_factor, time_range, full_time=True
    )

    downsample = get_best_downsamp(profile, downsampling_factor, snr_cutoff)
    logger.warning(
        "We will use {} as downsampling factor from now on".format(downsample)
    )

    # if downsampling factor changed, get a new power array
    if downsample != downsampling_factor:
        logger.warning(
            "New downsampling factor != original factor. Going to re-run get_snr as a result to get a new power array"
        )
        freq_id, freq, power, _, _, valid_channels, _, DM_dsamp, downsampling_factor = (
            get_snr(
                data,  ### Change here to DM_dsamp as it changes if DM was none
                DM=DM,
                downsample=downsample,
                fill_missing_time=fill_missing_time,
                diagnostic_plots=diagnostic_plots,
                # valid_channels = manual_valid_channels,
                spectrum_lim=spectrum_lim,
                return_full=return_full,
                refine_RFI=True,
                DM_range=None,
                raise_missing_signal=raise_missing_signal,
            )
        )

    # profile after any downsampling changes made
    logger.info(f"The total size of power is: {power.shape}")

    profile, start, end = get_signal_time_range(
        power, downsample, time_range, full_time=True
    )

    DM_err = 0

    # Fitting for DM if requested
    logger.info("---------DM DETERMINATION---------")
    if fit_DM:
        logger.info("Running get_snr() to get an estimate on S/N maximizing DM")
        dm_range_snr = dm_range
        freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = (
            get_snr(
                data,
                DM=DM,
                downsample=downsample,
                fill_missing_time=fill_missing_time,
                # valid_channels = manual_valid_channels,
                diagnostic_plots=False,
                spectrum_lim=spectrum_lim,
                return_full=True,
                refine_RFI=True,
                DM_range=dm_range_snr,
                raise_missing_signal=raise_missing_signal,
            )
        )
        logger.info("The S/N maximizing DM is {} pc/cc".format(DM))
        logger.info("Grabbing the burst profile")
        profile, start, end = get_signal_time_range(
            power, downsampling_factor, time_range, full_time
        )

        logger.info(f"The start and end time bin is ({start}, {end})")

        t_res = 2.56e-6 * downsampling_factor

        logger.info("Running structure maximizing DM script")
        DM_range = dm_range

        DM_corr, DM_err = get_structure_max_DM(
            power[..., start:end],
            freq,
            t_res=2.56e-6 * downsampling_factor,
            DM_range=dm_range,
        )

        DM = DM + DM_corr
        logger.info(
            "The structure maximizing DM is {} pc/cc +/- {} pc/cc".format(DM, DM_err)
        )
    else:
        logger.info(
            f"Not fitting for DM. Will use input DM of : {DM}. If you wanted to fit DM, set fit_DM = True.",
        )

    if co_dedisp:
        logger.info(
            "Coherently dedispersing the data at the given DM of {} pc/cc".format(DM)
        )
        DM_coh = DM
        coherent_dedisp(data, DM_coh, write=True)
        tiedbeam_baseband_to_power(
            data, time_downsample_factor=1, dm=DM_coh, dedisperse=True
        )

    logger.info(f"Running get_snr() again now at the final DM of: {DM}")

    freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM=DM,
        downsample=downsample,
        fill_missing_time=fill_missing_time,
        diagnostic_plots=True,
        spectrum_lim=spectrum_lim,
        # valid_channels = manual_valid_channels,
        return_full=True,
        DM_range=None,
        raise_missing_signal=raise_missing_signal,
    )

    profile, start, end = get_signal_time_range(
        power, downsampling_factor, time_range, full_time
    )

    t_res = 2.56e-6 * downsampling_factor

    logger.info(f"The start and end time bin is ({start}, {end})")

    if diagnostic_plots:
        logger.info("Plotting a waterfall of the power")
        plt.clf()
        plot_waterfall(power[..., start:end], t_res, freq)
        plt.show()

    if return_full:
        return (
            data,
            freq_id,
            freq,
            power[..., start:end],
            valid_channels,
            DM,
            DM_err,
            downsampling_factor,
            profile,
            t_res,
            start * t_res,
            power,
            start,
            end,
        )
    else:
        return profile, t_res


def get_peaks(
    profile: np.ndarray,
    t_res: float = 2.56e-6,
    ds: int = 1,
    diagnostic_plots: bool = True,
    l: int = None,
    n_peaks: int = None,
    peak_times: list = None,
    event_id: str = "",
    best_frac: float = None,
    f: int = None,
) -> tuple:
    """
    Find the peaks in the given dataset

    Paramteres
    ---------
    profile: np.ndarray
        The profile of the burst
    t_res : float
        The time resolution of the profile
    diagnostic_plot: bool
        if True will show the diagnostic plots generated by count component algorithm
    l : float
        The prominence cut-off, the lower it is the more peaks will the algorithm search
    n_peaks : int
        If peaks are overestimated by the algorithm you can provide the number of peaks you think present in the dataset
    peak_times: list
        If you already have the time bins of each peak at the given time-resolution, you can provide them as input
    event_id : str
        If you want to show event id on diagnostic plots
    best_frac : float, optional
        Lowess smoothing fraction, e.g. see
        https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35
    f : int
        np.ceil(7/f)+1 is the minimum distance between peaks

    Returns
    -------
    peaks : np.ndarray
        Returns the time bins of each peak in the data, multiply by t_res to get rough arrival times of each sub-burst
    """

    if peak_times is None:
        peaks, heights = count_components(
            profile,
            t_res,
            ds=ds,
            event_id=event_id,
            diagnostic_plots=diagnostic_plots,
            l=l,
            n_peaks=n_peaks,
            best_frac=best_frac,
            f=f,
        )
    else:
        # if peaks already given, just set the heights to ones
        peaks = (np.array(peak_times) / t_res).astype(int)
        heights = np.ones(len(peaks))

    return peaks, heights


def initial_guess_LS(
    profile: np.ndarray,
    peaks: np.int64,
    t_res: float,
    heights: np.int64 = None,
    min_width: float = None,
    diagnostic_plots: bool = True,
    show_plots: bool = True,
) -> tuple:
    """
    Get Initial guess for the fits using least square

    Parameters
    ----------

    profile: np.ndarray
        The profile of the burst
    peaks : np.int64
        Time bins of peaks
    t_res : float
        The time resolution of the profile
    heights : np.int64 optional
        Height of each peak, can be used as initial guess for LS fit
    min_width : float, optional
        Minimum width of burst you want to search, this is also helpful if LS if not latching on to all the components
    diagnostic_plot : bool, optional
        If true will show diagnostic plots generated by the list square fitting routine

    Returns
    -------
    The fit model, best fit parameters, best fit parameters error, covariance matrix
    """

    xvals = t_res * np.arange(len(profile))
    peak_times = peaks * t_res

    ###### Setting up sigmas #######
    lhs = profile[0 : np.where(profile == max(profile))[0][0]]
    if min_width is not None:
        sigma_guess = min_width
    else:
        sigma_guess = t_res * len(
            lhs[lhs > max(profile) / 2]
        )  #### this find the approx. width of the brightest sub-burst

    sigmas = np.full(len(peaks), sigma_guess)
    sigmas_ub = np.full(
        len(peaks), t_res * len(profile)
    )  #### Burst can't be wider than the file
    sigmas_lb = np.zeros(len(peaks))

    ###### Setting up the amplitudes #####
    if heights is not None:
        amps = (
            heights * t_res * len(lhs[lhs > max(profile) / 2]) * 5
        )  #### This arbitraty seems to work best don't know why?
        amps_ub = np.full(len(peaks), np.inf)
        amps_lb = np.zeros(len(peaks))
    else:
        amps = np.full(len(peaks), np.max(profile) * sigma_guess)
        amps_ub = amps + np.max(profile) / 10
        amps_lb = np.zeros(len(peaks))

    ###### Setting up the mus #######
    mus = peak_times
    mus_ub = peak_times + sigma_guess
    mus_lb = peak_times - sigma_guess

    mus_lb[mus_lb < 0] = 0

    ####### Putting it in format used by curve_fit ######

    ICs = []  ##### initial guesses ####
    ubs = []  ##### upper bounds #######
    lbs = []  ##### lower bounds #######

    for i in range(len(peaks)):
        ICs.extend([amps[i], mus[i], sigmas[i]])
        ubs.extend([amps_ub[i], mus_ub[i], sigmas_ub[i]])
        lbs.extend([amps_lb[i], mus_lb[i], sigmas_lb[i]])

    ##### Adding scattering initial guess #######

    ICs.extend([2 * sigma_guess])
    ubs.extend([np.inf])
    lbs.extend([0])

    # Force the initial guess in bounds:
    for index, ic in enumerate(ICs):
        temp = max(ic, lbs[index])
        temp = min(temp, ubs[index])
        ICs[index] = temp

    print("Initial guesses", ICs)
    print("\nUpper bounds", ubs)
    print("\nlower bounds", lbs)

    f = sum_emg

    x = xvals
    y = profile
    if show_plots:
        logger.info(
            "Plotting the profile in blue, the initial fit guess in orange, and the peak in red"
        )
        plt.title("Profile, initial fit for data (pre LS)")
        plt.plot(x, y)
        plt.xlabel("Time (s)")
        plt.ylabel("S/N")
        for p in peak_times:
            plt.axvline(p, color="r")
        plt.plot(x, f(x, ICs), color="orange")
        plt.show()
    logger.info("Now doing a curve_fit for the initial fit function to the data")
    logger.info(f"Using bounds: {(lbs, ubs)}")
    popt, pcov = curve_fit(f, x, y, p0=ICs, bounds=(lbs, ubs), maxfev=100000000)
    if show_plots:
        logger.info("Now plotting the results of the curve fit")
        plt.title("Profile, fit for data (post LS)")
        plt.plot(x, y)
        plt.plot(x, f(x, popt), color="orange")
        plt.xlabel("Time (s)")
        plt.ylabel("S/N")
        plt.show()
    if diagnostic_plots:
        logger.info("Curve_fit params: " + str(popt))
        show_fit(profile, xvals, popt, t_res, event_id="X", m="emg")

    return f(xvals, popt), popt, np.sqrt(np.diag(pcov)), pcov


# def initial_guess_mcmc(profile, best_params, res, event_id, peaks, logl = 0, ):

#     tau, flat_samples, final_pars, final_errs = mcmc(popt, profile, xvals, res,event_id = event_id,
#                                                      m_peaks = peaks_to_fit, logl = logl, nwalkers=nwalkers,
#                                                      nchain=nchain, ncores=ncores, plot=True, show_chains=show_chains)


def initial_guess_spec(
    power, freq, f_res, event_id: str = " ", diagnostic_plots: bool = True
):

    spectrum = get_spectrum(
        power
    )  #### This will change, you will have to find limit for each peak
    spect_peaks = spect_count_components(
        spectrum, freq, event_id=event_id, diagnostic_plots=diagnostic_plots
    )

    f = rpl
    ICs = [max(spectrum), 0.0, 0]
    ubs = [np.inf, np.inf, np.inf]
    lbs = [0, -np.inf, -np.inf]

    x = freq
    y = spectrum
    logger.info(
        "Plotting the profile in blue, the initial fit guess in orange, and the peak in red"
    )
    plt.title("Profile, initial fit for data (pre LS)")
    plt.plot(x, y)
    plt.xlabel("Freq (MHz)")
    plt.ylabel("S/N")
    # for p in spect_peaks:
    #     plt.axvline(p, color = 'r')
    plt.plot(x, f(x, ICs), color="orange")
    plt.show()
    logger.info("Now doing a curve_fit for the initial fit function to the data")
    popt, pcov = curve_fit(f, x, y, p0=ICs, bounds=(lbs, ubs), maxfev=100000000)
    logger.info("Now plotting the results of the curve fit")
    plt.title("Spectrum, fit for data (post LS)")
    plt.plot(x, y)
    plt.plot(x, f(x, popt), color="orange")
    plt.xlabel("Freq MHz")
    plt.ylabel("S/N")
    plt.show()
    if diagnostic_plots:
        logger.info("Curve_fit params: " + str(popt))
        show_fit(y, x, popt, f_res, event_id="X", m="rpl")

    return f(x, popt), popt, np.sqrt(np.diag(pcov)), pcov


def initial_guess_mcmc(
    best_params_LS: np.ndarray,
    profile: np.ndarray,
    t_res: float,
    event: int,
    peaks: np.ndarray,
    logl: int = 0,
    nwalkers: int = 50,
    nchain: int = 10000,
    ncores: int = 4,
    show_chains: bool = False,
) -> tuple:
    """
    Performs MCMC sampling on the burst profile (exponentially modified gaussian) and return the best fit parameters

    Parameters
    ----------
    popt : np.ndarray
        LS best fit parameters, used as initial condition for the MCMC fit.
    profile : np.ndarray
        The 1D data to be fit (either pulse profile or spectrum)
    res : float
        Resolution (time for EMG)
    event: int
        Event id of the FRB
    peaks : np.ndarray
        Time bins of peaks in the profile, can be found using peak finding fuction
    nwalkers : int, optional
        Number of walkers in MCMC process, default is 50
    nchain : int, optional
        Length of chain in MCMC process, default is 10k, use more if mcmc not converging, ideal 100k+
    ncores : int, optional
        Number cores that are used for the MCMC fitting,
        default is 4, in general, do not use more than ~15, depends on how many cores your machine has.
    logl : int, optional
        Two possible values:
        0 : default wide flat prior for params
        1 : narrower flat prior (useful for when the MCMC has trouble converging)
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    plot : bool, optional
        If True, shows corner plot of the fit. Default is True.
    Returns
    -------
    best fit values, errors on best fit values
    """

    popt = best_params_LS
    xvals = t_res * np.arange(len(profile))
    res = t_res
    event_id = str(event)
    peaks_to_fit = peaks * t_res

    assert logl in [0, 1], "logl can only take values 0 or 1"

    tau, flat_samples, final_pars, final_errs = mcmc(
        popt,
        profile,
        xvals,
        res,
        event_id=event_id,
        m_peaks=peaks_to_fit,
        logl=logl,
        nwalkers=nwalkers,
        nchain=nchain,
        ncores=ncores,
        plot=True,
        show_chains=show_chains,
    )

    return_pars = np.array(final_pars).flatten()
    return_errs = np.array(final_errs).flatten()

    return return_pars, return_errs


def create_fitburst_input(
    power: np.ndarray,
    freq_id: np.int64,
    freq: np.float64,
    DM: float,
    downsampling_factor: int,
    t_res: float,
    peaks: np.int64,
    event_id: str,
    ref_freq=600.0,
    use_default: bool = False,
    guess_params: np.ndarray = None,
    spec_params: np.ndarray = None,
    amp: tuple = None,
    gen_no_scat: bool = False,
    output: str = None,
):
    """
    Creates input file needed in fitburst format.

    Parameters
    ----------
    power : np.ndarray
        Stokes I power of the burst extracted from the singlebeam file (its an array of frequency and time)
    freq_id : np.int64
        frequency id of each frequency channel
    freq : np.float64
        each frequency inside the power
    DM : float
        The dispersion measure at which power is dedispersed to
    downsampling_factor : int
        The factor by which power has been downsampled in time
    t_res : float
        Time resolution of the power
    peaks : np.int64
        Time bins of all the peaks
    event_id : str
        Event id of the burst, used to generate output file
    ref_freq : float, optional
        The reference frequency used by fitburst to estimate, scattering, width etc.
    use_default : bool, optional
        If true will use default values as initial guesses
    guess_params : np.ndarray, optional
        The initial guess parameters generated by the LS function. This helps fitburst to converge faster and provide a correct fit
    spec_params : np.ndarray, optional
        The initial guess parameters for spectral index and spectral running
    output : str, oprional
        Output file name you want, default is fitburst_input_<event_id>.npz

    Returns
    -------
     A npz file with fitburst input format
    """

    data_full = np.zeros([1024, power.shape[-1]])
    data_full[1023 - freq_id] = power
    bad = []
    min_f = FREQ_BOTTOM_MHZ
    logger.info(f"Minimum frequency is {min_f}")

    logger.info(peaks)
    peaks = np.sort(peaks)
    arr_times = peaks * t_res
    logger.info(arr_times)

    #### default #####
    if use_default or guess_params is None:
        logger.warning("Using default parameters")
        amp = [0.0] * len(peaks)
        widths = [0.001] * len(peaks)
        scattering_t = 0.0
        spec_index = [0] * len(peaks)
        spec_run = [0] * len(peaks)

    else:
        logger.info("Using guess parameters")

        params, scattering_t = np.array(guess_params[:-1]), guess_params[-1]
        n = int(len(params) / 3)
        params = params.reshape((n, 3))
        mus = params[..., 1]
        widths = params[..., 2]

        if amp is None:
            amp = [-0.5] * len(peaks)
        else:
            amp = amp
        # amp = list(np.log10(params[...,0]))
        logger.info(amp)
        widths = list(widths)
        if gen_no_scat:
            logger.info("Generating with zero scattering")
            scattering_t = 0
        else:
            scattering_t = scattering_t

        if spec_params is not None:
            spec_index = [-spec_params[1]]
            spec_run = [spec_params[2]]
        else:
            spec_index = [0] * len(peaks)
            spec_run = [0] * len(peaks)

    logger.info(len(freq_id))
    for i in range(1024):
        if i not in freq_id:
            bad.append(1023 - i)

    logger.info(len(bad))

    # Metadata
    metadata = {
        "bad_chans": bad,  # a Python list of indices corresponding to frequency channels to zero-weight
        "freqs_bin0": min_f,  # a floating-point scalar indicating the value of frequency bin at index 0, in MHz
        "is_dedispersed": True,  # a boolean indicating if spectrum is already dedispersed (True) or not (False)
        "num_freq": data_full.shape[
            0
        ],  # an integer scalar indicating the number of frequency bins/channels
        "num_time": data_full.shape[
            -1
        ],  # an integer scalar indicating the number of time bins
        "times_bin0": 0.0,  # a floating-point scalar indicating the value of time bin at index 0, in seconds
        "res_freq": 400.0
        / 1024.0,  # a floating-point scalar indicating the frequency resolution, in MHz
        "res_time": t_res,  # a floating-point scalar indicating the time resolution, in seconds
    }

    # Output has shape burst_parameters, num components
    # Burst parameters
    burst_parameters = {
        "amplitude": amp,
        "arrival_time": list(
            arr_times
        ),  # a list containing the arrival times, in seconds
        "burst_width": widths,  # a list containing the temporal widths, in seconds
        "dm": [DM]
        * len(
            peaks
        ),  # a list containing the dispersion measures (DM), in parsec per cubic centimeter
        "dm_index": [
            -2 for i in range(len(peaks))
        ],  # a list containing the exponents of frequency dependence in DM delay
        "ref_freq": [
            ref_freq for i in range(len(peaks))
        ],  # a list containing the reference frequencies for arrival-time and power-law parameter estimates, in MHz (held fixed)
        "scattering_index": [],  # a list containing the exponents of frequency dependence in scatter-broadening
        "scattering_timescale": [
            scattering_t for i in range(len(peaks))
        ],  # a list containing the scattering timescales, in seconds
        "spectral_index": spec_index,  # a list containing the power-law spectral indices
        "spectral_running": spec_run,  # a list containing the power-law spectral running
    }

    if output is not None:
        logger.info("Using the output name")
        fname = output
    else:
        logger.warning("Using the default name")
        fname = "fitburst_input_" + event_id

    np.savez(
        fname + ".npz",
        data_full=data_full,
        metadata=metadata,
        burst_parameters=burst_parameters,
    )
    return


###### This function has to be looked at #######
def add_all_channels(power, freq_id):
    data_full = np.zeros([1024, power.shape[-1]])
    data_full[1023 - freq_id] = power
    f = np.linspace(FREQ_BOTTOM_MHZ, FREQ_TOP_MHZ, 1024)
    return data_full, f
