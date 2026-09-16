import os, json
from utils import load_parser_from_json
from copy import deepcopy

from fitburst.analysis.peak_finder import FindPeak

# from fitburst.analysis.fitter import LSFitter
from fitburstfitter import LSFitter  # Single line change to add multiprocessing
from fitburstmodel import FitBurstModel
import fitburst.utilities as ut

import oscfar as ocf

np = ocf.np
# Get the directory where the current file is located:
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the parser from the JSON file
json_file = f"{script_dir}/parser_configs/fitburst_parser_config.json"
parser = load_parser_from_json(json_file)

args = parser.parse_args()

input_file = args.file
amplitude = args.amplitude
arrival_time = args.arrival_time
dm = args.dm
dm_index = args.dm_index
factor_freq_downsample = args.factor_freq_downsample
factor_time_downsample = args.factor_time_downsample
factor_freq_upsample = args.factor_freq_upsample
factor_time_upsample = args.factor_time_upsample
is_folded = args.is_folded
num_iterations = args.num_iterations
parameters_to_fit = args.parameters_to_fit
parameters_to_fix = args.parameters_to_fix
peakfind_rms = args.peakfind_rms
peakfind_dist = args.peakfind_dist
preprocess_data = args.preprocess_data
ref_freq = args.ref_freq
remove_dispersion_smearing = args.remove_dispersion_smearing
use_outfile_substring = args.use_outfile_substring
scattering_timescale = args.scattering_timescale
scintillation = args.scintillation
spectral_index = args.spectral_index
spectral_running = args.spectral_running
solution_file = args.solution_file
variance_range = args.variance_range
verbose = args.verbose
weight_range = args.weight_range
width = args.width
window = args.window

parameters_to_fix += [
    "dm_index",
    "scattering_index",
    "scattering_timescale",
]

for current_fit_parameter in parameters_to_fit:
    if current_fit_parameter in parameters_to_fix:
        parameters_to_fix.remove(current_fit_parameter)

# Check for a solution file:
existing_results = None

if solution_file is not None and os.path.isfile(solution_file):
    try:
        with open(solution_file, "r") as solution_handle:
            existing_results = json.load(solution_handle)

    except:
        print(
            "WARNING: input solution cannot be read; using basic initial parameters..."
        )

else:
    print("INFO: no solution file found or provided; proceeding with fit...")

outfile_substring = ""

if use_outfile_substring:
    elems = os.path.basename(input_file).split(".")
    outfile_substring = "_" + ".".join(elems[0 : len(elems) - 1])

data = ocf.npz_writer(input_file)
print(f"INFO: there are {data.num_freq} frequencies and {data.num_time} time samples.")

data.good_freq = np.sum(data.data_weights, axis=1) != 0.0
data.good_freq = np.sum(data.data_full, axis=1) != 0.0

# just to be sure, loop over data and ensure channels aren't "bad".
for idx_freq in range(data.num_freq):
    if data.good_freq[idx_freq]:
        if data.data_full[idx_freq, :].min() == data.data_full[idx_freq, :].max():
            print(
                f"ERROR: bad data value of {data.data_full[idx_freq, :].min()} in channel {idx_freq}!"
            )
            data.good_freq[idx_freq] = False


if preprocess_data:
    data.preprocess_data(normalize_variance=True, variance_range=variance_range)

print(f"INFO: there are {data.good_freq.sum()} good frequencies...")

# now downsample after preprocessing, if desired.
data.downsample(factor_freq_downsample, factor_time_downsample)

# check if any initial guesses are missing, and overload 'basic' guess value if so.
initial_parameters = data.burst_parameters
num_components = len(initial_parameters["dm"])
basic_parameters = {
    "amplitude": [-2.0],
    "arrival_time": [np.mean(data.times)],
    "burst_width": [0.05],
    "dm": [0.0],
    "dm_index": [-2.0],
    "ref_freq": [np.min(data.freqs)],
    "scattering_index": [-4.0],
    "spectral_index": [0.0],
    "spectral_running": [0.0],
}

for current_parameter in initial_parameters.keys():
    current_list = initial_parameters[current_parameter]

    if len(current_list) == 0:
        print(
            f"WARNING: parameter '{current_parameter}' has no value in data file, overloading a basic guess..."
        )
        initial_parameters[current_parameter] = (
            basic_parameters[current_parameter] * num_components
        )

# now see if any parameters are missing in the dictionary.
for current_parameter in basic_parameters.keys():
    if current_parameter not in initial_parameters:
        initial_parameters[current_parameter] = (
            basic_parameters[current_parameter] * num_components
        )

current_parameters = deepcopy(initial_parameters)

# update DM value to use ("full" or DM offset) for dedispersion if
# input data are already dedispersed or not.
dm_incoherent = initial_parameters["dm"][0]

if data.is_dedispersed:
    print("INFO: input data cube is already dedispersed!")
    print("INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...")
    current_parameters["dm"] = [0.0] * len(initial_parameters["dm"])

if not remove_dispersion_smearing:
    dm_incoherent = 0.0

# if an existing solution is supplied in a JSON file, then read it or use basic guesses.
if existing_results is not None:
    current_parameters = existing_results["model_parameters"]
    num_components = len(current_parameters["dm"])

# if values are supplied at command line, then overload those here.
if amplitude is not None:
    current_parameters["amplitude"] = amplitude

if arrival_time is not None:
    current_parameters["arrival_time"] = arrival_time

if dm is not None:
    current_parameters["dm"] = dm

if dm is not None:
    current_parameters["dm_index"] = dm_index

if scattering_timescale is not None:
    current_parameters["scattering_timescale"] = scattering_timescale

if spectral_index is not None:
    current_parameters["spectral_index"] = spectral_index

if spectral_running is not None:
    current_parameters["spectral_running"] = spectral_running

if width is not None:
    current_parameters["burst_width"] = width

# now replace ref_freq value, if desired.
if ref_freq is not None:
    current_parameters["ref_freq"] = [ref_freq] * num_components

# print parameter info if desired.
if verbose:
    print(f"INFO: initial guess for {len(current_parameters['dm'])}-component model:")

    for current_parameter_label in current_parameters.keys():
        current_list = current_parameters[current_parameter_label]
        print(f"    * {current_parameter_label}: {current_list}")

# now compute dedisperse matrix for data, given initial DM (or DM offset),
# and grab windowed data.
print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    ref_freq=initial_parameters["ref_freq"][0],
    dm_offset=0.0,
)

data_windowed = data.data_full
times_windowed = data.times

if window is not None:
    data_windowed, times_windowed = data.window_data(
        current_parameters["arrival_time"][0], window=window
    )

# before instantiating model, run peak-finding algorithm if desired.
if peakfind_rms is not None:
    print("INFO: running FindPeak to isolate burst components...")
    peaks = FindPeak(data_windowed, times_windowed, data.freqs, rms=peakfind_rms)
    peaks.find_peak(distance=peakfind_dist)
    current_parameters = peaks.get_parameters_dict(current_parameters)
    num_components = len(current_parameters["dm"])

# now create initial model.
print("INFO: initializing model")
model = FitBurstModel(
    data.freqs,
    times_windowed,
    dm_incoherent=dm_incoherent,
    factor_freq_upsample=factor_freq_upsample,
    factor_time_upsample=factor_time_upsample,
    num_components=num_components,
    is_dedispersed=data.is_dedispersed,
    is_folded=is_folded,
    scintillation=scintillation,
    verbose=verbose,
)
model.update_parameters(current_parameters)

for current_iteration in range(num_iterations):
    print(f"INFO: iteration {current_iteration + 1} of {num_iterations}")

    # now fit the model to the data.
    fitter = LSFitter(
        data_windowed,
        model,
        data.good_freq,
        weighted_fit=True,
        weight_range=weight_range,
    )
    fitter.fix_parameter(parameters_to_fix)
    fitter.fit(exact_jacobian=True)

    if fitter.results.success:
        bestfit_params = fitter.fit_statistics["bestfit_parameters"]
        model.update_parameters(bestfit_params)
        current_params = model.get_parameters_dict()

        if not any([x == "dm" for x in parameters_to_fix]):
            current_params["dm"] = [x for x in bestfit_params["dm"] * num_components]

        if "scattering_timescale" not in parameters_to_fix:
            current_params["scattering_timescale"] = [
                x for x in bestfit_params["scattering_timescale"] * num_components
            ]

        if current_iteration == (num_iterations - 1):
            bestfit_parameters = fitter.fit_statistics["bestfit_parameters"]
            bestfit_uncertainties = fitter.fit_statistics["bestfit_uncertainties"]

            if verbose:
                print(
                    f"INFO: best-fit estimate for {len(current_parameters['dm'])}-component model:"
                )

                for current_parameter_label in bestfit_parameters.keys():
                    current_list = bestfit_parameters[current_parameter_label]
                    current_uncertainties = bestfit_uncertainties[
                        current_parameter_label
                    ]
                    print(
                        f"    * {current_parameter_label}: {current_list} +/- {current_uncertainties}"
                    )

                print("INFO: ratio of hessian matrix (approximate / exact):")
                print(fitter.hessian / fitter.hessian_approx)

            bestfit_model = model.compute_model(data=data_windowed)
            bestfit_residuals = data_windowed - bestfit_model

            data_grouped = ut.plotting.compute_downsampled_data(
                times_windowed,
                data.freqs,
                data_windowed,
                data.good_freq,
                spectrum_model=bestfit_model,
                factor_freq=factor_freq_downsample,
                factor_time=factor_time_downsample,
            )

            ut.plotting.plot_summary_triptych(
                data_grouped,
                output_name=f"summary_plot{outfile_substring}.png",
                show=False,
            )

            print(f"Saving summary plot to 'summary_plot{outfile_substring}.png'...")

            with open(f"results_fitburst{outfile_substring}.json", "w") as out:
                json.dump(
                    {
                        "initial_dm": initial_parameters["dm"][0],
                        "initial_time": data.times_bin0,
                        "model_parameters": current_params,
                        "fit_statistics": fitter.fit_statistics,
                        "fit_logistics": {
                            "weight_range": weight_range,
                        },
                    },
                    out,
                    indent=4,
                )

            print(f"Saving results to 'results_fitburst{outfile_substring}.json'...")
