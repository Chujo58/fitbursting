"""Run fitburst for one event over selected NPZ inputs.

Supports four execution modes: no-scattering, scattering, scintillation, and
scattering+scintillation. Outputs are moved into per-mode subdirectories under
`RESULTS_PATH/results_<eid>/`.
"""

import glob, subprocess, sys, os, argparse, shutil, oscfar  # noqa: E401
from datetime import datetime, UTC
from paths import TEMPORARY_NPZ_PATH, RESULTS_PATH, NPZ_SAVE_PATH
from sheets import update_row, get_info


parser = argparse.ArgumentParser(
    description="Run fitburst on temporary npz files for an event ID"
)
parser.add_argument("eid", type=str, help="The event ID to process")
parser.add_argument(
    "--scattering",
    action="store_true",
    help="Enable scattering_timescale fit (default)",
)
parser.add_argument(
    "--scintillation", action="store_true", help="Enable scintillation fit"
)
# Allow for the user to pick which n component to run on, if not specified, run on all npz files for the event ID
parser.add_argument(
    "--n_component",
    type=int,
    default=-1,
    help="The n component to run fitburst on (e.g., 1 for n1, 2 for n2). If not specified, runs on all npz files for the event ID. Defaults to -1, the Alice-Ketan npz file",
)
parser.add_argument(
    "--iterations", type=int, default=1, help="Number of fitburst iterations"
)
args = parser.parse_args()

eid = args.eid
scattering = args.scattering
scintillation = args.scintillation
message = "with scattering" if scattering else "without scattering"
message += " and with scintillation" if scintillation else " and without scintillation"
print(f"Running event ID: {eid} {message}")

update_row(eid, {"Status": "Running fitburst"})

env = os.environ.copy()

if args.n_component:
    print(f"Running fitburst for n{args.n_component} component.")
    if args.n_component < 0:
        # Using the default file in NPZ_SAVE_PATH
        print("Using the general NPZ file.")
        npz_files = glob.glob(f"{NPZ_SAVE_PATH}/{eid}.npz")
    else:
        npz_files = glob.glob(f"{TEMPORARY_NPZ_PATH}/{eid}_n{args.n_component}.npz")
else:
    npz_files = glob.glob(f"{TEMPORARY_NPZ_PATH}/{eid}_n*.npz")

ran_into_error = 0
total_files = len(npz_files)
print(f"Found {total_files} npz files to process for event ID: {eid}")

# Before updating, check if the event is part of the priority list:
try:
    priority_info = get_info(eid, "Prioritized Events")
except KeyError:
    priority_info = None


# If part of priority list, double check scattering run status
def isScatteringFitEmpty(priority_info, scattering_enabled=False):
    """Return True when prerequisite prioritized-sheet status cell is empty."""
    return (
        priority_info.get(
            "Run w/ scattering" if scattering_enabled else "Run w/o scattering"
        )
        == ""
    )


def isNoScatteringFitInvalid(priority_info: dict):
    info: str = priority_info.get("Run w/o scattering")
    return "error" in info


def isScatteringFitCompleted(priority_info, scattering_enabled=False):
    """Return True when prerequisite status starts with 'Completed'."""
    return priority_info.get(
        "Run w/ scattering" if scattering_enabled else "Run w/o scattering"
    ).startswith("Completed")


def isVerifierEmpty(priority_info, round=1):
    """Return True if verifier assignment is missing for prioritized events."""
    return not priority_info.get("Verifier" if round == 1 else "Verifier R2")


def validateEvent(eid, priority_info, scattering_enabled, scintillation_enabled):
    """Validate prioritized-event prerequisites before scintillation runs."""
    message = "scattering fit" if scattering_enabled else "non-scattering fit"
    if isNoScatteringFitInvalid(priority_info) and not scintillation_enabled:
        print(
            f"Event ID {eid} has not completed non-scattering fit. Aborting scattering fit."
        )
        exit(1)

    if (
        isScatteringFitEmpty(priority_info, scattering_enabled)
        and scintillation_enabled
    ):
        print(
            f"Event ID {eid} has not completed {message}. Aborting scintillation fit."
        )
        exit(1)

    if isVerifierEmpty(priority_info) and scintillation_enabled:
        print(
            f"Event ID {eid} is prioritized but has no verifier assigned. Aborting scintillation fit."
        )
        exit(1)

    if isScatteringFitCompleted(priority_info, scattering_enabled):
        print(
            f"Event ID {eid} has completed {message}. Proceeding with scintillation fit."
        )


if priority_info and (scintillation or scattering):
    validateEvent(eid, priority_info, scattering, scintillation)


def move(eid, file_strip, fit_type="scat-scint"):
    """Move fitburst artifacts into their mode-specific result directory."""
    dest_dir = f"{RESULTS_PATH}/results_{eid}/{fit_type}"
    os.makedirs(dest_dir, exist_ok=True)

    files_to_move = [
        f"results_fitburst_{file_strip}.json",
        f"summary_plot_{file_strip}.png",
    ]

    for filename in files_to_move:
        src = os.path.abspath(filename)
        dst = f"{dest_dir}/{filename}"

        if not os.path.exists(src):
            print(f"Warning: {src} not found, skipping.")
            continue

        # Remove destination if it exists
        if os.path.exists(dst):
            try:
                os.remove(dst)
                print(f"Removed existing {dst}")
            except Exception as e:
                print(f"Cannot remove existing {dst}: {e}")
                continue

        # Use copy() not copy2() - don't try to preserve metadata
        try:
            shutil.copy(src, dst)  # Changed from copy2
            print(f"✓ Copied {src} -> {dst}")
        except Exception as e:
            print(f"✗ Copy failed: {e}")
            continue

        # Delete source
        try:
            os.remove(src)
            print(f"✓ Removed source {src}")
        except Exception as e:
            print(f"✗ Cannot remove source {src}: {e}")
            print(f"  File successfully copied to {dst}")


def validate_file(file_path):
    """Inspect the npz file to make sure it has non-empty arrays before running fitburst."""
    reader = oscfar.npz_reader(file_path)
    # Check the burst_parameters first:
    burst_parameters: dict = reader.burst_parameters
    if not burst_parameters:
        print(f"Error: 'burst_parameters' is empty in {file_path}. Skipping this file.")
        return False
    # Check each array in burst_parameters to make sure they are not empty:
    for key, value in burst_parameters.items():
        if isinstance(value, (list, tuple, oscfar.np.ndarray)) and len(value) == 0:
            print(
                f"Error: 'burst_parameters[{key}]' is empty in {file_path}. Skipping this file."
            )
            return False
    # Check the data:
    data = reader.data_full
    if data is None or len(data) == 0:
        print(f"Error: 'data_full' is empty in {file_path}. Skipping this file.")
        return False

    return True


def get_component_count_and_scattering(file):
    reader = oscfar.npz_reader(file)
    count = len(reader.burst_parameters["arrival_time"])
    scattering = reader.burst_parameters["scattering_timescale"]

    return count, scattering


def generate_scat_arg(count, value: list = None):
    if value is None:
        def_scat = ["0"] * count
    else:
        def_scat = value
    arg_out = "--scattering_timescale "
    scat_out = " ".join(map(str, def_scat))

    return arg_out + scat_out


for npz_file in npz_files:
    # Validate the npz file before running fitburst. If validation fails during Alice-Ketan mode (args.n_component == -1), crash the code.
    if not validate_file(npz_file):
        if args.n_component == -1:
            print(
                f"Validation failed for {npz_file} in Alice-Ketan mode. Exiting with error."
            )
            exit(1)
        else:
            print(f"Validation failed for {npz_file}. Skipping this file.")
            ran_into_error += 1
            continue
    # Maybe add some sort of try catch for exception with the subprocess.run -> mostly to catch some of the SingularMatrix errors from fitburst or fit didn't succeed stuff.
    try:
        file_strip = npz_file.strip(".npz").split("/")[-1]
        comp_count, scattering_values = get_component_count_and_scattering(npz_file)
        if scattering:
            scat_arg = generate_scat_arg(comp_count, scattering_values)
        else:
            scat_arg = generate_scat_arg(comp_count)

        # Run fitburst command
        if scintillation and scattering:
            # For scintillation and scattering, use the solution from the scattering fit
            solution = (
                f"{RESULTS_PATH}/results_{eid}/scat/results_fitburst_{file_strip}.json"
            )
            subprocess.run(
                f"{sys.executable} fitburst/basecat2_fitburst.py {npz_file} --verbose --outfile --scintillation --fit scattering_timescale --solution {solution} {scat_arg} --iterations {args.iterations}",
                shell=True,
                check=True,
                env=env,
            )
            move(eid, file_strip, fit_type="scat-scint")
        elif scintillation and not scattering:
            # For scintillation, use the solution from the no-scattering fit
            solution = f"{RESULTS_PATH}/results_{eid}/noscat/results_fitburst_{file_strip}.json"
            subprocess.run(
                f"{sys.executable} fitburst/basecat2_fitburst.py {npz_file} --verbose --outfile --scintillation --solution {solution} {scat_arg} --iterations {args.iterations}",
                shell=True,
                check=True,
                env=env,
            )
            move(eid, file_strip, fit_type="scint")

        elif scattering and not scintillation:
            # For scattering only, feed the no-scattering fit:
            solution = f"{RESULTS_PATH}/results_{eid}/noscat/results_fitburst_{file_strip}.json"
            subprocess.run(
                f"{sys.executable} fitburst/basecat2_fitburst.py {npz_file} --verbose --outfile --fit scattering_timescale --solution {solution} {scat_arg} --iterations {args.iterations}",
                shell=True,
                check=True,
                env=env,
            )
            move(eid, file_strip, fit_type="scat")
        else:
            # The default first run, no scattering, no scintillation, nothing.
            subprocess.run(
                f"{sys.executable} fitburst/basecat2_fitburst.py {npz_file} --verbose --outfile {scat_arg} --iterations {args.iterations}",
                shell=True,
                check=True,
                env=env,
            )
            move(eid, file_strip, fit_type="noscat")

    except Exception as e:
        print(f"Error processing {npz_file}: {e}")
        ran_into_error += 1


def fitburst_info(errornum):
    """Format completion status string from fit failure count."""
    return (
        "Completed"
        if errornum == 0
        else f"Completed with {errornum}/{total_files} errors"
    )


if not priority_info:
    update_row(
        eid,
        {"Status": fitburst_info(ran_into_error)},
    )

    print(f"Completed fitburst for event ID: {eid} with {ran_into_error} errors.")
    print(eid, ran_into_error)

else:
    # Only scattering
    if scattering and not scintillation:
        update_row(
            eid,
            {
                "Run w/ scattering": fitburst_info(ran_into_error),
            },
            sheet_tab="Prioritized Events",
        )
    # Only scintillation
    elif scintillation and not scattering:
        update_row(
            eid,
            {
                "Run w/ scintillation & w/o scattering": fitburst_info(ran_into_error),
            },
            sheet_tab="Prioritized Events",
        )
    # Both scattering and scintillation
    elif scattering and scintillation:
        update_row(
            eid,
            {
                "Run w/ scintillation & scattering": fitburst_info(ran_into_error),
            },
            sheet_tab="Prioritized Events",
        )
    # Neither scattering nor scintillation
    else:
        update_row(
            eid,
            {
                "Run w/o scattering": fitburst_info(ran_into_error),
            },
            sheet_tab="Prioritized Events",
        )

    # Last timestamp update:
    update_row(
        eid,
        {"Last fitburst timestamp": str(datetime.now(UTC)).split(".")[0]},
        sheet_tab="Prioritized Events",
    )
    # General tab update:
    update_row(eid, {"Status": "Completed in prioritized list"})
