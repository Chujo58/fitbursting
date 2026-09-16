#!/usr/bin/env python3
"""Run fitburst for one event after overriding burst parameters in a local NPZ copy.

Workflow:
1. Read default event NPZ from NPZ_SAVE_PATH/<eid>.npz.
2. Copy to local ./<eid>.npz while updating burst_parameters via oscfar.
3. Run fitburst modes (noscat/scat/scint/scat-scint) on the local file.
4. Move outputs into RESULTS_PATH/results_<eid>/<mode>/.
5. Delete local ./<eid>.npz when finished.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from sheets import update_row, get_info
from datetime import datetime, UTC

import oscfar

from paths import NPZ_SAVE_PATH, RESULTS_PATH


FITBURST_PARSER_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fitburst",
    "parser_configs",
    "fitburst_parser_config.json",
)

# These are updated in the local NPZ burst_parameters and replicated as needed.
BURST_PARAMETER_FLAGS = {
    "amplitude",
    "arrival_time",
    "dm",
    "dm_index",
    "width",
    "spectral_index",
    "spectral_running",
    "scattering_timescale",
    "scattering_index",
}

# Handled directly by this wrapper.
WRAPPER_MANAGED_FLAGS = {
    "file",
    "scintillation",
    "solution",
    "verbose",
    "outfile",
}


def _json_type_to_python_type(type_name):
    """Convert parser-config type names into Python argparse types."""
    if type_name == "int":
        return int
    if type_name == "float":
        return float
    return str


def load_passthrough_specs(config_path=FITBURST_PARSER_CONFIG):
    """Load fitburst CLI options from config, excluding burst/wrapper-managed flags."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    specs = []
    for arg_spec in config.get("arguments", []):
        name = arg_spec.get("name")
        if not name or not name.startswith("--"):
            continue

        flag = name.lstrip("-")
        if flag in BURST_PARAMETER_FLAGS or flag in WRAPPER_MANAGED_FLAGS:
            continue

        specs.append(arg_spec)

    return specs


def add_passthrough_arguments(parser):
    """Register passthrough options from fitburst parser config.

    Returns a mapping of argparse destination names to fitburst CLI flag names.
    """
    arg_map = {}

    for spec in load_passthrough_specs():
        cli_name = spec["name"]
        flag = cli_name.lstrip("-")
        dest = spec.get("dest", flag.replace("-", "_"))

        kwargs = {"dest": dest, "help": spec.get("help")}
        action = spec.get("action")

        if action:
            kwargs["action"] = action
            # For store_true passthrough flags we only pass through when set.
            if action == "store_true":
                kwargs["default"] = False
        else:
            kwargs["default"] = None
            if spec.get("type") is not None:
                kwargs["type"] = _json_type_to_python_type(spec["type"])
            if spec.get("nargs") is not None:
                kwargs["nargs"] = spec["nargs"]

        parser.add_argument(cli_name, **kwargs)
        arg_map[dest] = flag

    return arg_map


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

    str_def_scat = [str(float(v)) for v in def_scat]

    arg_out = "--scattering_timescale"

    out = [arg_out]
    out.extend(str_def_scat)

    return out


def build_fitburst_command(
    npz_file,
    scattering=False,
    scintillation=False,
    solution=None,
    passthrough_overrides=None,
):
    """Build fitburst command string."""
    cmd_parts = [
        sys.executable,
        "fitburst/basecat2_fitburst.py",
        npz_file,
        "--verbose",
        "--outfile",
    ]

    comp_count, scattering_values = get_component_count_and_scattering(npz_file)
    print(f"Scattering values: {scattering_values}")
    if solution:
        if scattering:
            scat_arg = generate_scat_arg(comp_count, scattering_values)
        else:
            scat_arg = generate_scat_arg(comp_count)

    print(solution)

    if scintillation:
        cmd_parts.append("--scintillation")

    if scattering:
        cmd_parts.extend(["--fit", "scattering_timescale"])

    if solution:
        cmd_parts.extend(["--solution", solution])
        cmd_parts.extend(scat_arg)

    if passthrough_overrides:
        for param, values in passthrough_overrides.items():
            cmd_parts.append(f"--{param}")
            if isinstance(values, bool):
                # Presence-only flags (e.g. --preprocess) should not add a value.
                continue
            if isinstance(values, (list, tuple)):
                cmd_parts.extend(str(v) for v in values)
            else:
                cmd_parts.append(str(values))

    return " ".join(shlex.quote(part) for part in cmd_parts)


def get_overrides(args, passthrough_arg_map, file):
    """Build burst-parameter and passthrough overrides from CLI.

    Multi-value burst parameters are replicated to match the longest provided
    length. This ensures fitburst gets consistent component counts.
    """
    burst_defaults = {
        "amplitude": [1.0],
        "arrival_time": [0.0],
        "dm": [0],
        "dm_index": [-2],
        "burst_width": [0.001],
        "spectral_index": [0],
        "spectral_running": [0],
        "scattering_timescale": [0.001],
        "scattering_index": [-4],
    }

    if args.npz_values:
        temp = oscfar.npz_reader(file)
        burst_defaults = temp.burst_parameters

    burst_params = {
        "amplitude": (
            args.amplitude
            if args.amplitude is not None
            else burst_defaults["amplitude"]
        ),
        "arrival_time": (
            args.arrival_time
            if args.arrival_time is not None
            else burst_defaults["arrival_time"]
        ),
        "dm": args.dm if args.dm is not None else burst_defaults["dm"],
        "dm_index": (
            args.dm_index if args.dm_index is not None else burst_defaults["dm_index"]
        ),
        "burst_width": (
            args.width if args.width is not None else burst_defaults["burst_width"]
        ),
        "spectral_index": (
            args.spectral_index
            if args.spectral_index is not None
            else burst_defaults["spectral_index"]
        ),
        "spectral_running": (
            args.spectral_running
            if args.spectral_running is not None
            else burst_defaults["spectral_running"]
        ),
        "scattering_timescale": (
            args.scattering_timescale
            if args.scattering_timescale is not None
            else burst_defaults["scattering_timescale"]
        ),
        "scattering_index": (
            args.scattering_index
            if args.scattering_index is not None
            else burst_defaults["scattering_index"]
        ),
    }

    max_len = 1
    for values in burst_params.values():
        if isinstance(values, (list, tuple)):
            max_len = max(max_len, len(values))

    replicated_burst_params = {}
    for param, values in burst_params.items():
        if isinstance(values, (list, tuple)):
            replicated = (list(values) * ((max_len // len(values)) + 1))[:max_len]
            replicated_burst_params[param] = replicated
        else:
            replicated_burst_params[param] = values

    passthrough = {}
    for dest, flag in passthrough_arg_map.items():
        value = getattr(args, dest, None)
        if value is None:
            continue

        if isinstance(value, bool):
            if value:
                passthrough[flag] = True
            continue

        passthrough[flag] = value

    return replicated_burst_params, passthrough


def apply_freq_channel_mask(writer, mask_freq_range):
    """Zero out a contiguous range of frequency channels in the local NPZ copy."""
    if not mask_freq_range:
        return None

    start_chan, end_chan = sorted(mask_freq_range)
    num_freq = int(writer.data_full.shape[0])

    if end_chan < 0 or start_chan >= num_freq:
        raise ValueError(
            f"Mask range [{start_chan}, {end_chan}] is outside valid channel indices [0, {num_freq - 1}]"
        )

    start_chan = max(0, start_chan)
    end_chan = min(num_freq - 1, end_chan)

    writer.data_full[start_chan : end_chan + 1, :] = 0.0

    # Keep weights consistent with data when available.
    if hasattr(writer, "data_weights") and writer.data_weights is not None:
        writer.data_weights[start_chan : end_chan + 1, :] = 0.0

    return start_chan, end_chan


def create_local_npz_copy(
    source_npz, local_npz, burst_param_overrides, mask_freq_range=None
):
    """Create local NPZ copy and apply burst-parameter overrides."""
    reader = oscfar.npz_reader(source_npz)
    writer = oscfar.npz_writer(reader)
    writer.update_burst_parameters(**burst_param_overrides)
    applied_mask = apply_freq_channel_mask(writer, mask_freq_range)
    writer.save(local_npz)
    return applied_mask


def move_npz(eid):
    src = os.path.abspath(f"{eid}.npz")
    dst = f"{NPZ_SAVE_PATH}/{eid}.npz"

    if not os.path.exists(src):
        print(f" Warning {src} not found, skipping.")

    if os.path.exists(dst):
        try:
            os.remove(dst)
            print(f"✓ Removed existing {dst}")
        except Exception as e:
            print(f"✗ Cannot remove existing {dst}: {e}")
            return

    try:
        shutil.copy(src, dst)
        print(f"✓ Copied {src} -> {dst}")
    except Exception as e:
        print(f"✗ Copy failed: {e}")
        return

    try:
        os.remove(src)
        print(f"✓ Removed source {src}")
    except Exception as e:
        print(f"✗ Cannot remove source {src}: {e}")
        return

    print(f"File successfully copied to {dst}")


def move(eid, fit_type="scat-scint"):
    """Move fitburst artifacts into mode-specific result directory."""
    dest_dir = f"{RESULTS_PATH}/results_{eid}/{fit_type}"
    os.makedirs(dest_dir, exist_ok=True)

    file_strip = eid
    files_to_move = [
        f"results_fitburst_{file_strip}.json",
        f"summary_plot_{file_strip}.png",
    ]

    for filename in files_to_move:
        src = os.path.abspath(filename)
        dst = f"{dest_dir}/{filename}"

        if not os.path.exists(src):
            print(f" Warning: {src} not found, skipping.")
            continue

        if os.path.exists(dst):
            try:
                os.remove(dst)
                print(f"✓ Removed existing {dst}")
            except Exception as e:
                print(f"✗ Cannot remove existing {dst}: {e}")
                continue

        try:
            shutil.copy(src, dst)
            print(f"✓ Copied {src} -> {dst}")
        except Exception as e:
            print(f"✗ Copy failed: {e}")
            continue

        try:
            os.remove(src)
            print(f"✓ Removed source {src}")
        except Exception as e:
            print(f"✗ Cannot remove source {src}: {e}")

        print(f"File successfully copied to {dst}")


def validate_file(file_path):
    """Inspect NPZ file for non-empty data and burst parameters."""
    reader = oscfar.npz_reader(file_path)

    burst_parameters = reader.burst_parameters
    if not burst_parameters:
        print(f"Error: burst_parameters is empty in {file_path}.")
        return False

    for key, value in burst_parameters.items():
        if isinstance(value, (list, tuple, oscfar.np.ndarray)) and len(value) == 0:
            print(f"Error: burst_parameters[{key}] is empty in {file_path}.")
            return False

    data = reader.data_full
    if data is None or len(data) == 0:
        print(f"Error: data_full is empty in {file_path}.")
        return False

    return True


parser = argparse.ArgumentParser(
    description="Run fitburst for one event with burst-parameter overrides in a local NPZ copy"
)
parser.add_argument("eid", type=str, help="The event ID to process")
parser.add_argument("--scattering", action="store_true", help="Enable scattering fit")
parser.add_argument(
    "--scintillation", action="store_true", help="Enable scintillation fit"
)

parser.add_argument("--amplitude", nargs="+", type=float)
parser.add_argument("--arrival_time", nargs="+", type=float)
parser.add_argument("--dm", nargs="+", type=float)
parser.add_argument("--dm_index", nargs="+", type=float)
parser.add_argument("--width", nargs="+", type=float)
parser.add_argument("--spectral_index", nargs="+", type=float)
parser.add_argument("--spectral_running", nargs="+", type=float)
parser.add_argument("--scattering_timescale", nargs="+", type=float)
parser.add_argument("--scattering_index", nargs="+", type=float)
parser.add_argument(
    "--mask-freq-range",
    nargs=2,
    type=int,
    metavar=("START_CHAN", "END_CHAN"),
    help="Zero out data_full rows for a contiguous frequency-channel range in the temporary local NPZ copy (inclusive indices).",
)

passthrough_arg_map = add_passthrough_arguments(parser)
parser.add_argument(
    "--use-solution",
    action="store_true",
    help="Use previous solution from RESULTS_PATH if available.",
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print actions/commands but do not execute fitburst.",
)
parser.add_argument(
    "--copy-over",
    action="store_true",
    help="Copy the results over to the results path.",
)
parser.add_argument(
    "--copy-npz",
    action="store_true",
    help="Copy the local NPZ into NPZ_SAVE_PATH.",
)
parser.add_argument(
    "--copy-all", action="store_true", help="Copies the results and the NPZ files."
)
parser.add_argument(
    "--npz-values", action="store_true", help="Use the default values from the npz"
)
parser.add_argument(
    "--solution",
    default="null",
    type=str,
    help="If set, use existing solution in fitburst-compliant JSON format.",
)

args = parser.parse_args()


if __name__ == "__main__":
    eid = args.eid
    scattering = args.scattering
    scintillation = args.scintillation
    dry_run = args.dry_run
    copy_over = args.copy_over
    copy_npz = args.copy_npz
    copy_all = args.copy_all
    solution = args.solution

    if copy_all:
        copy_over = True
        copy_npz = True

    try:
        priority_info = get_info(eid, "Prioritized Events")
    except KeyError:
        priority_info = None

    source_npz = f"{NPZ_SAVE_PATH}/{eid}.npz"
    local_npz = os.path.abspath(f"{eid}.npz")

    burst_overrides, passthrough_overrides = get_overrides(
        args, passthrough_arg_map, source_npz
    )

    mode_msg = "with scattering" if scattering else "without scattering"
    mode_msg += (
        " and with scintillation" if scintillation else " and without scintillation"
    )
    print(f"Running event ID: {eid} {mode_msg}")
    print(f"Source NPZ: {source_npz}")
    print(f"Local NPZ copy: {local_npz}")

    if dry_run:
        print("[DRY RUN MODE - no file writes or subprocess calls]\n")
        print(f"Would update burst_parameters with: {burst_overrides}")
        if args.mask_freq_range:
            start_chan, end_chan = sorted(args.mask_freq_range)
            print(
                f"Would mask local NPZ channels (inclusive): [{start_chan}, {end_chan}]"
            )
        if copy_npz:
            print("Would copy local NPZ into the results directory.")
    else:
        applied_mask = create_local_npz_copy(
            source_npz,
            local_npz,
            burst_overrides,
            mask_freq_range=args.mask_freq_range,
        )
        print(f"Created local NPZ copy with updated burst parameters: {local_npz}")
        if applied_mask is not None:
            print(
                "Applied local channel mask (inclusive): "
                f"[{applied_mask[0]}, {applied_mask[1]}]"
            )

    if not dry_run and not validate_file(local_npz):
        print("Validation failed for local NPZ copy. Exiting.")
        if os.path.exists(local_npz):
            os.remove(local_npz)
        sys.exit(1)

    ran_into_error = 0

    try:
        if scintillation and scattering:
            # solution = None
            if args.use_solution:
                solution = (
                    f"{RESULTS_PATH}/results_{eid}/scat/results_fitburst_{eid}.json"
                )
            cmd = build_fitburst_command(
                local_npz,
                scattering=True,
                scintillation=True,
                solution=solution,
                passthrough_overrides=passthrough_overrides,
            )
            print(f"[scat-scint] Command: {cmd}")
            if not dry_run:
                subprocess.run(cmd, shell=True, check=True, env=os.environ.copy())
                if copy_over:
                    move(eid, fit_type="scat-scint")

        elif scintillation and not scattering:
            # solution = None
            if args.use_solution:
                solution = (
                    f"{RESULTS_PATH}/results_{eid}/noscat/results_fitburst_{eid}.json"
                )
            cmd = build_fitburst_command(
                local_npz,
                scattering=False,
                scintillation=True,
                solution=solution,
                passthrough_overrides=passthrough_overrides,
            )
            print(f"[scint] Command: {cmd}")
            if not dry_run:
                subprocess.run(cmd, shell=True, check=True, env=os.environ.copy())
                if copy_over:
                    move(eid, fit_type="scint")

        elif scattering and not scintillation:
            # solution = None
            if args.use_solution:
                solution = (
                    f"{RESULTS_PATH}/results_{eid}/noscat/results_fitburst_{eid}.json"
                )
            cmd = build_fitburst_command(
                local_npz,
                scattering=True,
                scintillation=False,
                solution=solution,
                passthrough_overrides=passthrough_overrides,
            )
            print(f"[scat] Command: {cmd}")
            if not dry_run:
                subprocess.run(cmd, shell=True, check=True, env=os.environ.copy())
                if copy_over:
                    move(eid, fit_type="scat")

        else:
            cmd = build_fitburst_command(
                local_npz,
                scattering=False,
                scintillation=False,
                solution=solution,
                passthrough_overrides=passthrough_overrides,
            )
            print(f"[noscat] Command: {cmd}")
            if not dry_run:
                subprocess.run(cmd, shell=True, check=True, env=os.environ.copy())
                if copy_over:
                    move(eid, fit_type="noscat")

    except Exception as e:
        print(f"Error while running fitburst for {eid}: {e}")
        ran_into_error += 1

    finally:
        if not dry_run and os.path.exists(local_npz):
            if copy_npz:
                move_npz(eid)
            else:
                try:
                    os.remove(local_npz)
                    print(f"Deleted local NPZ copy: {local_npz}")
                except Exception as e:
                    print(f"Warning: failed to delete local NPZ copy {local_npz}: {e}")

    if dry_run:
        print(f"[DRY RUN] Completed preview for event ID: {eid}")
    else:
        print(f"Completed fitburst for event ID: {eid} with {ran_into_error} errors.")
        if priority_info and (copy_all or copy_over):
            update_row(
                eid,
                {"Last fitburst timestamp": str(datetime.now(UTC)).split(".")[0]},
                sheet_tab="Prioritized Events",
            )
        if ran_into_error != 0:
            sys.exit(1)
