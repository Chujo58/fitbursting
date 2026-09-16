#!/usr/bin/env python3
"""Move fitburst result files for a given event ID into their final location.

Looks for `summary_plot_<eid>.png` and `results_fitburst_<eid>.json` in the
current working directory and moves them to:

    RESULTS_PATH/results_<eid>/<fit_type>/

where RESULTS_PATH is imported from paths.py.

Usage:
    python move_results.py <eid> <fit_type>

    fit_type must be one of: noscat, scat, scat-scint, scint

Existing files at the destination are overwritten.
"""

import argparse
import shutil
import sys
from pathlib import Path
from sheets import get_info, update_row
from datetime import datetime, UTC
from paths import RESULTS_PATH

VALID_FIT_TYPES = ("noscat", "scat", "scat-scint", "scint")
PRIORITY_COL = {
    "noscat": "Run w/o scattering",
    "scat": "Run w/ scattering",
    "scint": "Run w/ scintillation & w/o scattering",
    "scat-scint": "Run w/ scintillation & scattering",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move summary plot and results json for an eid into its results folder."
    )
    parser.add_argument("eid", help="Event ID")
    parser.add_argument(
        "fit_type",
        choices=VALID_FIT_TYPES,
        help=f"Fit type, one of: {', '.join(VALID_FIT_TYPES)}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eid = args.eid
    fit_type = args.fit_type

    try:
        priority_info = get_info(eid, "Prioritized Events")
    except KeyError:
        priority_info = None

    cwd = Path.cwd()
    png_name = f"summary_plot_{eid}.png"
    json_name = f"results_fitburst_{eid}.json"

    src_png = cwd / png_name
    src_json = cwd / json_name

    missing = [p.name for p in (src_png, src_json) if not p.exists()]
    if missing:
        print(
            f"Error: could not find the following file(s) in {cwd}:\n"
            + "\n".join(f"  - {m}" for m in missing),
            file=sys.stderr,
        )
        sys.exit(1)

    dest_dir = Path(RESULTS_PATH) / f"results_{eid}" / fit_type
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_png = dest_dir / png_name
    dest_json = dest_dir / json_name

    for src, dest in [(src_png, dest_png), (src_json, dest_json)]:
        if dest.exists():
            print(f"Overwriting existing file: {dest}")
            dest.unlink()
        shutil.move(str(src), str(dest))
        print(f"Moved {src.name} -> {dest}")

    if priority_info:
        update_row(
            eid,
            {
                "Last fitburst timestamp": str(datetime.now(UTC)).split(".")[0],
                PRIORITY_COL[fit_type]: "Completed",
            },
            sheet_tab="Prioritized Events",
        )

    print(f"\nDone. Files are now in: {dest_dir}")


if __name__ == "__main__":
    main()
