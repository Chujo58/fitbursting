from dataclasses import dataclass
import json
import sys
from pathlib import Path
import oscfar as ocf
import getpass
from datetime import UTC, datetime

import fitburst.analysis.stats as f_s
import ui as u
import sheets as s
import paths as p

from pint import UnitRegistry

ureg = UnitRegistry()
dm_unit = ureg.parsec / (ureg.cm**3)

FIT_TYPES = ["noscat", "scat", "scint", "scat-scint"]
PREVIOUS_DIAGNOSTIC_FIELDS = [
    "Verifier",
    "Best result",
    "Best number of components",
    "Notes",
]
UNIT_VIEWS = ["Compacted", "Default to ms", "Unitless"]
SHOW_UNITS = "Compacted"


# Global variables needed for application
@dataclass
class RatingDraft:
    best_result: str = ""
    rerun: bool | None = None
    note: str = ""
    best_number_of_components: str = ""


@dataclass
class AppState:
    index: int = 0
    show_image: bool = True
    rating: RatingDraft = None
    selected_fit: str = ""


def run_app(event_ids: list, update_direction="n"):
    app_state = AppState()
    if not event_ids:
        print("No events to display.")
        return

    selected_fits: dict[str, str] = {}
    rating_drafts: dict[str, RatingDraft] = {}

    while True:
        event_id = event_ids[app_state.index]
        results_path = Path(p.RESULTS_PATH) / f"results_{event_id}"
        valid_fits = find_valid_fits(results_path, event_id)

        event_info = s.get_info(event_id, "Prioritized Events")
        app_state.selected_fit = selected_fits.get(event_id, "")
        if valid_fits:
            if app_state.selected_fit not in valid_fits:
                app_state.selected_fit = valid_fits[0]
        else:
            app_state.selected_fit = ""
        selected_fits[event_id] = app_state.selected_fit

        app_state.rating = rating_drafts.get(event_id)
        if app_state.rating is None:
            app_state.rating = draft_from_event_info(event_info)
            rating_drafts[event_id] = app_state.rating

        draw_ui(
            app_state.index,
            event_ids,
            event_info,
            valid_fits,
            app_state.selected_fit,
            app_state.rating,
        )

        if app_state.show_image:
            display_fit_image(results_path, event_id, app_state.selected_fit)
            app_state.show_image = False

        # Define functions for nagivation
        def move(add: int):
            if add > 0:
                app_state.index = min(app_state.index + add, len(event_ids) - 1)
            else:
                app_state.index = max(app_state.index + add, 0)

            app_state.show_image = True

        def jump():
            app_state.index = prompt_jump(event_ids, app_state.index)
            app_state.show_image = True

        def rate():
            app_state.rating = prompt_rating(app_state.selected_fit, app_state.rating)
            rating_drafts[event_id] = app_state.rating

        def switch():
            app_state.selected_fit = prompt_fit_choice(
                valid_fits, app_state.selected_fit
            )
            selected_fits[event_id] = app_state.selected_fit
            app_state.show_image = True

        def show():
            app_state.show_image = True

        def push():
            push_rating_to_sheet(event_id, event_info, app_state.rating)
            moving = 1 if update_direction == "n" else -1
            move(moving)

        options = [
            u.SelectionOption(
                "Previous event",
                "p",
                lambda: move(-1),
            ),
            u.SelectionOption(
                "Next event",
                "n",
                lambda: move(1),
            ),
            u.SelectionOption("Jump to event", "j", jump),
            u.SelectionOption("Rate event", "r", rate),
            u.SelectionOption("Edit unit type", "e", prompt_unit_change),
            u.SelectionOption("Upload rating to spreadsheet", "u", push),
            u.SelectionOption("Switch fit", "f", switch),
            u.SelectionOption("Show image", "i", show),
            u.SelectionOption("Quit", "q", lambda: exit(0)),
        ]
        u.selection_box("Actions", options, 1, 0)
        u.ask_user_input(options)


def setUnitType(unit_type):
    global SHOW_UNITS
    if unit_type in UNIT_VIEWS:
        SHOW_UNITS = unit_type


def prompt_unit_change():
    options = [
        u.SelectionOption("Compacted", "1", lambda: setUnitType("Compacted")),
        u.SelectionOption("Default to ms", "2", lambda: setUnitType("Default to ms")),
        u.SelectionOption("Unitless", "3", lambda: setUnitType("Unitless")),
    ]
    u.selection_box("Pick a unit view", options, 1, 0)
    u.ask_user_input(options)


def prompt_jump(event_ids: list, current_idx: int) -> int:
    raw = input("Jump to index or event ID: ").strip()
    if not raw:
        return current_idx

    print(raw in event_ids)
    if raw in event_ids:
        return event_ids.index(raw)

    try:
        target = int(raw)
    except ValueError:
        print(f"Unrecognized jump target: {raw}")
        return current_idx

    if 1 <= target <= len(event_ids):
        return target - 1

    print(f"Index must be between 1 and {len(event_ids)}")
    return current_idx


def prompt_fit_choice(valid_fits: list[str], current_fit: str) -> str:
    if not valid_fits:
        print("No available fits for this event.")
        return ""

    option_lines = [
        u.SelectionOption(
            color_fit_label(fit, fit == current_fit),
            str(index + 1),
            lambda: None,
        )
        for index, fit in enumerate(valid_fits)
    ]
    u.selection_box("Available fits", option_lines, 1, 15)
    default_prompt = f" [{current_fit}]" if current_fit else ""
    raw = input(f"Choose fit by number or name{default_prompt}: ").strip().lower()
    if not raw and current_fit in valid_fits:
        return current_fit

    if raw in valid_fits:
        return raw

    if raw.isdigit():
        choice = int(raw)
        if 1 <= choice <= len(valid_fits):
            return valid_fits[choice - 1]

    print(f"Unknown fit selection: {raw}")
    return current_fit if current_fit in valid_fits else valid_fits[0]


def prompt_yes_no(prompt: str, current: bool | None) -> bool | None:
    default_text = "y" if current is True else "n" if current is False else ""
    raw = (
        input(f"{prompt} [y/n]{f' ({default_text})' if default_text else ''}: ")
        .strip()
        .lower()
    )
    if not raw:
        return current
    if raw in ("y", "yes", "true", "1"):
        return True
    if raw in ("n", "no", "false", "0"):
        return False
    print(f"Unrecognized yes/no value: {raw}")
    return current


def prompt_text(prompt: str, current: str = "") -> str:
    default_text = f" [{current}]" if current else ""
    raw = input(f"{prompt}{default_text}: ").strip()
    return raw if raw else current


def prompt_rating(selected_fit: str, current: RatingDraft | None = None) -> RatingDraft:
    draft = current or RatingDraft()
    rerun = prompt_yes_no("Rerun needed", draft.rerun)
    note = prompt_text("Optional note", draft.note)

    return RatingDraft(
        best_result=selected_fit or draft.best_result,
        rerun=rerun,
        note=note,
        best_number_of_components="-1",
    )


def draft_from_event_info(event_info: dict) -> RatingDraft:
    def pick_value(candidates: list[str]) -> str:
        for candidate in candidates:
            value = event_info.get(candidate, "")
            if value:
                return value
        return ""

    rerun_value = pick_value(["Rerun R2", "Rerun"])
    rerun = parse_yes_no(rerun_value)

    return RatingDraft(
        best_result=pick_value(["Best result R2", "Best result"]),
        rerun=rerun,
        note=pick_value(["Notes R2", "Notes"]),
        best_number_of_components=pick_value(
            ["Best number of components R2", "Best number of components"]
        )
        or "-1",
    )


def push_rating_to_sheet(event_id: str, event_info: dict, rating: RatingDraft):
    if (
        not rating.best_result
        and rating.rerun is None
        and not rating.note
        and not rating.best_number_of_components
    ):
        print("Nothing to push yet.")
        return

    column_keys, _ = s.get_column_keys("Prioritized Events")
    row_values = {}
    target_suffix = choose_rating_suffix(event_info)

    result_key = pick_column(
        column_keys, [f"Best result {target_suffix}", "Best result"]
    )
    if result_key and rating.best_result:
        row_values[result_key] = rating.best_result

    components_key = pick_column(
        column_keys,
        [f"Best number of components {target_suffix}", "Best number of components"],
    )
    if components_key and rating.best_number_of_components:
        row_values[components_key] = rating.best_number_of_components

    note_key = pick_column(column_keys, [f"Notes {target_suffix}", "Notes"])
    if note_key:
        note_value = rating.note
        # rerun_text = (
        #     None if rating.rerun is None else f"rerun={'yes' if rating.rerun else 'no'}"
        # )
        # if rerun_text and note_value:
        #     note_value = f"{rerun_text}; {note_value}"
        # elif rerun_text:
        #     note_value = rerun_text
        if note_value:
            row_values[note_key] = note_value

    rerun_key = pick_column(
        column_keys, [f"Need to rerun {target_suffix}", "Need to rerun"]
    )
    if rerun_key and rating.rerun is not None:
        row_values[rerun_key] = "yes" if rating.rerun else "no"

    verifier_key = pick_column(column_keys, [f"Verifier {target_suffix}", "Verifier"])
    if verifier_key:
        value = getpass.getuser()
        # if value == "chujo":
        #     value = "clegue"
        row_values[verifier_key] = value

    now_text = str(datetime.now(UTC)).split(".")[0]

    row_values["Notes timestamp"] = now_text

    if not row_values:
        print("No matching columns found in the spreadsheet.")
        return

    s.update_row(event_id, row_values, "Prioritized Events")
    print(f"Pushed rating for {event_id}.")


def choose_rating_suffix(event_info: dict) -> str:
    r1_keys = [
        "Verifier",
        "Best result",
        "Best number of components",
        "Notes",
        "Rerun",
    ]
    for key in r1_keys:
        if event_info.get(key):
            return "R2"

    return "R1"


def pick_column(column_keys: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in column_keys:
            return candidate
    return ""


def display_fit_image(results_path: Path, event_id: str, fit_type: str):
    if not fit_type:
        print("No fit selected to display.")
        return

    image_path = results_path / fit_type / f"summary_plot_{event_id}.png"
    if not image_path.exists():
        print(f"Missing image: {image_path}")
        return

    try:
        from PIL import Image

        Image.open(image_path).show()
    except Exception as exc:
        print(f"Could not display image {image_path}: {exc}")


def draw_ui(
    idx: int,
    event_ids: list,
    event_info: dict,
    valid_fits: list[str],
    selected_fit: str,
    rating: RatingDraft,
):
    u.clear()
    total = len(event_ids)
    event_id = event_ids[idx]
    results_path = Path(p.RESULTS_PATH) / f"results_{event_id}"
    fit_status = generate_fit_status(valid_fits, selected_fit)
    previous_diagnostic_lines = generate_previous_diagnostic_lines(event_info)
    current_rating_lines = generate_current_rating_lines(selected_fit, rating)
    model_info, error_info, og_dm, ref_freq = load_fitburst_model_info(
        results_path, event_id, selected_fit
    )
    model_panel_lines = generate_model_panel_lines(
        model_info, error_info, og_dm, ref_freq
    )

    f_test_info = get_f_test_info(results_path, event_id, valid_fits)

    # UI
    # Title of app + event id + progress bar idk
    u.seperator(110)
    print(u.Color.fg(1) + "fitburst Diagnostics Tool" + u.Style.RESET_ALL)
    u.seperator(110)

    u.progress_bar(
        idx + 1,
        total,
        40,
        f"Event ID: {u.Color.fg(3) + event_id + u.Style.RESET_ALL}",
        None,
        True,
        end="\n",
    )

    left_panel = u.table_lines(["Fit Type", "Does it exist?"], fit_status, 2, 4)
    right_panel = u.box_lines(
        "Previous diagnostic information",
        u.table_lines(
            ["Key", "R1", "R2"],
            previous_diagnostic_lines,
            2,
            4,
            color=[u.Color.fg(2), None, None],
            header_color=[None, None, None],
            col_size=[15, 20, 20],
            sep_color=u.Color.fg(4),
        ),
        1,
        50,
        u.Color.fg(4),
    )
    u.side_by_side(left_panel, right_panel, gap=6)

    print(u.Style.RESET_ALL)  # Adds an empty row

    current_rating_panel = u.box_lines(
        "Current rating",
        u.table_lines(
            ["Key", "Current edit"],
            current_rating_lines,
            2,
            4,
            [u.Color.fg(8), None],
            col_size=[15, 60],
        ),
        1,
        15,
        color=u.Color.fg(6),
    )

    f_test_lines = u.table_lines(
        ["Fit Comparision", "F-Test from fitburst"],
        [[k, v] for k, v in f_test_info.items()],
        2,
        4,
        [u.Color.fg(8), None],
    )

    f_test_panel = u.box_lines(
        "fitburst F-Test", f_test_lines, 1, 15, color=u.Color.fg(6)
    )

    u.side_by_side(current_rating_panel, f_test_panel, gap=2)
    u.box("Current model", model_panel_lines, 1, 0, u.Color.fg(3))

    print(u.Style.RESET_ALL)

    # Selection options


def generate_fit_status(valid_fits, selected_fit):
    fit_box_content = []
    for fit in FIT_TYPES:
        fit_name = (
            fit
            if fit != selected_fit
            else u.Color.bg(9) + u.Color.fg(0) + fit + u.Style.RESET_ALL
        )
        if fit in valid_fits:
            fit_box_content.append(
                [fit_name, u.Color.rgb(0, 255, 0) + "" + u.Style.RESET_ALL]
            )
        else:
            fit_box_content.append(
                [fit_name, u.Color.rgb(255, 0, 0) + "" + u.Style.RESET_ALL]
            )

    return fit_box_content


def generate_previous_diagnostic_lines(event_info):
    lines = []
    for field in PREVIOUS_DIAGNOSTIC_FIELDS:
        r1_value = event_info.get(field, event_info.get(f"{field} R1", ""))
        r2_value = event_info.get(f"{field} R2", "")

        lines.append([field, r1_value, r2_value])

    return lines


def generate_current_rating_lines(selected_fit: str, rating: RatingDraft):
    return [
        ["Selected fit", selected_fit or "none"],
        ["Best result", rating.best_result or "unset"],
        ["Rerun", format_bool(rating.rerun)],
        ["Best number of components", rating.best_number_of_components or "unset"],
        ["Note", rating.note or "unset"],
    ]


def get_f_test_info(results_path: Path, event_id: str, available_fits: list):
    data_returned = {"noscat/scat": "", "noscat/scint": "", "scat/scat-scint": ""}
    if ("scat" in available_fits) and ("noscat" not in available_fits):
        data_returned["noscat/scat"] = "Data unavailable"
    if ("scat-scint" in available_fits) and ("scat" not in available_fits):
        data_returned["scat/scat-scint"] = "Data unavailable"
    if ("scint" in available_fits) and ("noscat" not in available_fits):
        data_returned["noscat/scint"] = "Data unavailable"

    def load_json(fit_type, eid):
        return ocf.utils.FitburstResultsReader(
            results_path / fit_type / f"results_fitburst_{eid}.json"
        )

    def extract_fit_statistics(json1):
        chisq = json1.get_fit_statistics()["chisq_final"]
        p = json1.get_fit_statistics()["num_fit_parameters"]
        n = json1.get_fit_statistics()["num_observations"]
        return chisq, p, n

    def calculate_f_test(event_id, fit1, fit2):
        json1 = load_json(fit1, event_id)
        json2 = load_json(fit2, event_id)

        chisq1, p1, n1 = extract_fit_statistics(json1)
        chisq2, p2, n2 = extract_fit_statistics(json2)
        fres = f_s.compute_test_f(chisq1, chisq2, p1, p2, n1, n2)
        return fres

    if "scat" in available_fits and "noscat" in available_fits:
        data_returned["noscat/scat"] = calculate_f_test(event_id, "noscat", "scat")

    if "scat" in available_fits and "scat-scint" in available_fits:
        data_returned["scat/scat-scint"] = calculate_f_test(
            event_id, "scat", "scat-scint"
        )

    if "noscat" in available_fits and "scint" in available_fits:
        data_returned["noscat/scint"] = calculate_f_test(event_id, "noscat", "scint")

    return data_returned


def load_fitburst_model_info(results_path: Path, event_id: str, fit_type: str):
    if not fit_type:
        print(
            u.Color.rgb(255, 0, 0)
            + "Cannot open fitburst results, invalid fit selected"
            + u.Style.RESET_ALL
        )
        return {}, {}, {}, {}

    json_path = results_path / fit_type / f"results_fitburst_{event_id}.json"
    if not json_path.exists():
        print(
            u.Color.rgb(255, 0, 0)
            + "Cannot open fitburst results, file doesn't exist."
            + u.Style.RESET_ALL
        )
        return {}, {}, {}, {}

    try:
        with json_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        print(
            u.Color.rgb(255, 0, 0)
            + "Issue opening fitburst results"
            + u.Style.RESET_ALL
        )
        return {}, {}, {}, {}

    original_dm = data.get("initial_dm")
    ref_freq = data.get("model_parameters").get("ref_freq")[0]
    # model_info = data.get("model_parameters")
    fit_statistics = data.get("fit_statistics")
    error_info = {}
    if fit_statistics:
        model_info = fit_statistics.get("bestfit_parameters")
        error_info = fit_statistics.get("bestfit_uncertainties")

    if not model_info:
        model_info["Current model"] = "unset"
    if not error_info:
        error_info["Current errors"] = "unset"

    return model_info, error_info, original_dm, ref_freq


def extract_first_json_value(data, candidate_keys: list[str]):
    if isinstance(data, dict):
        for key in candidate_keys:
            if key in data:
                value = data[key]
                if value not in (None, ""):
                    return value

        for value in data.values():
            found = extract_first_json_value(value, candidate_keys)
            if found not in (None, ""):
                return found

    if isinstance(data, list):
        for item in data:
            found = extract_first_json_value(item, candidate_keys)
            if found not in (None, ""):
                return found

    return ""


def format_model_value(value):
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            return json.dumps(value, ensure_ascii=False)
        return ", ".join(str(item) for item in value)

    return str(value)


def create_value(value, error, unit):
    if not value:  # The value is None
        return ""
    if error:
        return (value * unit).plus_minus(error)
    else:
        return value * unit


def format_value(data, isDM=False, timeUnit=False):
    fmt = lambda x: "{:~P}".format(x)
    if isinstance(data, str):
        return ""
    if isDM:
        return fmt(data)
    if not timeUnit or (timeUnit and SHOW_UNITS == "Unitless"):
        return fmt(data)

    if SHOW_UNITS == "Compacted":
        return fmt(data.to_compact())

    if SHOW_UNITS == "Default to ms":
        return fmt(data.to("ms"))


def generate_model_panel_lines(
    model_info: dict[str, object],
    error_info: dict[str, object],
    initial_dm: int,
    ref_freq: float,
):
    rows = []
    component_count = 0

    for field, value in model_info.items():
        if isinstance(value, list):
            component_count = max(component_count, len(value))
        else:
            component_count = max(component_count, 1)

    if component_count == 0:
        return u.table_lines(
            ["Parameter", "Component 1"], [["Current model", "unset"]], 2, 0
        )
    # First, we need to create the headers from the parameter list
    parameter_aliases = {
        "amplitude": "A",
        "arrival_time": "t₀",
        "burst_width": "σ",
        "dm": "dm",
        "scattering_timescale": f"τ@{(ref_freq * ureg.Hz * 1e6).to_compact():~P}",
        "scattering_index": "τ(idx)",
        "spectral_running": "spec",
        "spectral_index": "spec(idx)",
    }
    parameter_units = {
        "amplitude": ureg.dimensionless,
        "arrival_time": ureg.second,
        "burst_width": ureg.second,
        "dm": dm_unit,
        "spectral_index": ureg.dimensionless,
        "spectral_running": ureg.dimensionless,
        "scattering_timescale": ureg.second,
        "scattering_index": ureg.dimensionless,
    }

    parameter_aliases_kept = {
        k: v for k, v in parameter_aliases.items() if k in model_info.keys()
    }

    headers = ["Component Number"]
    headers.extend(
        parameter_aliases_kept.values()
        # [e if e != "τ" else "τ@400MHz" for e in list(parameter_aliases.values())]
    )

    for i in range(component_count):
        row = [i + 1]
        for field in parameter_aliases_kept.keys():
            noFormat = False
            timeUnit = False
            value = model_info.get(field, [None] * component_count)
            error = error_info.get(field, [None] * component_count)
            unit = parameter_units.get(field)

            value_to_use = None
            error_to_use = None
            if len(value) == 1:
                value_to_use = value[0]
                error_to_use = error[0]
            else:
                value_to_use = value[i]
                error_to_use = error[i]

            if field == "dm":
                noFormat = True
                value_to_use += initial_dm

            if unit == ureg.second:
                timeUnit = True

            if SHOW_UNITS == "Unitless":
                unit = ureg.dimensionless

            row.append(
                format_value(
                    create_value(value_to_use, error_to_use, unit),
                    isDM=noFormat,
                    timeUnit=timeUnit,
                )
            )
        rows.append(row)

    # for idx, (field, value) in enumerate(model_info.items()):
    #     error = error_info.get(field)
    #
    #     value_expanded = value
    #     if len(value) != component_count:
    #         value_expanded = value * component_count
    #
    #     row = [idx] + value_expanded
    #     rows.append(row)

    # headers = ["Parameter"] + [
    #     f"Component {index}" for index in range(1, component_count + 1)
    # ]
    #
    # for field, value in model_info.items():
    #     if isinstance(value, list):
    #         row = [field] + [
    #             format_model_value(item) if item not in (None, "") else ""
    #             for item in value
    #         ]
    #         row.extend([""] * (component_count - len(value)))
    #     else:
    #         row = [field] + [format_model_value(value)] + [""] * (component_count - 1)
    #     rows.append(row)

    return u.table_lines(headers, rows, 2, 0)


def format_bool(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unset"


def color_fit_label(fit: str, selected: bool) -> str:
    if selected:
        return u.Color.fg(4) + fit + u.Style.RESET_ALL
    return fit


def parse_yes_no(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in ("y", "yes", "true", "1"):
        return True
    if normalized in ("n", "no", "false", "0"):
        return False
    return None


def find_valid_fits(results_path, eid):
    valid_fits = []
    for fit_type in FIT_TYPES:
        if (results_path / fit_type).exists() and (
            results_path / fit_type / f"summary_plot_{eid}.png"
        ).exists():
            valid_fits.append(fit_type)

    return valid_fits


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # print(sys.argv)
        # exit(0)
        tab = sys.argv[1]
        skip = 2
    else:
        tab = "Prioritized Events"
        skip = 1

    if len(sys.argv) == 3:
        direction = sys.argv
    else:
        direction = "n"
    # event_ids, sheet = s.get_spreadsheet("Prioritized Events")
    # df = pd.DataFrame(sheet[1:])
    # df.columns = sheet[0]

    df = s.get_spreadsheet_df(tab, row_skip=skip)

    masked_df = df[df["To rerun?"] == "to diagnose"]
    run_app(masked_df["Event ID"].astype(str).tolist(), direction)
