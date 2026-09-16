"""Google Sheets helpers for pipeline status tracking.

This module provides utilities for reading from and writing to a Google Sheet
that tracks event processing across the magnetron2-fitburst pipeline. It handles
authentication via service account credentials and provides high-level functions
for common operations like fetching event data, updating rows, and managing
batch processing status.

The sheet is expected to have event IDs in the first column and various metadata
columns (Status, runtime, etc.) in subsequent columns.

Examples
--------
Get all event IDs and their data from the sheet:

    >>> event_ids, column_values = get_spreadsheet()
    >>> print(event_ids)  # List of event ID strings
    >>> print(column_values)  # All data including headers

Check if an event exists:

    >>> if row_exists('123456789'):
    ...     print("Event already in sheet")

Update an event's status:

    >>> update_row('123456789', {'Status': 'Complete', 'Runtime': '2.5 hours'})
"""

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas
import os
import time

path = os.path.dirname(os.path.abspath(__file__))

SERVICE_ACCOUNT_FILE = f"{path}/google_service_account.json"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

credentials = Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)  # authentication
service = build("sheets", "v4", credentials=credentials)

spreadsheet_id = (
    "1IbxciCSm2gIkBLxVR1_jQz5efxdJFQ7WdLYaXz-6iyw"  # This is for magnetron2
)
# default tab name; functions accept an optional `sheet_tab` parameter
sheet_tab = "Sheet1"
spreadsheet_range = f"{sheet_tab}!A:Z"


def _execute_with_backoff(request, max_retries: int = 5, base_delay: float = 1.0):
    """Execute a Google Sheets API request with exponential backoff retry logic.

    Handles transient HTTP errors (429 Too Many Requests, 503 Service Unavailable)
    by automatically retrying with exponentially increasing delays. Permanent errors
    are raised immediately without retry.

    Parameters
    ----------
    request : googleapiclient.http.HttpRequest
        The API request object to execute. Typically obtained from service methods
        like service.spreadsheets().values().get().
    max_retries : int, optional
        Maximum number of retry attempts after the initial request fails.
        Default is 5.
    base_delay : float, optional
        Initial delay in seconds before the first retry. Subsequent retries use
        exponential backoff with delay = base_delay * 2^attempt_number.
        Default is 1.0 second.

    Returns
    -------
    dict
        The response from the API request.

    Raises
    ------
    googleapiclient.errors.HttpError
        If a non-transient error occurs or max_retries is exceeded.

    Examples
    --------
    >>> request = service.spreadsheets().values().get(spreadsheetId=sid, range='Sheet1!A:A')
    >>> result = _execute_with_backoff(request, max_retries=5, base_delay=1.0)
    >>> values = result.get('values', [])
    """
    for attempt in range(max_retries + 1):
        try:
            return request.execute()
        except HttpError as err:
            status = getattr(err.resp, "status", None)
            is_retryable = status in (429, 500, 503)
            if (not is_retryable) or attempt == max_retries:
                raise

            delay = base_delay * (2**attempt)
            print(
                f"Google Sheets request failed with status {status}; retrying in {delay:.1f}s"
            )
            time.sleep(delay)


def get_sheet_id_mapping(spreadsheet_id):
    """Retrieves a dictionary mapping sheet titles to their unique sheetIds."""
    request = service.spreadsheets().get(spreadsheetId=spreadsheet_id)
    metadata = _execute_with_backoff(request)

    # Create a map of { "SheetName": integer_id }
    return {
        s["properties"]["title"]: s["properties"]["sheetId"] for s in metadata["sheets"]
    }


def _column_letter(index: int) -> str:
    """Convert a 0-based column index to Excel-style column letter(s).

    Maps numeric indices to column labels used in Excel/Google Sheets A1 notation.
    Supports unlimited columns through multi-letter labels (A, B, ..., Z, AA, AB, ...).

    Parameters
    ----------
    index : int
        0-based column index. Must be >= 0.

    Returns
    -------
    str
        Column label in A1 notation (e.g., 'A', 'Z', 'AA', 'ABC').

    Raises
    ------
    ValueError
        If index is negative.

    Examples
    --------
    >>> _column_letter(0)
    'A'
    >>> _column_letter(25)
    'Z'
    >>> _column_letter(26)
    'AA'
    >>> _column_letter(701)
    'ZZ'
    >>> _column_letter(702)
    'AAA'
    """
    if index < 0:
        raise ValueError("Column index must be non-negative")

    label = ""
    current = index
    while True:
        current, remainder = divmod(current, 26)
        label = chr(65 + remainder) + label
        if current == 0:
            break
        current -= 1

    return label


def _resolve_sheet_target(
    sheet_tab: str | None = None, spreadsheet_id_value: str | None = None
) -> tuple[str, str]:
    """Resolve sheet tab name and spreadsheet ID, using provided values or defaults.

    This internal helper determines which sheet tab and spreadsheet to operate on,
    allowing per-function call overrides while falling back to module-level defaults.

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab (e.g., 'Sheet1'). If None, uses the module-level
        `sheet_tab` global variable.
    spreadsheet_id_value : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        `spreadsheet_id` global variable.

    Returns
    -------
    tuple[str, str]
        A tuple of (resolved_sheet_tab, resolved_spreadsheet_id).

    Examples
    --------
    >>> tab, sid = _resolve_sheet_target('CustomSheet', 'abc123xyz')
    >>> tab
    'CustomSheet'
    >>> tab, sid = _resolve_sheet_target()  # Uses module defaults
    """
    tab = sheet_tab or globals().get("sheet_tab")
    sid = spreadsheet_id_value or globals().get("spreadsheet_id")
    return tab, sid


def get_spreadsheet(
    sheet_tab: str | None = None,
    spreadsheet_id: str | None = None,
    column_range: str = "A:Z",
):
    """Fetch all event IDs and data from the target Google Sheet.

    Downloads the full sheet data including headers and all rows. Returns both
    a convenient list of event IDs and the complete raw data for flexibility.

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab to read from. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.
    column_range : str, optional
        The range of column you want to fetch. If not specified, uses the
        default A:Z range.

    Returns
    -------
    tuple[list[str], list[list]]
        A tuple of:
        - event_ids : List of event ID strings extracted from the first column
          (starting from row 2, skipping empty rows)
        - column_values : All sheet data as a list of rows, where each row is
          a list of column values. Row 1 contains headers.

    Examples
    --------
    >>> event_ids, all_data = get_spreadsheet()
    >>> print(f"Found {len(event_ids)} events")
    >>> print(f"Headers: {all_data[0]}")  # First row contains headers
    >>> print(f"First event: {event_ids[0]}")

    >>> # Use custom sheet
    >>> eids, data = get_spreadsheet(sheet_tab='Archive', spreadsheet_id='other_sid_xyz')
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    sheet = service.spreadsheets()
    result = _execute_with_backoff(
        sheet.values().get(spreadsheetId=sid, range=f"{tab}!{column_range}")
    )
    column_values = result.get("values", [])
    event_ids = [item[0] for item in column_values[1:] if item]

    return event_ids, column_values


def get_spreadsheet_df(
    sheet_tab: str | None = None,
    spreadsheet_id: str | None = None,
    column_range: str = "A:Z",
    row_skip: int = 1,
) -> pandas.DataFrame:
    """Fetch data from a spreadsheet and format it to a pandas.DataFrame

    Makes use of the `get_spreadsheet` method.

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab to read from. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.
    column_range : str, optional
        The range of column you want to fetch. If not specified, uses the
        default A:Z range.
    row_skip : int, optional
        The number of rows to skip, or the index location of the first row of data.

    Returns
    -------
    pandas.DataFrame
        A dataframe which contains the entire spreadsheet for more convenient parsing and filtering.
    """
    assert row_skip > 0, (
        "The index location of the first data row cannot be 0 or negative. Please use another value."
    )
    _, data = get_spreadsheet(sheet_tab, spreadsheet_id, column_range)
    df = pandas.DataFrame(data[row_skip:])
    df.columns = data[row_skip - 1]
    return df


def get_event_row_map(sheet_tab: str | None = None, spreadsheet_id: str | None = None):
    """Create a mapping from event IDs to their row numbers in the sheet.

    Useful for determining which sheet row contains a specific event's data.
    Row numbers are 1-based (matching Google Sheets) with row 1 as headers.

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab to read from. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    dict[str, int]
        A dictionary mapping event ID (as string) to sheet row number (1-based).
        Empty rows are skipped. Whitespace is stripped from event IDs.

    Examples
    --------
    >>> row_map = get_event_row_map()
    >>> row_number = row_map['123456789']
    >>> print(f"Event 123456789 is in row {row_number}")
    >>> if '987654321' in row_map:
    ...     print("Event exists in sheet at row", row_map['987654321'])
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    sheet = service.spreadsheets()
    result = _execute_with_backoff(
        sheet.values().get(spreadsheetId=sid, range=f"{tab}!A2:A")
    )

    values = result.get("values", [])
    row_map = {}
    for row_index, row in enumerate(values, start=2):
        if not row:
            continue

        event_id = str(row[0]).strip()
        if not event_id:
            continue

        row_map[event_id] = row_index

    return row_map


def row_exists(
    eventid: str, spreadsheet_or_tab=None, spreadsheet_id: str | None = None
) -> bool:
    """Check whether an event ID exists in the sheet.

    Provides two usage patterns: pass pre-fetched spreadsheet data for efficiency
    when checking multiple events, or pass a sheet_tab name for convenience.

    Parameters
    ----------
    eventid : str or int
        The event ID to search for. Will be converted to string for comparison.
    spreadsheet_or_tab : tuple or str, optional
        Either:
        - A tuple (event_ids, column_values) returned by get_spreadsheet(),
          for efficient batch checking without re-fetching
        - A sheet tab name (str), to fetch data from that specific tab
        - None (default) to use the module-level default sheet tab
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. Only used if spreadsheet_or_tab is
        a string or None. If None, uses the module-level default.

    Returns
    -------
    bool
        True if the event ID exists in the sheet, False otherwise.

    Examples
    --------
    >>> # Simple usage with defaults
    >>> if row_exists('123456789'):
    ...     print("Event already recorded")

    >>> # Efficient batch checking with pre-fetched data
    >>> event_ids, data = get_spreadsheet()
    >>> for eid in ['111111', '222222', '333333']:
    ...     if row_exists(eid, (event_ids, data)):
    ...         print(f"Event {eid} exists")

    >>> # Check in custom sheet
    >>> if row_exists('123456789', sheet_tab='Archive'):
    ...     print("Event is archived")
    """
    eventid_str = str(eventid)

    # If caller passed a pre-fetched spreadsheet tuple
    if isinstance(spreadsheet_or_tab, tuple):
        event_ids, _ = spreadsheet_or_tab
        return any(str(existing_id) == eventid_str for existing_id in event_ids)

    # Otherwise treat it as a sheet_tab name (or None)
    sheet_tab_arg = spreadsheet_or_tab
    row_map = get_event_row_map(sheet_tab=sheet_tab_arg, spreadsheet_id=spreadsheet_id)
    return eventid_str in row_map


def get_column_keys(sheet_tab: str | None = None, spreadsheet_id: str | None = None):
    """Fetch the header row and get column names from the target sheet.

    Returns both the column names directly and a callable function for creating
    index lookup dictionaries.

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab to read from. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    tuple[list[str], callable]
        A tuple of:
        - column_keys : List of header names (strings) from the first row
        - map_func : A callable that returns a dict mapping column names to
          their 0-based column indices

    Examples
    --------
    >>> columns, get_map = get_column_keys()
    >>> print(columns)  # ['EID', 'Status', 'Runtime', 'DM', ...]
    >>> col_map = get_map()
    >>> status_idx = col_map['Status']
    >>> print(f"Status is in column index {status_idx}")
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    sheet = service.spreadsheets()
    result = _execute_with_backoff(
        sheet.values().get(spreadsheetId=sid, range=f"{tab}!1:1")
    )
    column_keys = result.get("values", [])[0] if result.get("values") else []

    def map_func():
        res = {}
        for index, key in enumerate(column_keys):
            res[key] = index
        return res

    return column_keys, map_func


def append_row(
    row_values: dict, sheet_tab: str | None = None, spreadsheet_id: str | None = None
) -> None:
    """Add a new event row to the bottom of the sheet.

    Automatically maps dictionary keys to the correct columns based on the
    header row. Missing columns are left empty.

    Parameters
    ----------
    row_values : dict
        Dictionary mapping column names (from header row) to cell values.
        Keys should match the sheet's column headers exactly. Missing keys
        result in empty cells.
    sheet_tab : str, optional
        The name of the sheet tab to append to. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    None

    Examples
    --------
    >>> append_row({
    ...     'EID': '123456789',
    ...     'Status': 'Pending',
    ...     'Runtime': '',
    ...     'DM': '375.2'
    ... })

    >>> # Append to custom sheet
    >>> append_row(
    ...     {'EID': '987654321', 'Status': 'Complete'},
    ...     sheet_tab='Archive'
    ... )
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    column_keys, _ = get_column_keys(sheet_tab=tab, spreadsheet_id=sid)

    new_row = [row_values.get(key, "") for key in column_keys]

    _execute_with_backoff(
        service.spreadsheets()
        .values()
        .append(
            spreadsheetId=sid,
            range=tab,
            valueInputOption="USER_ENTERED",
            body={"values": [new_row]},
        )
    )


def update_row(
    eventid: str,
    row_values: dict,
    sheet_tab: str | None = None,
    spreadsheet_id: str | None = None,
) -> None:
    """Update specific cells in the row matching the given event ID.

    Only modifies the columns specified in row_values; other columns are
    left unchanged. Column names must match the sheet headers exactly.

    Parameters
    ----------
    eventid : str or int
        The event ID to locate and update. Will be converted to string.
    row_values : dict
        Dictionary mapping column names (from header row) to new values.
        Only specified columns will be updated. Other columns are unchanged.
    sheet_tab : str, optional
        The name of the sheet tab to update. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the event ID does not exist in the sheet.

    Examples
    --------
    >>> # Update status for an event
    >>> update_row('123456789', {'Status': 'In Progress'})

    >>> # Update multiple columns at once
    >>> update_row('123456789', {
    ...     'Status': 'Complete',
    ...     'Runtime': '2.5 hours',
    ...     'Notes': 'Processing finished successfully'
    ... })

    >>> # Update in custom sheet
    >>> update_row('987654321', {'Status': 'Archived'}, sheet_tab='Archive')
    """
    if not row_values:
        return

    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    column_keys, _ = get_column_keys(sheet_tab=tab, spreadsheet_id=sid)
    column_index_map = {key: index for index, key in enumerate(column_keys)}

    row_map = get_event_row_map(sheet_tab=tab, spreadsheet_id=sid)
    eventid_str = str(eventid)
    if eventid_str not in row_map:
        raise ValueError(f"Event ID {eventid} does not exist in the spreadsheet.")

    row_index = row_map[eventid_str]

    data = []
    for key, value in row_values.items():
        col_index = column_index_map.get(key)
        if col_index is None:
            continue

        col_letter = _column_letter(col_index)
        data.append({"range": f"{tab}!{col_letter}{row_index}", "values": [[value]]})

    if not data:
        return

    _execute_with_backoff(
        service.spreadsheets()
        .values()
        .batchUpdate(
            spreadsheetId=sid,
            body={"valueInputOption": "USER_ENTERED", "data": data},
        )
    )


def batch_rows(
    eids: list, sheet_tab: str | None = None, spreadsheet_id: str | None = None
) -> None:
    """Mark multiple events as "In batch processing" in the Status column.

    Efficiently updates the Status column for several events in a single batch
    operation. Safely handles non-existent event IDs by skipping them with a
    warning.

    Parameters
    ----------
    eids : list
        List of event IDs to mark as batch-processing. Can contain strings or
        integers; all will be converted to strings for lookup.
    sheet_tab : str, optional
        The name of the sheet tab to update. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the sheet does not have a 'Status' column.

    Notes
    -----
    This function will print a warning for each event ID that doesn't exist
    in the sheet, but continues processing remaining events.
    Updates are sent in batches of up to 200 rows for efficiency.

    Examples
    --------
    >>> eids_to_process = ['111111', '222222', '333333']
    >>> batch_rows(eids_to_process)

    >>> # Mark events in a custom sheet
    >>> batch_rows(['987654321', '876543210'], sheet_tab='Archive')
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)
    row_map = get_event_row_map(sheet_tab=tab, spreadsheet_id=sid)
    column_keys, _ = get_column_keys(sheet_tab=tab, spreadsheet_id=sid)

    if "Status" not in column_keys:
        raise KeyError("Cannot find 'Status' column in spreadsheet headers.")

    status_col = _column_letter(column_keys.index("Status"))
    data = []

    for eid in eids:
        row_index = row_map.get(str(eid))
        if row_index is None:
            print(f"Event ID {eid} does not exist in the spreadsheet. Skipping...")
            continue

        print(f"Queueing Event ID: {eid}")
        data.append(
            {
                "range": f"{tab}!{status_col}{row_index}",
                "values": [["In batch processing"]],
            }
        )

        # Keep each API call at a manageable size.
        if len(data) >= 200:
            _execute_with_backoff(
                service.spreadsheets()
                .values()
                .batchUpdate(
                    spreadsheetId=sid,
                    body={"valueInputOption": "USER_ENTERED", "data": data},
                )
            )
            data = []

    if data:
        _execute_with_backoff(
            service.spreadsheets()
            .values()
            .batchUpdate(
                spreadsheetId=sid,
                body={"valueInputOption": "USER_ENTERED", "data": data},
            )
        )


def get_info(
    eid, sheet_tab: str | None = None, spreadsheet_id: str | None = None
) -> dict:
    """Retrieve all column data for a specific event as a dictionary.

    Fetches a complete row of data for the given event ID and returns it
    as a dictionary mapping column names to cell values.

    Parameters
    ----------
    eid : str or int
        The event ID to look up. Will be converted to string.
    sheet_tab : str, optional
        The name of the sheet tab to read from. If None, uses the module-level
        default `sheet_tab` variable.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    dict
        Dictionary with column headers as keys and corresponding cell values
        as values. Empty cells are represented as empty strings ('').

    Raises
    ------
    KeyError
        If the event ID does not exist in the sheet.

    Examples
    --------
    >>> info = get_info('123456789')
    >>> print(info['Status'])
    'Complete'
    >>> print(info['Runtime'])
    '2.5 hours'
    >>> print(info)  # See all columns
    {'EID': '123456789', 'Status': 'Complete', 'Runtime': '2.5 hours', ...}

    >>> # Get info from custom sheet
    >>> archived_info = get_info('987654321', sheet_tab='Archive')
    """
    spreadsheet_data = get_spreadsheet(
        sheet_tab=sheet_tab, spreadsheet_id=spreadsheet_id
    )

    eid_str = str(eid)
    try:
        index = next(
            i
            for i, existing_eid in enumerate(spreadsheet_data[0])
            if str(existing_eid) == eid_str
        )
    except StopIteration:
        raise KeyError(f"Cannot find the selected ID: {eid}")

    keys = spreadsheet_data[1][0]
    current_info = spreadsheet_data[1][index + 1]
    # Pad the current info to match the length of keys
    while len(current_info) < len(keys):
        current_info.append("")
    info_dict = {key: current_info[i] for i, key in enumerate(keys)}

    return info_dict


def get_rows_by_fill_color(
    target_rgb: tuple[int, int, int],
    sheet_tab: str | None = None,
    check_column_index: int = 0,
    start_row: int = 2,
    spreadsheet_id: str | None = None,
) -> list[str | int]:
    """Find all event IDs in rows with cells matching a specific background color.

    Retrieves event IDs for rows where a particular cell has the target background
    color. Useful for finding rows that have been manually highlighted or styled
    with conditional formatting. Resolves theme colors and conditional formatting
    to handle dynamic styling.

    Parameters
    ----------
    target_rgb : tuple[int, int, int]
        The target background color in RGB format with values 0-255.
        For example, (255, 0, 0) for red, (0, 255, 0) for green.
    sheet_tab : str, optional
        The name of the sheet tab to search in. If None, uses the module-level
        default `sheet_tab` variable.
    check_column_index : int, optional
        0-based column index to check for the background color. Default is 0,
        which checks column A (typically the event ID column).
    start_row : int, optional
        1-based sheet row number to start searching from. Default is 2, which
        skips the header row. Set to 1 to include headers.
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID. If None, uses the module-level
        default `spreadsheet_id` variable.

    Returns
    -------
    list[str | int]
        List of event IDs (from column A) when checking column A, otherwise
        1-based sheet row numbers for all matching rows. Empty if no rows match
        the target color.

    Notes
    -----
    This function reads effective formatting, which resolves conditional
    formatting and theme colors. It may be slower than other functions due to
    retrieving full formatting information.

    Examples
    --------
    >>> # Find all events highlighted in red
    >>> red_events = get_rows_by_fill_color((255, 0, 0))
    >>> print(f"Found {len(red_events)} red-highlighted events")

    >>> # Find events highlighted in green in a specific column
    >>> green_events = get_rows_by_fill_color(
    ...     (0, 255, 0),
    ...     check_column_index=2,  # Column C
    ...     start_row=2
    ... )

    >>> # Find highlighted events in custom spreadsheet
    >>> custom_events = get_rows_by_fill_color(
    ...     (255, 165, 0),  # Orange
    ...     sheet_tab='Review',
    ...     spreadsheet_id='custom_id_xyz'
    ... )
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)

    # Read values + formatting for the whole tab (A:Z)
    resp = _execute_with_backoff(
        service.spreadsheets().get(
            spreadsheetId=sid,
            ranges=[f"{tab}!A:Z"],
            includeGridData=True,
            fields=(
                "sheets(data(rowData(values("
                "formattedValue,"
                "userEnteredFormat(backgroundColor,backgroundColorStyle),"
                "effectiveFormat(backgroundColor,backgroundColorStyle)"
                "))))"
            ),
        )
    )

    def rgb01_to_255(c: dict) -> tuple[int, int, int]:
        r = int(round(c.get("red", 0.0) * 255))
        g = int(round(c.get("green", 0.0) * 255))
        b = int(round(c.get("blue", 0.0) * 255))
        return (r, g, b)

    matches: list[str | int] = []
    sheets = resp.get("sheets", [])
    if not sheets:
        return matches

    row_data = sheets[0].get("data", [{}])[0].get("rowData", [])
    for i, row in enumerate(row_data, start=1):
        if i < start_row:
            continue

        values = row.get("values", [])
        if check_column_index >= len(values):
            continue

        cell = values[check_column_index]

        # Prefer effective format so conditional formatting/theme resolves.
        fmt = cell.get("effectiveFormat", {}) or cell.get("userEnteredFormat", {})
        color = fmt.get("backgroundColor") or {}
        if not color:
            continue

        if rgb01_to_255(color) == target_rgb:
            if check_column_index == 0:
                # event id from column A
                event_id = values[0].get("formattedValue", "") if values else ""
                matches.append(event_id)
            else:
                matches.append(i)

    return matches


def get_rows_by_strikethrough(
    sheet_tab: str | None = None,
    check_column_index: int = 0,
    start_row: int = 2,
    spreadsheet_id: str | None = None,
) -> list[str | int]:
    """Find all event IDs or row numbers where the text is crossed out (strikethrough).

    Parameters
    ----------
    sheet_tab : str, optional
        The name of the sheet tab to search in.
    check_column_index : int, optional
        0-based column index to check for strikethrough. Default is 0 (Column A).
    start_row : int, optional
        1-based sheet row number to start searching from. Default is 2 (skips headers).
    spreadsheet_id : str, optional
        The Google Sheets spreadsheet ID.

    Returns
    -------
    list[str | int]
        List of event IDs (from column A) when checking column A, otherwise
        1-based sheet row numbers for all matching rows.
    """
    tab, sid = _resolve_sheet_target(sheet_tab, spreadsheet_id)

    # Fetch values + text formatting properties
    resp = _execute_with_backoff(
        service.spreadsheets().get(
            spreadsheetId=sid,
            ranges=[f"{tab}!A:Z"],
            includeGridData=True,
            fields=(
                "sheets(data(rowData(values("
                "formattedValue,"
                "userEnteredFormat(textFormat(strikethrough)),"
                "effectiveFormat(textFormat(strikethrough))"
                "))))"
            ),
        )
    )

    matches: list[str | int] = []
    sheets = resp.get("sheets", [])
    if not sheets:
        return matches

    row_data = sheets[0].get("data", [{}])[0].get("rowData", [])
    for i, row in enumerate(row_data, start=1):
        if i < start_row:
            continue

        values = row.get("values", [])
        if check_column_index >= len(values):
            continue

        cell = values[check_column_index]

        # Check effective format first, fall back to user entered format
        fmt = cell.get("effectiveFormat", {}) or cell.get("userEnteredFormat", {})
        text_fmt = fmt.get("textFormat", {})

        # strikethrough is a boolean (True/False)
        is_crossed_out = text_fmt.get("strikethrough", False)

        if is_crossed_out:
            if check_column_index == 0:
                # Get the event ID from column A
                event_id = values[0].get("formattedValue", "") if values else ""
                matches.append(event_id)
            else:
                # Return the 1-based row number for other columns
                matches.append(i)

    return matches
