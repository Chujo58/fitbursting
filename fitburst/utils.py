import json, argparse, os, logging, numpy

logger = logging.getLogger(__name__)


def load_parser_from_json(json_file):
    with open(json_file, "r") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(description=config.get("description", ""))

    for arg in config.get("arguments", []):
        name = arg.pop("name")
        # Convert the string representation of the type to an actual Python type
        arg_type_str = arg.get("type")  # Get the type as a string from JSON
        if arg_type_str == "str":
            arg["type"] = str
        elif arg_type_str == "int":
            arg["type"] = int
        elif arg_type_str == "float":
            arg["type"] = float
        # Handle boolean types
        elif arg_type_str == "bool":
            arg["type"] = bool  # Note: argparse doesn't directly support type=bool
            # You'll likely need to use action='store_true' or a custom type converter
        elif "type" in arg:
            del arg["type"]  # Remove the type if it's not a standard type

        parser.add_argument(name, **arg)

    return parser


def rename_file(original, new):
    try:
        os.rename(original, new)
        logger.info(f"File '{original}' successfully renamed to '{new}'.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File {original} not found.")
    except OSError as e:
        logger.error(f"Error renaming file: {e}")


def format_dict_for_json(data):
    def convert_value(val):
        if isinstance(val, numpy.integer):
            return int(val)
        elif isinstance(val, numpy.floating):
            return float(val)
        elif isinstance(val, numpy.ndarray):
            return convert_value(val.tolist())
        elif isinstance(val, dict):
            return {k: convert_value(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [convert_value(v) for v in val]
        else:
            return val

    return convert_value(data)