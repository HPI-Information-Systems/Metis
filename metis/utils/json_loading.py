import json


def load_json_string_or_path(config: str):
    """
    Load a JSON configuration from a file path or directly from a JSON string.

    :param config: A string that is either a path to a JSON file or a JSON string.
    :return: Any valid JSON value.
    :raises ValueError: If the input string is neither a valid file path nor a valid JSON string.
    """
    try:
        if config.endswith(".json"):
            with open(config, "r") as f:
                return json.load(f)
        else:
            return json.loads(config) if len(config) > 0 else {}
    except Exception as e:
        raise ValueError(f"Failed to load json from {config}: {e}") from e
