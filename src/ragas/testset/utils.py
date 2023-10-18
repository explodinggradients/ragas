import json
import warnings


def load_as_json(text):
    try:
        return json.loads(text)
    except ValueError:
        warnings.warn("Invalid json")

    return {}
