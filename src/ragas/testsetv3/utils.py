import numpy as np

MODEL_MAX_LENGTHS = {
    "gpt-3.5-turbo": 16385,
}


rng = np.random.default_rng(seed=42)


def merge_dicts(*dicts):
    merged_dict = {}

    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                elif isinstance(value, str) and isinstance(merged_dict[key], str):
                    merged_dict[key] += "-" + value
                elif isinstance(value, dict) and isinstance(merged_dict[key], dict):
                    merged_dict[key] = merge_dicts(merged_dict[key], value)
                else:
                    raise TypeError("Inconsistent value types for key: {}".format(key))
            else:
                merged_dict[key] = value

    return merged_dict
