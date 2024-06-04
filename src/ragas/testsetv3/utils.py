
MODEL_MAX_LENGTHS = {
    "gpt-3.5-turbo": 16385,
}
def merge_dicts(*dicts):
    merged_dict = {}
    
    for d in dicts:
        for key, value in d.items():
            if key in merged_dict:
                if isinstance(value, list) and isinstance(merged_dict[key], list):
                    merged_dict[key].extend(value)
                elif isinstance(value, str) and isinstance(merged_dict[key], str):
                    merged_dict[key] += '-' + value
                else:
                    raise TypeError("Inconsistent value types for key: {}".format(key))
            else:
                if isinstance(value, list):
                    merged_dict[key] = list(value)
                elif isinstance(value, str):
                    merged_dict[key] = value
                else:
                    raise TypeError("Unsupported value type for key: {}".format(key))
                    
    return merged_dict