import numpy as np

FLOAT_TYPE = np.float32


def format_float(data):
    if data.dtype != FLOAT_TYPE:
        return data.astype(FLOAT_TYPE)
    return data
