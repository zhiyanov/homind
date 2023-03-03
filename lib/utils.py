import numpy as np

FLOAT_TYPE = np.float64
INT_TYPE = np.int64


def format_float(data):
    if data.dtype != FLOAT_TYPE:
        return data.astype(FLOAT_TYPE)
    return data

def format_int(data):
    if data.dtype != INT_TYPE:
        return data.astype(INT_TYPE)
    return data
