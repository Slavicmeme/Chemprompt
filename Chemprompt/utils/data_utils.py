import numpy as np

def ensure_numpy_array(data):
    """
    Converts list data to numpy array if necessary.
    """
    if isinstance(data, list):
        return np.array(data)
    return data
