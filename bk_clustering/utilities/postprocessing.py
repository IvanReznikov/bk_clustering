import numpy as np


def convert_to_array(lst):
    """
    Convert a list of values to a NumPy array of indices based on a sorted unique set of values.

    Args:
        lst (list): The input list of values to be converted to indices.

    Returns:
        np.ndarray: The NumPy array of indices corresponding to the input list of values.
    """
    value_to_index = {value: idx for idx, value in enumerate(sorted(set(lst)))}
    converted_list = [value_to_index[value] for value in lst]
    return np.array(converted_list)
