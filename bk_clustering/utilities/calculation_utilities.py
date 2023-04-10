# explicit function to normalize array
import pandas as pd
import scipy.spatial.distance as distance
import numpy as np


def normalize(arr: pd.Series, t_min: int = 1, t_max: int = 1000) -> pd.Series:
    """Normalize a pandas series to a custom range.

    Args:
        arr (pd.Series): The pandas series to normalize.
        t_min (int): The minimum value of the output range. Defaults to 1.
        t_max (int): The maximum value of the output range. Defaults to 1000.

    Returns:
        pd.Series: The normalized pandas series.

    Raises:
        ValueError: If the input series is empty or the output range is invalid.

    Example:
        >>> series = pd.Series([10, 20, 30, 40, 50])
        >>> normalize(series, 1, 1000)
        [1.0, 250.75, 500.5, 750.25, 1000.0]
    """
    if arr.empty:
        raise ValueError("Input series is empty.")
    if t_min >= t_max:
        raise ValueError("Invalid output range.")

    diff_arr = max(arr) - min(arr)
    if diff_arr:
        diff = t_max - t_min
        normalized_arr = [((i - min(arr)) * diff / diff_arr) + t_min for i in arr]
        return pd.Series(normalized_arr)
    return arr


def get_sq_matrix(data_points):
    """
    Calculates the square matrix of pairwise distances between data points.

    Args:
        data_points (array-like): Input data points as an array-like object.

    Returns:
        np.ndarray: Square matrix of pairwise distances between data points.
    """
    return distance.cdist(data_points, data_points)


def get_distance_matrix(xarr, yarr):
    """
    Calculates the distance matrix and the number of data points.

    Args:
        xarr (array-like): X-coordinates of data points.
        yarr (array-like): Y-coordinates of data points.

    Returns:
        tuple: Tuple containing the distance matrix and the number of data points.
    """
    data_points = np.vstack((xarr, yarr)).T
    sq_matrix = distance.cdist(data_points, data_points)
    return distance.squareform(sq_matrix), len(sq_matrix)
