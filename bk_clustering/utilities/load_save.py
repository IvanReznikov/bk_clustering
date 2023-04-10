import numpy as np
import pandas as pd
from scipy.io import arff as scipy_arff
from bk_clustering.utilities import preprocessing
import ujson


def read_arff(folder, filename, skip_columns = []):
    """
    Read an ARFF file from a specified folder and filename, and preprocess the data.

    Args:
        folder (str): The folder containing the ARFF file.
        filename (str): The filename of the ARFF file without the extension.

    Returns:
        pd.DataFrame: The preprocessed data as a pandas DataFrame, or None if the file is not found.
    """
    try:
        # Load ARFF data using scipy_arff and create a pandas DataFrame
        data = scipy_arff.loadarff(f"../data/{folder}/{filename}.arff")
        df = pd.DataFrame(data[0])
        df.columns = df.columns.str.lower()  # Convert column names to lowercase
    except FileNotFoundError:
        print(f"Dataset {folder}/{filename} not present")
        return None  # Return None if the file is not found
    for col in skip_columns:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = preprocessing.preprocessing(
        df
    )  # Preprocess the data using a preprocessing function
    return df  # Return the preprocessed data as a DataFrame


'''
def read_dataset(filename: str, train=True, validation=True):
    """
    filename -- dataset name. Example: a1,a2,s1,etc
    """
    xarr, yarr, validation_list = None, None, None
    relative_dir = "/"
    if train:
        f = open(f".{relative_dir}data/benchmarks/data/{filename}.txt", "r")
        data_points = np.array(
            [[float(_dp) for _dp in x.split(" ") if len(_dp) > 0] for x in f]
        )
        xarr = data_points[:, 0]
        yarr = data_points[:, 1]
    if validation:
        f = open(f".{relative_dir}data/benchmarks/validation/{filename}.pa", "r")
        validation_list = [int(x.replace("\n", "")) for x in f]
    return xarr, yarr, validation_list
'''


def save_json(data, filename):
    """
    Save data to a JSON file.

    Args:
        data (dict): The data to be saved as JSON.
        filename (str): The filename of the JSON file to be created.

    Returns:
        None
    """
    with open(filename, "w") as outfile:
        ujson.dump(data, outfile)  # Use ujson.dump to write data to the JSON file


def load_json(filename):
    with open(filename) as outfile:
        data = ujson.load(outfile)
        return data


def format_results(results: dict, time: float) -> dict:
    """
    Format the results dictionary by adding the performance time.

    Args:
        results (dict): The results dictionary.
        time (float): The performance time to be added to the results.

    Returns:
        dict: The formatted results dictionary with the performance time added.
    """
    obj = {**results}  # Create a shallow copy of the results dictionary
    obj["performance_s"] = time  # Add the performance time to the results
    return obj
