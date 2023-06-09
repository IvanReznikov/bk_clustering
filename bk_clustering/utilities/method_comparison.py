from timeit import default_timer as timer
from typing import List
import numpy as np
from utilities import load_save, metrics, density_peak
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AffinityPropagation,
    AgglomerativeClustering,
    Birch,
    DBSCAN,
    MeanShift,
    OPTICS,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
import hdbscan


def get_data(dataset_name: str):
    """
    Load and preprocess a dataset from an ARFF file.

    Args:
        dataset_name (tuple): A tuple containing the name of the ARFF file and the name of the class attribute.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the true labels (true_labels).
    """
    # Load the ARFF file using the specified dataset_name
    df = load_save.read_arff(dataset_name[0], dataset_name[1])

    # Extract the feature matrix (X) and the true labels from the loaded ARFF file
    X, true_labels = df.loc[:, df.columns != "class"], df["class"]

    return X, true_labels


def collect_all_results(
    results: dict, dataset_name: str, true_labels: List, model, start
):
    """
    Collect and store the results of a model's performance on a dataset.

    Args:
        results (dict): A dictionary containing the results of multiple models on different datasets.
        dataset_name (str): The name of the dataset.
        true_labels (array-like): The true labels of the dataset.
        model (object): The trained model.
        start (float): The starting time when the model evaluation began.

    Returns:
        dict: The updated results dictionary with the collected performance metrics.
    """
    # Calculate the error metrics for the model's predicted labels
    error_results = metrics.calculate_metrics(true_labels, model.labels_)

    # Format the error results and store them in the results dictionary along with the elapsed time
    results[dataset_name] = load_save.format_results(error_results, timer() - start)

    return results


# TODO: itterate kmeans through different random seeds and check if some our outliered?
# TODO: check how the results change with number of clusters mistaken
def run_kmeans(dataset_names: List, number_of_clusters: List, random_state: int = 1):
    """
    Run the K-means clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.
        random_state (int): An optional random seed for reproducibility. Default is 0.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(
            X
        )  # Train the K-means model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/k_means_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_kmeans_mini_batch(
    dataset_names: List, number_of_clusters: List, random_state: int = 1
):
    """
    Run the Mini-Batch K-means clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.
        random_state (int): An optional random seed for reproducibility. Default is 0.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state).fit(
            X
        )  # Train the Mini-Batch K-means model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = f"./../results/mini_batch_kmeans_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_affinity_propagation(
    dataset_names: List, number_of_clusters: List, random_state: int = 1
):
    """
    Run the Affinity Propagation clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.
        random_state (int): An optional random seed for reproducibility. Default is 0.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = AffinityPropagation(random_state=random_state).fit(
            X
        )  # Train the Affinity Propagation model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/affinity_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_agglomerative(dataset_names: List, number_of_clusters: List):
    """
    Run the Agglomerative Clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = AgglomerativeClustering(n_clusters=n_clusters).fit(
            X
        )  # Train the Agglomerative Clustering model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = f"./../results/agglomerative_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_birch(dataset_names, number_of_clusters):
    """
    Run the Birch Clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = Birch(n_clusters=n_clusters).fit(X)  # Train the Birch Clustering model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/birch_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_dbscan(dataset_names: List, number_of_clusters: List):
    """
    Run the DBSCAN clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = DBSCAN().fit(X)  # Train the DBSCAN model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/dbscan_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_mean_shift(dataset_names: List, number_of_clusters: List):
    """
    Run the Mean Shift clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = MeanShift().fit(X)  # Train the Mean Shift model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = f"./../results/mean_shift_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_optics(dataset_names: List, number_of_clusters: List):
    """
    Run the OPTICS clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = OPTICS().fit(X)  # Train the OPTICS model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/optics_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_spectral(dataset_names: List, number_of_clusters: List):
    """
    Run the Spectral Clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        model = SpectralClustering(n_clusters=n_clusters).fit(
            X
        )  # Train the Spectral Clustering model
        results = collect_all_results(
            results, dataset_name, true_labels, model, start
        )  # Collect and store results

    filename = (
        f"./../results/spectral_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_gmm(dataset_names: List, number_of_clusters: List, random_state=1):
    """
    Run the Gaussian Mixture Model (GMM) clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.
        random_state (int): An optional random seed for reproducibility. Default is 1.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        # as it was discovered for some datasets ("jm1", "KDDTest+", etc.) GaussianMixture throws an error
        try:
            start = timer()  # Record the start time of model evaluation
            X, true_labels = get_data(
                dataset_name
            )  # Load the dataset and extract true labels
            model = GaussianMixture(
                n_components=n_clusters, random_state=random_state
            ).fit(
                X
            )  # Train the GMM model
            pred_labels = model.predict(X)  # Predict the cluster labels
            error_results = metrics.calculate_metrics(
                true_labels, pred_labels
            )  # Calculate error metrics
            results[dataset_name] = load_save.format_results(
                error_results, timer() - start
            )  # Collect and store results
        except Exception as e:
            print(e)  # Print any exception that occurs during model evaluation

    filename = f"./../results/gaussian_mixture_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_density_peak(dataset_names: List, number_of_clusters: List):
    """
    Run the Density Peak clustering algorithm on multiple datasets and collect and save the results.

    Args:
        dataset_names (list): A list of dataset names.
        number_of_clusters (list): A list of the number of clusters to use for each dataset.

    Returns:
        None
    """
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        start = timer()  # Record the start time of model evaluation
        X, true_labels = get_data(
            dataset_name
        )  # Load the dataset and extract true labels
        pred_labels = density_peak.run_density_peak(
            X
        )  # Run the Density Peak algorithm to get predicted labels
        error_results = metrics.calculate_metrics(
            true_labels, pred_labels
        )  # Calculate clustering metrics
        results[dataset_name] = load_save.format_results(
            error_results, timer() - start
        )  # Collect and store results

    filename = f"./../results/density_peak_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file


def run_hdbscan(dataset_names: List, number_of_clusters: List, random_state=1):
    results = {}  # Initialize an empty dictionary to store the results

    # Loop through each dataset and corresponding number of clusters
    for dataset_name, n_clusters in zip(dataset_names, number_of_clusters):
        try:
            start = timer()  # Record the start time of model evaluation
            X, true_labels = get_data(
                dataset_name
            )  # Load the dataset and extract true labels

            model = hdbscan.HDBSCAN()
            model.fit(X)  # Train the hdbscan model
            pred_labels = model.labels_  # Cluster labels
            error_results = metrics.calculate_metrics(
                true_labels, pred_labels
            )  # Calculate error metrics
            results[dataset_name] = load_save.format_results(
                error_results, timer() - start
            )  # Collect and store results
        except Exception as e:
            print(e)  # Print any exception that occurs during model evaluation

    filename = (
        f"./../results/hdbscan_results.json"  # Define the filename to save the results
    )
    load_save.save_json(results, filename)  # Save the results to a JSON file


def generate_corr_matrix(dataset_size_dict):
    import itertools
    from collections import OrderedDict
    from bk_clustering import postprocessing, metrics, load_save

    filename = "./../results/predictions/bk_clustering_predictions.json"
    bk_clustering_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/k_means_predictions.json"
    k_means_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/mini_batch_kmeans_predictions.json"
    mini_batch_kmeans_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/affinity_predictions.json"
    affinity_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/agglomerative_predictions.json"
    agglomerative_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/birch_predictions.json"
    birch_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/hdbscan_predictions.json"
    hdbscan_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/dbscan_predictions.json"
    dbscan_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/mean_shift_predictions.json"
    mean_shift_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/optics_predictions.json"
    optics_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/gaussian_mixture_predictions.json"
    gaussian_mixture_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    filename = "./../results/predictions/density_peak_predictions.json"
    density_peak_predictions = {
        eval(k): v
        for (k, v) in OrderedDict(sorted(load_save.load_json(filename).items())).items()
    }

    prediction_dict = {
        "bk_clustering": [
            postprocessing.convert_to_array(eval(x))
            for x in bk_clustering_predictions.values()
        ],
        "kmeans": [eval(x) for x in k_means_predictions.values()],
        "affinity": [eval(x) for x in affinity_predictions.values()],
        "dbscan": [eval(x) for x in dbscan_predictions.values()],
        "hdbscan": [eval(x) for x in hdbscan_predictions.values()],
        "density_peak": [eval(x) for x in density_peak_predictions.values()],
        "mean_shift_predictions": [eval(x) for x in mean_shift_predictions.values()],
        "optics": [eval(x) for x in optics_predictions.values()],
        "agglomerative": [eval(x) for x in agglomerative_predictions.values()],
        "birch": [eval(x) for x in birch_predictions.values()],
        "gmm": [eval(x) for x in gaussian_mixture_predictions.values()],
        "mb_kmeans": [eval(x) for x in mini_batch_kmeans_predictions.values()],
    }

    raw_metrics_dict = {}
    for key in list(itertools.combinations(prediction_dict, 2)):
        raw_metrics_dict[key] = {}
        raw_metrics_dict[key]["v_measure"] = np.array(
            [
                list(
                    metrics.calculate_v_measure(ds1, ds2)
                    for ds1, ds2 in zip(
                        prediction_dict[key[0]], prediction_dict[key[1]]
                    )
                )
            ]
        )
        raw_metrics_dict[key]["mutual_similarity"] = np.array(
            [
                list(
                    metrics.calculate_mutual_similarity(ds1, ds2)
                    for ds1, ds2 in zip(
                        prediction_dict[key[0]], prediction_dict[key[1]]
                    )
                )
            ]
        )
        raw_metrics_dict[key]["rand_index"] = np.array(
            [
                list(
                    metrics.calculate_rand_index(ds1, ds2)
                    for ds1, ds2 in zip(
                        prediction_dict[key[0]], prediction_dict[key[1]]
                    )
                )
            ]
        )
        raw_metrics_dict[key]["fm_score"] = np.array(
            [
                list(
                    metrics.calculate_fm_score(ds1, ds2)
                    for ds1, ds2 in zip(
                        prediction_dict[key[0]], prediction_dict[key[1]]
                    )
                )
            ]
        )

    final_metrics_dict = {}
    for key in list(itertools.combinations(prediction_dict, 2)):
        final_metrics_dict[key] = {}
        final_metrics_dict[key]["v_measure_0"] = raw_metrics_dict[key]["v_measure"][
            :, :, 0
        ][0]
        final_metrics_dict[key]["v_measure_1"] = raw_metrics_dict[key]["v_measure"][
            :, :, 1
        ][0]
        final_metrics_dict[key]["v_measure_2"] = raw_metrics_dict[key]["v_measure"][
            :, :, 2
        ][0]
        final_metrics_dict[key]["mutual_similarity_0"] = raw_metrics_dict[key][
            "mutual_similarity"
        ][:, :, 0][0]
        final_metrics_dict[key]["mutual_similarity_1"] = raw_metrics_dict[key][
            "mutual_similarity"
        ][:, :, 1][0]
        final_metrics_dict[key]["mutual_similarity_2"] = raw_metrics_dict[key][
            "mutual_similarity"
        ][:, :, 2][0]
        final_metrics_dict[key]["rand_index_0"] = raw_metrics_dict[key]["rand_index"][
            :, :, 0
        ][0]
        final_metrics_dict[key]["rand_index_1"] = raw_metrics_dict[key]["rand_index"][
            :, :, 1
        ][0]
        final_metrics_dict[key]["fm_score"] = raw_metrics_dict[key]["fm_score"][:][0]

    json_dict = {k: {} for k in final_metrics_dict}
    for k in final_metrics_dict:
        for m in final_metrics_dict[k]:
            json_dict[k][m] = load_save.encode_json(
                {
                    dataset: metric
                    for (dataset, metric) in zip(
                        dataset_size_dict, final_metrics_dict[k][m]
                    )
                }
            )

    load_save.save_json(json_dict, "./../results/aggregations/aggregated.json")
