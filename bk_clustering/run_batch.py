from utilities import load_save, metrics #method_comparison
import os
from timeit import default_timer as timer
from main import BurjKhalifaClustering

SKIP_COLUMNS_IN_DATASET = {
    ("real", "wdbc"): ["idnumber"],
    ("real", "spectrometer"): ["LRS-name"],
}


def run_multiple_datasets(
    dataset_names,
    depth=2,
    chain_ratio=5,
    parent_split_ratio=10,
    min_leaves=0,
    n_clusters=None,
):
    result_dict = {}
    for idx, dataset_name in enumerate(dataset_names):
        try:
            start = timer()
            skip_columns = (
                []
                if dataset_name not in SKIP_COLUMNS_IN_DATASET
                else SKIP_COLUMNS_IN_DATASET[dataset_name]
            )
            df = load_save.read_arff(dataset_name[0], dataset_name[1], skip_columns)
            X, true_labels = df.loc[:, df.columns != "class"], df["class"]
            bk_model = BurjKhalifaClustering(
                depth=depth,
                chain_ratio=chain_ratio,
                parent_split_ratio=parent_split_ratio,
                min_leaves=min_leaves,
                n_clusters=n_clusters,
                linkage="ward",
            )
            bk_model.fit(X)
            predict_labels = bk_model.labels_
            error_results = metrics.calculate_metrics(true_labels, predict_labels)
            result_dict[dataset_name] = load_save.format_results(
                error_results, timer() - start
            )
        except Exception as e:
            print(e)

    return result_dict


def run_batch():
    skip_datasets = [
        "water-treatment",  # no class
        "autos",
        "credit.a",  # duplicate dataset
        "credit.g",  # duplicate dataset
        "sick",  # duplicate dataset
        "golfball",  # as 1 cluster, incorrect metric definition for clustering methods
        "Colon",  # multiple duplicated column names
        "jm1",  # gmm throws error
        "KDDTest+",  # gmm throws error
        "Rice_MSC_Dataset",  # run separately
        "click_data",  # takes forever long for kmeans?
    ]
    folders = ["real", "artificial"]

    dataset_names = []
    for folder in folders:
        dataset_names += [
            (folder, x[:-5])
            for x in os.listdir(f"./../data/{folder}")
            if x[:-5] not in skip_datasets
        ]

    number_of_clusters = [
        load_save.read_arff(x[0], x[1]).iloc[:, -1].nunique() for x in dataset_names
    ]

    # run bk_clustering
    """
    results = run_multiple_datasets(dataset_names)
    filename = f"./../results/bk_clustering_results.json"  # Define the filename to save the results
    load_save.save_json(results, filename)  # Save the results to a JSON file
    """
    # method_comparison.run_birch(dataset_names, number_of_clusters)
    # method_comparison.run_dbscan(dataset_names, number_of_clusters)
    # method_comparison.run_kmeans(dataset_names, number_of_clusters)
    # method_comparison.run_kmeans_mini_batch(dataset_names, number_of_clusters)
    # method_comparison.run_mean_shift(dataset_names, number_of_clusters)
    # method_comparison.run_agglomerative(dataset_names, number_of_clusters)
    # method_comparison.run_gmm(dataset_names, number_of_clusters)
    # method_comparison.run_affinity_propagation(dataset_names, number_of_clusters)
    # method_comparison.run_density_peak(dataset_names, number_of_clusters)
    # method_comparison.run_optics(dataset_names, number_of_clusters)
    # method_comparison.run_hdbscan(dataset_names, number_of_clusters)


# python3 run_batch.py
if __name__ == "__main__":
    run_batch()
