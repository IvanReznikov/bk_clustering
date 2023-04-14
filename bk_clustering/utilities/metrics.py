from typing import List, Tuple
import numpy as np
from sklearn.metrics import (
    rand_score,
    adjusted_rand_score,
    mutual_info_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score,
)


def calculate_metrics(true_labels, pred_labels):
    """
    Calculate various clustering evaluation metrics based on true and predicted labels.

    Args:
        true_labels (List): List of true labels.
        pred_labels (List): List of predicted labels.

    Returns:
        dict: A dictionary containing various clustering evaluation metrics.
    """
    # Get the mapping of predicted labels to true labels and a boolean array indicating correct predictions
    map_dict, correct_list = get_correct_list(true_labels, pred_labels)

    # Calculate Rand Index
    rand_index = calculate_rand_index(true_labels, pred_labels)

    # Calculate Mutual Similarity
    mutual_similarity = calculate_mutual_similarity(true_labels, pred_labels)

    # Calculate V-Measure
    v_measure = calculate_v_measure(true_labels, pred_labels)

    # Create a dictionary to store the calculated metrics
    errors_dict = {
        "clusters_detected": len(map_dict),
        "clusters_true": len(set(true_labels)),
        "correctly_detected": int(sum(correct_list)),
        "number_of_datapoints": len(correct_list),
        "mutual_similarity_0": mutual_similarity[0],
        "mutual_similarity_1": mutual_similarity[1],
        "mutual_similarity_2": mutual_similarity[2],
        "rand_index_0": rand_index[0],
        "rand_index_1": rand_index[1],
        "v_measure_0": v_measure[0],
        "v_measure_1": v_measure[1],
        "v_measure_2": v_measure[2],
        "fm_score": calculate_fm_score(true_labels, pred_labels),
    }

    return errors_dict


def get_correct_list(true_labels: List, pred_labels: List):
    """
    Get the mapping of predicted labels to true labels and a boolean array indicating correct predictions.

    Args:
        true_labels (List): List of true labels.
        pred_labels (List): List of predicted labels.

    Returns:
        tuple: A tuple containing:
            - map_dict (dict): A dictionary mapping predicted labels to true labels.
            - correct (np.array): A boolean array indicating correct predictions.
    """
    # Initialize a nested dictionary to store counts of true labels for each predicted label
    res = {x: {y: 0 for y in set(true_labels)} for x in set(pred_labels)}

    # Update the counts in the nested dictionary based on the true and predicted labels
    for r, v in zip(pred_labels, true_labels):
        res[r][v] += 1

    # Get the mapping of predicted labels to true labels by selecting the label with the highest count
    map_dict = {r: max(res[r], key=res[r].get) for r in res}

    # Create a boolean array indicating correct predictions by comparing predicted labels with mapped true labels
    correct = np.array(true_labels) == np.array([map_dict[x] for x in pred_labels])

    return map_dict, correct


def calculate_v_measure(
    true_labels: List, pred_labels: List, beta: float = 1.0
) -> Tuple[float, float, float]:
    """
    The V-measure metric evaluates the accuracy of cluster assignments using conditional entropy analysis.
    A higher score indicates a higher degree of similarity.
    The V-measure metric comprises two intuitive sub-metrics that emerge from supervised learning:
        Homogeneity: This sub-metric evaluates whether each cluster comprises only members of a single class
            (similar to "precision").
        Completeness: This sub-metric assesses whether all members of a specific class are assigned to the
            same cluster (similar to "recall").

    Use:
    - When you require interpretability since V-measure is intuitive and easy to comprehend concerning
        homogeneity and completeness.
    - When you are uncertain about the cluster structure, since V-measure makes no assumptions about the
        cluster structure and is adaptable to all clustering algorithms.
    - When you want to establish a basis for comparison since homogeneity, completeness, and V-measure
        are bounded between [0, 1]. This bounded range simplifies score comparisons across various algorithms.
    Not Use:
    - When you don't have the ground truth labels since homogeneity, completeness, and V-measure are extrinsic
        measures and necessitate ground truth cluster assignments.
    - When your sample size is less than 1000 and the number of clusters is over 10 since V-measure doesn't
        adjust for chance. This implies that random labelling may not produce zero scores, especially when the
        number of clusters is large.

    Args:
        true_labels (List): True labels
        pred_labels (List): Predicted labels
        beta (float, optional): Ratio of weight attributed to homogeneity vs completeness.
            beta > 1, completeness is weighted more strongly in the calculation.
            beta < 1, homogeneity is weighted more strongly.
            Defaults to 1.0.

    Returns:
        Tuple[float, float, float]: homogeneity, completeness, v_measure
    """
    hs = homogeneity_score(true_labels, pred_labels)
    cs = completeness_score(true_labels, pred_labels)
    v = v_measure_score(true_labels, pred_labels, beta=beta)
    return hs, cs, v


def calculate_mutual_similarity(
    true_labels: List, pred_labels: List
) -> Tuple[float, float, float]:
    """
    The Mutual Information (MI), Normalized Mutual Information (NMI), and Adjusted Mutual Information (AMI)
    measures quantify the degree of agreement between cluster assignments.
    A larger score indicates greater similarity between the clusters.

    Use:
    - When you want to establish a basis for comparison.
    - When you want to compare MI, NMI, and AMI with an upper bound of 1.
    Not Use:
    - When you don't have the ground truth labels.

    Args:
        true_labels (List): True labels
        pred_labels (List): Predicted labels

    Returns:
        Tuple[float, float, float]: mutual_score, normalized_mutual_score, adjusted_mutual_score
    """
    mi = mutual_info_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(
        true_labels, pred_labels, average_method="geometric"
    )
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    return mi, nmi, ami


def calculate_rand_index(true_labels: List, pred_labels: List) -> Tuple[float, float]:
    """Rand Index (RI, ARI) measures the similarity between the cluster assignments by making pair-wise comparisons.
    A higher score signifies higher similarity.

    Use:
    - When you want interpretability
    - When you are unsure about cluster structure
    - When you want a basis for comparison
    Not Use:
    - When you do not have the ground truth labels.

    Args:
        true_labels (List): True labels
        pred_labels (List): Predicted labels

    Returns:
        Tuple[float, float]: rank index and adjusted rank index metrics
    """
    ri = rand_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    return ri, ari


def calculate_fm_score(true_labels: List, pred_labels: List) -> float:
    """The Fowlkes-Mallows scores quantify the accuracy of the cluster assignments using pairwise precision and recall.
    A higher score indicates a greater degree of similarity.

    Use:
    - When you are uncertain about the cluster structure since the Fowlkes-Mallows Score does not assume any specific
        cluster structure and can be used with any clustering algorithm.
    - When you need to establish a basis for comparison since the Fowlkes-Mallows Score has a maximum limit of 1.
        This restricted range simplifies the comparison of scores across various algorithms.
    Not Use:
    - When you don't have the ground truth labels since the Fowlkes-Mallows Scores are extrinsic measures that need
        ground truth cluster assignments.

    Args:
        true_labels (List): True labels
        pred_labels (List): Predicted labels

    Returns:
        float: Fowlkes Mallows score
    """
    return fowlkes_mallows_score(true_labels, pred_labels)
