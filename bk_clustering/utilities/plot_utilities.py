from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import scipy.cluster.hierarchy as shc
from typing import List
from matplotlib.colors import ListedColormap
from bk_clustering.utilities import metrics
from anytree import RenderTree


def get_scatter(xarr: List, yarr: List, figsize=(16, 10), **kwargs):
    """Draw scatter plot

    Args:
        _xarr (List): x coordinates
        _yarr (List): y coordinates
        figsize (tuple, optional): _description_. Defaults to (16, 10).

    Returns:
        _type_: _description_
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(xarr, yarr, **kwargs)
    plt.legend(*scatter.legend_elements())
    return plt


def get_heatmap(matrix, figsize=(15, 12)):
    """Draw heatmap

    Args:
        matrix (_type_): nxn list
        figsize (tuple, optional): _description_. Defaults to (15, 12).

    Returns:
        _type_: sns.heatmap
    """
    fig, ax = plt.subplots(figsize=figsize)
    return sns.heatmap(
        matrix,
        ax=ax,
        annot=True,
    )


def get_error_plots(_xarr: List, _yarr: List, true_labels: List, pred_labels: List):
    """Calculating accuracy of predicted labels.
    1. Map closest predicted cluster with ground true cluster
    2. Calculate accuracy by finding how many data points outside the max

    Args:
        _xarr (List): x coordinates
        _yarr (List): y coordinates
        true_labels (List): True labels
        pred_labels (List): Predicted labels

    Returns:
        Tuple[int, int, float]: number of correct predictions, number of predicted classes, accuracy
    """

    map_dict, correct_list = metrics.get_correct_list(true_labels, pred_labels)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    # define custom color map - need to handle 100% of fails
    cmap = ListedColormap([[1, 0, 0], [0, 1, 0]])
    if sum(correct_list) == len(_xarr):
        cmap = ListedColormap([[0, 1, 0], [1, 0, 0]])
    axs[0, 0].scatter(_xarr, _yarr)
    axs[0, 0].set_title("initial")
    axs[0, 1].scatter(_xarr, _yarr, c=true_labels)
    axs[0, 1].set_title("ground truth")
    axs[1, 0].scatter(_xarr, _yarr, c=pred_labels)
    axs[1, 0].set_title("proposed method")
    axs[1, 1].scatter(_xarr, _yarr, c=correct_list, cmap=cmap)
    axs[1, 1].set_title("error points")
    return axs


def augmented_dendrogram(
    dtf, limit=20, ids=[], positive_ids=[], log=False, figsize=(14, 10), *args, **kwargs
):
    if dtf.shape[1] > 4:
        _dtf = dtf.iloc[:, :4]
    else:
        _dtf = dtf.copy()
    if log == True:
        _dtf["distance"] = _dtf["distance"].apply(np.log)
    ddata = get_dendrogram(_dtf, figsize=figsize, **kwargs)
    h_mapping = {}
    if "point_id" in dtf.columns:
        h_mapping = {
            x[0]: x[1]
            for x in dtf.sort_values("distance", ascending=False)[
                ["distance", "point_id"]
            ]
            .iloc[:limit]
            .to_dict("split")["data"]
        }

    if not kwargs.get("no_plot", False):
        for idx, (i, d) in enumerate(zip(ddata["icoord"], ddata["dcoord"])):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if ids:
                if ids[idx] in positive_ids:
                    plt.plot(x, y, "go")
                else:
                    plt.plot(x, y, "ro")
            if y in h_mapping:
                plt.annotate(
                    h_mapping[y],
                    (x, y),
                    xytext=(10, 15),
                    textcoords="offset points",
                    va="top",
                    ha="center",
                )
    return ddata


def get_dendrogram(dtf, title=None, figsize=(14, 10), **kwargs):
    """Draw dendrogram

    Args:
        dtf (_type_): modified df with additional d parameter calculated
        figsize (tuple, optional): _description_. Defaults to (14, 10).

    Returns:
        _type_: _description_
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    return shc.dendrogram(dtf, **kwargs)


def draw_tree(tree, maxlevel=5):
    """
    Draw a tree structure using ASCII art representation.

    Args:
        tree: Tree object
            The tree structure to be drawn.
        maxlevel: int, optional (default=5)
            The maximum level of the tree to be drawn.

    Returns:
        None
    """
    for pre, fill, node in RenderTree(tree, maxlevel=maxlevel):
        # Print the tree node's id, indented with pre and fill characters
        print("%s%s" % (pre, node.id))
