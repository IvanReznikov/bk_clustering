from scipy.cluster.hierarchy import linkage
from collections import Counter
import pandas as pd
from timeit import default_timer as timer

import sys

# TODO: think of a better way
sys.setrecursionlimit(25000)

# TODO: confidence of the split through sum of solidity/number of points
# TODO: Find dataset with ground truth for mulitlabelling?

from bk_clustering.utilities import (
    calculation_utilities,
    cluster_calculations,
    tree_traversal,
    load_save,
    plot_utilities,
)


class BurjKhalifaClustering:
    def __init__(
        self,
        dH0: float = 0,
        depth: int = 2,
        chain_ratio: float = 5,
        parent_split_ratio: float = 10,
        min_leaves: int = 0,
        n_clusters: int = None,
        linkage=None,
    ):
        self.n_clusters = n_clusters
        self.dH0 = dH0
        self.depth = depth
        self.chain_ratio = chain_ratio
        self.parent_split_ratio = parent_split_ratio
        self.min_leaves = min_leaves
        self.linkage = linkage
        self.linkage_types = ["ward", "single"]

    def fit(
        self,
        X,
    ):
        """
        Wrapper function for clustering algorithm.

        Args:
            X (array-like): Input data for clustering.
            linkage_type (str, optional): Name of the clustering method. Defaults to None.
            min_depth (int, optional): Minimum depth for clustering. Defaults to 2.
            chain_ratio (int, optional): Ratio for chain merge threshold. Defaults to 5.
            parent_split_ratio (int, optional): Ratio for parent split threshold. Defaults to 10.
            min_leaf_ratio (float, optional): Minimum leaf ratio for clustering. Defaults to 0.01.
            number_of_clusters (int, optional): Number of clusters to form. Defaults to None.

        Returns:
            tuple: A tuple containing result_dict, dtf, cluster_df, cluster_info, and node_tree.
        """

        # If linkage_type is not provided, calculate clusters for "ward" and "single" methods
        if not self.linkage:
            method_dict = {}
            for linkage in self.linkage_types:
                method_dict[linkage] = {}
                (
                    result_dict,
                    dtf,
                    cluster_df,
                    cluster_info,
                    node_tree,
                ) = self.calculate_clusters(
                    X,
                    linkage,
                )
                if result_dict:
                    method_dict[linkage]["result_dict"] = result_dict
                    method_dict[linkage]["dtf"] = dtf
                    method_dict[linkage]["cluster_df"] = cluster_df
                    method_dict[linkage]["cluster_info"] = cluster_info
                    method_dict[linkage]["node_tree"] = node_tree
                    method_dict[linkage]["average_solidity"] = sum(
                        [
                            (
                                Counter(result_dict.values())[x]
                                * cluster_info[x]["solidity"]
                            )
                            for x in set(result_dict.values())
                        ]
                    )
                else:
                    method_dict[linkage]["average_solidity"] = float("-inf")

            # Select method with maximum average_solidity
            linkage = max(method_dict.items(), key=lambda k: k[1]["average_solidity"])[
                0
            ]
            result_dict = method_dict[linkage]["result_dict"]
            dtf = method_dict[linkage]["dtf"]
            cluster_df = method_dict[linkage]["cluster_df"]
            cluster_info = method_dict[linkage]["cluster_info"]
            node_tree = method_dict[linkage]["node_tree"]

            self.linkage = linkage
        # If linkage_type is provided, calculate clusters for the given linkage_type
        else:
            (
                result_dict,
                dtf,
                cluster_df,
                cluster_info,
                node_tree,
            ) = self.calculate_clusters(X, linkage_type=self.linkage)

        self.labels_ = list(result_dict.values())
        self.n_clusters = len(set(self.labels_))
        self.dtf_ = dtf
        self.cluster_info_ = cluster_info
        self.cluster_df_ = cluster_df
        self.node_tree_ = node_tree

        return self

    def calculate_clusters(self, X, linkage_type: str):
        """
        #TODO:
        # improve performance to calculate the distance matrix once

        import scipy.spatial.distance as distance
        from . import _hierarchy

        replace with
        n = int(distance.num_obs_y(y))
        method_code = _LINKAGE_METHODS[method]

        if method == 'single':
            result = _hierarchy.mst_single_linkage(y, n)
        elif method in ['complete', 'average', 'weighted', 'ward']:
            result = _hierarchy.nn_chain(y, n, method_code)
        """
        # calculating linkage dataframe
        linkage_df = pd.DataFrame(linkage(X, linkage_type))

        # renaming columns, assigning types
        linkage_df.columns = ["point_1", "point_2", "distance", "points_count"]
        linkage_df[["point_1", "point_2"]] = linkage_df[["point_1", "point_2"]].astype(
            int
        )
        linkage_df["point_id"] = (
            linkage_df.index + linkage_df["points_count"].iloc[-1]
        ).astype(int)

        # defining root_id
        root_id = linkage_df["point_id"].iloc[-1]
        min_leaves = self.min_leaves if self.min_leaves else 1

        # normalizing the distance parameter
        linkage_df["distance"] = calculation_utilities.normalize(linkage_df["distance"])

        # formal density as ratio of the number of data points per distance
        # linkage matrix with formal density calculated as number of data_point/normalized_distance
        linkage_df["formal_density"] = (
            linkage_df["points_count"] / linkage_df["distance"]
        )

        # building a ClusterTree
        node_tree = cluster_calculations.ClusterTree(linkage_df)
        node_tree.build_tree()
        root = node_tree.tree_structure[root_id]

        # creating a ClusterInfoObject object and defining cluster_info
        cluster_info_object = cluster_calculations.ClusterInfoObject(
            linkage_df["points_count"].iloc[-1], linkage_df, self.dH0
        )
        cluster_info_object.set_cluster_info(node_tree.tree_structure)
        cluster_info = cluster_info_object.cluster_info

        # setting leaf_nodes varaible
        leaf_nodes = set(
            cluster_info[x]["node"]
            for x in cluster_info
            if cluster_info[x]["level"] == 0
        )

        # setting node_tree parameters for every node: solidity, chain_solidity, self_cover, chain_cover and children_count
        for n in node_tree.tree_structure:
            node_tree.tree_structure[n].solidity = cluster_info[n]["solidity"]
            node_tree.tree_structure[n].chain_solidity = cluster_info[n]["solidity"]
            node_tree.tree_structure[n].self_cover = False
            node_tree.tree_structure[n].chain_cover = False

        if self.n_clusters:
            nodes_combinations = tree_traversal.get_all_covered_combinations(
                root, self.n_clusters
            )
            pre_cork_leaves = tree_traversal.get_max_solidity(nodes_combinations)

        else:
            tree_traversal.postorder_children_count_traversal(root)

            # update chain_solidity parameter
            tree_traversal.calculate_chain_solidity(node_tree.tree_structure[root_id])

            # get leaves calculated through preorder traversal
            leaves = tree_traversal.preorder_solidity_traversal(
                node_tree.tree_structure[root_id],
                depth=self.depth,
                parent_split_ratio=self.parent_split_ratio,
                chain_ratio=self.chain_ratio,
            )

            # remove leaves that don't match min_leaves condition
            leaves = [leaf for leaf in leaves if leaf.children_count >= min_leaves]

            # find missing corks
            leaves = tree_traversal.bottom_up_cork_leaves(
                leaves, leaf_nodes, node_tree, root_id
            )

            # modify the self_cover attribute for all leaves
            for node in leaves:
                node_tree.tree_structure[node.id].self_cover = True

            # modification of chain_cover attribute
            tree_traversal.postorder_chain_cover_traversal(root)

            # final stage of the end leaves selection
            pre_cork_leaves = set()
            tree_traversal.preorder_cork_traversal(
                root, pre_cork_leaves, min_leaves, self.depth
            )

        cork_nodes = tree_traversal.get_ancestors(leaf_nodes, pre_cork_leaves)

        # getting result as a dictionary {leaf_id:cluster_id}
        result_dict = {i: cork_nodes[i][-1] for i in sorted(list(cork_nodes.keys()))}

        # aggregating all information from the cluster_info object to a pandas dataframe
        cluster_df = pd.DataFrame.from_dict(cluster_info, orient="index")

        return result_dict, linkage_df, cluster_df, cluster_info, node_tree
