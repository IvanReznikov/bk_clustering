from anytree import AnyNode


class ClusterTree:
    def __init__(self, dtf):
        self.dtf = dtf

    def build_tree(self):
        self.tree_structure = {}
        for i, row in self.dtf.sort_values(by="point_id", ascending=False)[
            ["point_1", "point_2", "point_id"]
        ].iterrows():
            p1, p2, pi = row.values
            self.tree_structure[pi] = (
                AnyNode(id=pi)
                if pi not in self.tree_structure
                else self.tree_structure[pi]
            )
            self.tree_structure[p1] = AnyNode(id=p1, parent=self.tree_structure[pi])
            self.tree_structure[p2] = AnyNode(id=p2, parent=self.tree_structure[pi])

    # def add_property(self):


class ClusterInfoObject:
    """
    Class to store node data for clustering algorithm
    """

    def __init__(self, number_of_points, dtf, dH0):
        """
        Initializes the ClusterInfo object and populates it with data
        """
        self.number_of_points = number_of_points
        self.dtf = dtf
        self.dH0 = dH0

        # calculating maximum number of nodes in out tree: initial_nodes + (initial_nodes - 1)
        self.max_number_of_nodes = int(self.number_of_points * 2 - 1)

        # average formal density for bottom leafs
        self.density_0 = self.dtf.loc[
            self.dtf["points_count"] == 2, "formal_density"
        ].mean()

        # creating cluster_info dictionary
        self.cluster_info = {i: {} for i in range(self.max_number_of_nodes)}

    def set_cluster_info(self, tree):
        # setting basic node properties
        for i in self.cluster_info:
            # if node is initial data point
            if i < self.number_of_points:
                self.cluster_info[i]["height"] = 1
                self.cluster_info[i]["count"] = 1
                self.cluster_info[i]["density"] = self.density_0

                self.cluster_info[i]["children"] = []
            else:
                adjusted_i = int(i - self.number_of_points)

                self.cluster_info[i]["height"] = self.dtf.iloc[adjusted_i]["distance"]
                self.cluster_info[i]["count"] = self.dtf.iloc[adjusted_i][
                    "points_count"
                ]
                self.cluster_info[i]["density"] = self.dtf.iloc[adjusted_i][
                    "formal_density"
                ]
                # setting children property of current node
                self.cluster_info[i]["children"] = [
                    int(self.dtf.iloc[adjusted_i]["point_1"]),
                    int(self.dtf.iloc[adjusted_i]["point_2"]),
                ]

        for i in self.cluster_info:
            # setting nodes
            self.cluster_info[i]["node"] = tree[i]

            # setting hierarchical level
            self.cluster_info[i]["level"] = self.cluster_info[i]["node"].height

            # setting parent properties
            self.cluster_info[i]["parent"] = self.get_parent(i)

            # setting sibling properties
            # self.cluster_info[i]["siblings"] = self.get_siblings(i)

            # calculating height to next level
            self.cluster_info[i]["height_to_next"] = self.calculate_height_to_parent(i)

            # calculating solidity of the subcluster as density * height_to_next / height
            self.cluster_info[i]["solidity"] = (
                self.cluster_info[i]["density"]
                * self.cluster_info[i]["height_to_next"]
                / self.cluster_info[i]["height"]
            )

    # TODO remove
    def calculate_d_solidity(self, node):
        """
        Calculates the d_solidity value for a given node
        """
        _arr = [
            self.cluster_info[n]["solidity"]
            for n in self.cluster_info[node]["children"]
        ]
        return self.cluster_info[node]["solidity"] - sum(_arr) / 1

    # TODO remove
    def calculate_d_density(self, node):
        """
        Calculates the d_density value for a given node
        """
        _arr = [
            self.cluster_info[n]["density"] for n in self.cluster_info[node]["children"]
        ]
        return self.cluster_info[node]["density"] - sum(_arr) / 1

    def calculate_height_to_parent(self, node):
        """
        Calculating height to parent
        """
        parent = self.cluster_info[node]["parent"]
        return (
            self.cluster_info.get(parent, {"height": 0})["height"]
            - self.cluster_info[node]["height"]
        )

    def get_parent(self, node):
        """
        Get parent name
        """
        return (
            self.cluster_info[node]["node"].parent.id
            if self.cluster_info[node]["node"].parent
            else "root"
        )

    # replace with siblings attribute in anytree
    def get_siblings(self, node):
        """
        Get sibling.
        """
        if self.cluster_info[node]["parent"] != "root":
            return [
                x
                for x in self.cluster_info[self.cluster_info[node]["parent"]][
                    "children"
                ]
                if x != node
            ]
        else:
            return None
