from collections import Counter
from typing import List


def preorder_solidity_traversal(
    node, depth: int = 2, chain_ratio: float = 4, parent_split_ratio: float = 10
) -> List:
    """
    The aim of this method is to return possible unique list of nominates for being the final nodes
    (pseudo-roots for stable solid clusters)
    """

    result = []
    """
    If a node has children, we first check if the chain.solidity is ratio higher than own solidity.
    If that is the case - the node is "weak" and we should nominate children as final nodes (result list).

    We recursively check if children's solidity is higher than parents. If such a case, we enter recursion
    and add the result to our results list.
    Contrary, if there are no children more solid than the parent - we add the current parent node.
    """
    if node.children:
        if node.chain_solidity > parent_split_ratio * node.solidity:
            for child in node.children:
                result += [child]

        add = True
        if node.children[0].solidity >= node.solidity:
            result += preorder_solidity_traversal(
                node.children[0],
                depth=depth,
                chain_ratio=chain_ratio,
                parent_split_ratio=parent_split_ratio,
            )
            add = False
        if node.children[1].solidity >= node.solidity:
            result += preorder_solidity_traversal(
                node.children[1],
                depth=depth,
                chain_ratio=chain_ratio,
                parent_split_ratio=parent_split_ratio,
            )
            add = False
        if add:
            result += [node]

    """
    There are usual cases, when *grand children are more solid than their *(grand)parents.
    For this case we implement depth parameter. depth = 2, allows you to check the depth of 2 generations,
    aka grandchildren.
    For *grandchildren we check the following conditions:
    a. grandchild.solidity and grandchild.chain_solidity / grandchild.solidity -- if the chain solidity of grandchild
        is high -- we continue the dig and add recursive result
    b. grandchild.solidity and grandchild.solidity >= node.solidity -- if the solidity of the grandchild is higher than
        the nodes' -- we continue the dig and add recursive result
    
    else the current node is added to the final result
    """
    if depth >= 1 and node.children:
        for child in node.children:
            if child.children:
                for grandchild in child.children:
                    if grandchild.solidity and grandchild.solidity >= node.solidity:
                        result += preorder_solidity_traversal(
                            grandchild, depth=depth - 1, chain_ratio=chain_ratio
                        )
                    else:
                        result += [node]

    # the end result might have nodes added by different conditions - so we return a unique list
    return list(set(result))


def bottom_up_cork_leaves(nodes: List, basic_leaves: set, node_tree, root_id: int):
    """
    The aim of this method is to find possible "corks" to make the whole tree full covered.
    TODO: check if a parent cork is better than current or if children is better than current cork
    TODO: define a condition for above TODO

    The idea is to check all the ancestors leaf-to-root for all basic_leafs (get_ancestors() method).
    If the root_id is present in the basic_ancestors set -- that means not all ground leaves are covered.
    The leaves_ancestors list is created to eliminate intermediate stable nodes and the corks dict is
    sorted by the frequency of node appearance and it's level (value).
    While corks object has nodes we add the top leaf to our nodes list, remove all descendants of the added
    cork and check the corks dict yet again.

    Args:
        nodes (List): _description_
        basic_leaves (set): _description_
        node_tree (_type_): _description_
        root_id (int): id of the root element

    Returns:
        _type_: _description_
    """
    root_id = node_tree.tree_structure[0].root.id if not root_id else root_id
    basic_ancestors = get_ancestors(basic_leaves, nodes)
    basic_ancestors_set = set([e for l in list(basic_ancestors.values()) for e in l])

    if root_id in basic_ancestors_set:
        leaves_ancestors = get_ancestors(nodes)
        leaves_ancestors_set = set(
            [e for l in list(leaves_ancestors.values()) for e in l]
            + list(leaves_ancestors.keys())
        )
        corks = {
            k: v
            for k, v in Counter(
                [e for l in list(basic_ancestors.values()) for e in l]
            ).items()
            if k not in leaves_ancestors_set
        }

        while corks:
            new_leaf = sorted(corks.items(), key=lambda x: (x[1], -x[0]), reverse=True)[
                0
            ][0]
            nodes.append(node_tree.tree_structure[new_leaf])
            descendants_list = [
                x.id for x in node_tree.tree_structure[new_leaf].descendants
            ] + [new_leaf]
            corks = {key: corks[key] for key in corks if key not in descendants_list}

    return nodes


# used when the number of clusters is specified
# maybe some minmax implementation to optimize for large number of clusters
def get_all_covered_combinations(node, n):
    """
    Get all possible combinations of covered nodes in a tree up to a given depth.

    Args:
        node (TreeNode): The root node of the tree.
        n (int): The depth up to which the combinations are to be generated.

    Returns:
        list: A list of lists containing all possible combinations of covered nodes.
    """
    result = [[node]]  # Initialize the result list with the root node

    for _ in range(1, n):
        new_result = []  # Create a new list to store the updated combinations
        for combination in result:
            for idx, node in enumerate(combination):
                if node.children:  # Check if the current node has children
                    # Generate a new combination by removing the current node and adding its children
                    res_list = (
                        combination[:idx] + combination[idx + 1 :] + list(node.children)
                    )
                    res_list = sorted(
                        res_list, key=lambda x: x.id
                    )  # Sort the nodes by their id
                    if (
                        res_list not in new_result
                    ):  # Check if the new combination is not already in the result list
                        new_result.append(
                            res_list
                        )  # Add the new combination to the result list
        result = new_result  # Update the result list with the new combinations

    return result


def get_max_solidity(combinations):
    """
    Get the combination of objects with the maximum sum of solidity.

    Args:
        combinations (list): A list of combinations, where each combination is a list of objects.

    Returns:
        list: The combination of objects with the maximum sum of solidity.
    """
    current_max = float(
        "-inf"
    )  # Initialize the maximum sum of solidity to a very small value
    best_combination = None  # Initialize the best combination to None

    for combination in combinations:
        _sum = sum(
            [x.solidity for x in combination]
        )  # Calculate the sum of solidity for the current combination
        if (
            _sum > current_max
        ):  # If the sum of solidity is greater than the current maximum
            current_max = _sum  # Update the current maximum
            best_combination = combination  # Update the best combination

    return best_combination  # Return the combination of objects with the maximum sum of solidity


def preorder_cork_traversal(
    node, visited: set = set(), min_leaf: int = 1, depth: int = 2
):
    """
    Recursively traverses a binary tree in a preorder manner and collects all nodes to visited
    that meet one of the following conditions:
    a. one of children's children_count attribute is less than min_leaves
    b. not (child_1.self_cover or child_1.chain_cover) or not (child_2.self_cover or child_2.chain_cover)
        -- at least one of the children is both not self_cover and chain_cover

    Args:
        node (_type_): a Node object, the root node of the subtree to traverse.
        visited (set): a set of Node objects, keeping track of visited nodes that do not meet certain conditions.
        min_leaf (int): an integer, the minimum number of children a node should have.
        depth (int): Maximum depth to check for solidity comparison.

    Returns:
        None, the function updates the visited set.
    """

    def is_root_solidity_higher(node, child=None, depth=1):
        """
        Check if root node's solidity is higher than its children's solidity up to a certain depth.

        Args:
            node (Node): Root node of the binary tree.
            depth (int): Maximum depth to check for solidity comparison.

        Returns:
            bool: True if root node's solidity is higher than its children's solidity up to a certain depth, False otherwise.
        """
        if not node.children or depth == 0:
            return True

        if child and not child.children:
            return True

        if not child:
            for child in node.children:
                if node.solidity <= child.solidity:
                    return False
                elif not is_root_solidity_higher(node, child, depth=depth - 1):
                    return False
        else:
            for grandchild in child.children:
                if node.solidity <= grandchild.solidity:
                    return False
                elif not is_root_solidity_higher(node, grandchild, depth=depth - 1):
                    return False

        return True

    if is_root_solidity_higher(node=node, child=None, depth=depth):
        visited.add(node)
        return

    if node.children:
        child_1, child_2 = node.children[0], node.children[1]
        if (
            child_1.children_count < min_leaf
            or child_2.children_count < min_leaf
            or not child_1.children
            or not child_2.children
        ):
            visited.add(node)
            return

        if not (child_1.self_cover or child_1.chain_cover) or not (
            child_2.self_cover or child_2.chain_cover
        ):
            visited.add(node)
            return

        preorder_cork_traversal(child_1, visited, min_leaf)
        preorder_cork_traversal(child_2, visited, min_leaf)


def calculate_chain_solidity(node):
    """
    Calculating the solidity for the tree recursively.
    Example: chain_solidity_calculation(root)

    Args:
        node (_type_): AnyTree Node

    Returns:
        _type_: AnyTree Node
    """
    if node.children:
        # Process the current node
        _sum = sum(child.chain_solidity for child in node.children)
        node.chain_solidity = (
            _sum if node.chain_solidity < _sum else node.chain_solidity
        )

        # Recursively process the children nodes
        for child in node.children:
            calculate_chain_solidity(child)

    return node


def postorder_chain_cover_traversal(node):
    """Modifies parent chain_cover attribute to True if both children's self_cover is True

    Args:
        node (_type_): _description_

    Returns:
        _type_: _description_
    """
    if node.children:
        res_1 = postorder_chain_cover_traversal(node.children[0])
        res_2 = postorder_chain_cover_traversal(node.children[1])

        if res_1 and res_2:  # or node.self_cover:
            node.chain_cover = True
            return True

    return node.self_cover


# ratio of solidities that are larger in depth = min depth
def postorder_children_count_traversal(node):
    if node.children:
        res_1 = postorder_children_count_traversal(node.children[0])
        res_2 = postorder_children_count_traversal(node.children[1])
        node.children_count = res_1 + res_2
    else:
        node.children_count = 1
    return node.children_count


# use Node.ancestors parameter?
def get_ancestors(leaves, check_list=[]):
    """
    For all leaves in leaves collect ancestors in an ancestors dictionary.
    If a particular leaf is present in the check_list, break the while loop.

    TODO: remove node = leaf

    Args:
        leaves (_type_): _description_
        check_list (list, optional): _description_. Defaults to [].

    Returns:
        _type_: _description_
    """
    ancestors = {}
    for leaf in leaves:
        node = leaf
        ancestors[leaf.id] = []
        while node.parent:
            ancestors[leaf.id] += [node.parent.id]
            if node.parent not in check_list:
                node = node.parent
            else:
                break
    return ancestors


# to consoder using for final_nodes
# Node.root
# Node.siblings
