![Logo](https://github.com/IvanReznikov/bk_clustering/blob/main/docs/images/logo_black.png?raw=true)
# Burj Khalifa Clustering
The `bk_clustering` package is a Python realization of the Burj Khalifa clustering method.
The Burj Khalifa method can be considered as Agglomerative clustering on steroids: great quality with no parameters required!

The idea is to automatically detect solid clusters, based on the dendrogram. Read more in the publication section.

## Installation
To install the package, run:
```python
pip install bk_clustering
```

## Usage
Here's an example of using Burj Khalifa clustering algorithm:
```python
from bk_clustering import BurjKhalifaClustering

# Initialize BurjKhalifaClustering object
bk_model = BurjKhalifaClustering()

# Fit data to the algorithm
bk_model.fit(X)

# Get labels
labels = bk_model.labels_
```

## Examples
- [Iris Dataset](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/iris_dataset.ipynb)
- [Comparison with other clustering methods](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/aggregation_dataset.ipynb)
- [Mall Customer Segmentation](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/Mall%20Customer%20Segmentation.ipynb)
- [Bank Customer Segmentation](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/Bank%20Customer%20Segmentation.ipynb)

## Time limitations
The time complexity of distance-based algorithms typically depends on the number of data points, the number of features, and the number of clusters.
Building a distance matrix for hierarchical clustering is essential in the clustering process. The size of the distance matrix is N x N, where N is the number of data points in the dataset. The time complexity of building the distance matrix for hierarchical clustering is O(N^2), which means that the time required to compute the pairwise distances between all data points increases quadratically with the size of the dataset.
As the data size increases, the time complexity of distance computation snowballs, making these algorithms computationally expensive.
Below are time performance matrixes attached for 10 and 100 clusters.
![Time performance](https://github.com/IvanReznikov/bk_clustering/blob/main/docs/images/time_performance.png?raw=true)
As one can see, even with 10.000^2 data points, the algorithm works with acceptable timings (just a bit longer than 10 minutes [Performance notebook](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/Performance%20testing.ipynb)).

Further work will be dedicated to optimizing python code and parallelizing some steps.

## Publication
The publication is under draft at the moment.
[Plots and tables](https://github.com/IvanReznikov/bk_clustering/blob/main/publication/plots%20and%20tables.ipynb) for the paper.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository.

## License
This package is licensed under the MIT License.

## Inspiration
The Burj Khalifa's ladder-view design consists of a succession of terraces or setbacks that, when viewed from the outside, resemble the branches of a tree or a dendrogram and gradually get smaller as the structure rises. The inspiration for the method was taken from the view of the building: instead of having the width changed gradually, the terraces are located at specific, uneven from different sides and levels, skipping some of them. Such construction is closely associated with the proposed method, where a dendrogram is modified to a tree structure on solid levels.