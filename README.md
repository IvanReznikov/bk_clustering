![Logo](https://github.com/IvanReznikov/bk_clustering/blob/main/docs/images/logo_white.png?raw=true)
# Burj Khalifa Clustering
The `bk_clustering` package is a Python realization of the Burj Khalifa clustering method.

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
[Iris Dataset](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/iris_dataset.ipynb)
[Comparison with other clustering methods](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/aggregation_dataset.ipynb)
[Mall Customer Segmentation](https://github.com/IvanReznikov/bk_clustering/blob/main/examples/Customer%20segmentation%20example.ipynb)

## Limitations
The method is distance-based so time performance might not be so great.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue on the GitHub repository.

## License
This package is licensed under the MIT License.

## Inspiration
The Burj Khalifa's ladder-view design consists of a succession of terraces or setbacks that, when viewed from the outside, resemble the branches of a tree or a dendrogram and gradually get smaller as the structure rises. The inspiration for the method was taken from the view of the building: instead of having the width changed gradually, the terraces are located at specific, uneven from different sides and levels, skipping some of them. Such construction is closely associated with the proposed method, where a dendrogram is modified to a tree structure on solid levels.