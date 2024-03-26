import pathlib

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def plot_datasets(datasets, titles):
    fig, axs = plt.subplots(1, len(datasets), figsize=(15, 3))
    for i, (X, y) in enumerate(datasets):
        axs[i].scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='viridis')
        axs[i].set_title(titles[i])
    plt.show()


def save_datasets(datasets, filenames):
    for (X, y), filename in zip(datasets, filenames):
        np.savetxt(pathlib.Path(__file__).parent / 'clustering-classes' / f"{filename}_X.csv", X, delimiter=",")
        np.savetxt(pathlib.Path(__file__).parent / 'clustering-classes' / f"{filename}_y.csv", y, delimiter=",")


# Generate circular clusters
X_circular, y_circular = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

# Generate elongated clusters
X_elongated, y_elongated = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

transformation_matrix = np.array([[0.6, -0.6], [-0.4, 0.8]])
X_elongated = np.dot(X_elongated, transformation_matrix)

# Generate "G"-shaped clusters
X_g_shape, y_g_shape = make_moons(n_samples=300, noise=0.05, random_state=42)
transformation_matrix_g = np.array([[1.2, -0.8], [0.5, 1.5]])
X_g_shape = np.dot(X_g_shape, transformation_matrix_g) + [20, 10]

# Elongated clusters with different sizes (Reusing the elongated concept with variations)
X_elongated_diff_sizes, y_elongated_diff_sizes = make_blobs(n_samples=1000,
                                                            centers=[[10, 0], [20, 5], [30, -5], [40, 5], [50, 0]],
                                                            cluster_std=[1.0, 2.0, 0.5, 1.5, 2.5], random_state=42)
X_elongated_diff_sizes = np.dot(X_elongated_diff_sizes, transformation_matrix)

# G-shaped clusters closer together (Adjusting parameters for closer proximity)
X_g_shape_close, y_g_shape_close = make_moons(n_samples=300, noise=0.1, random_state=42)
X_g_shape_close = np.dot(X_g_shape_close, transformation_matrix_g) + [20, 20]

# Circular clusters with varying proximity (using make_circles with noise for proximity)
X_circular_var_proximity, y_circular_var_proximity = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

# Elongated clusters very close together (Using transformation for elongation and proximity)
X_elongated_very_close, y_elongated_very_close = make_blobs(n_samples=300, centers=[[1, 1], [2, 2]], cluster_std=0.1,
                                                            random_state=42)
X_elongated_very_close = np.dot(X_elongated_very_close, transformation_matrix)

# G-shaped clusters with varying sizes (Using make_moons with noise and scaling)

X1, y1 = make_moons(n_samples=150, noise=0.05, random_state=42)
X2, y2 = make_moons(n_samples=450, noise=0.05, random_state=42)
X2 += np.array([1.5, 1.5])
X_g_shape_var_sizes = np.concatenate([X1, X2])
y_g_shape_var_sizes = np.concatenate([y1, y2 + 1])

# Circular clusters with mixed sizes and proximity (Combination of different sizes and proximities)
X_circular_mixed, y_circular_mixed = make_blobs(n_samples=[50, 100, 150], centers=[[0, 0], [4, 4], [8, 8]],
                                                cluster_std=[0.5, 1.5, 0.5], random_state=42)

datasets_plt = [
    (X_circular, y_circular),
    (X_elongated, y_elongated),
    (X_g_shape, y_g_shape),
    (X_elongated_diff_sizes, y_elongated_diff_sizes),
    (X_g_shape_close, y_g_shape_close),
    (X_circular_var_proximity, y_circular_var_proximity),
    (X_elongated_very_close, y_elongated_very_close),
    (X_g_shape_var_sizes, y_g_shape_var_sizes),
    (X_circular_mixed, y_circular_mixed),
]
titles = [
    'Circular Clusters',
    'Elongated Clusters',
    'G-shaped Clusters',
    'Elongated Clusters with Different Sizes',
    'G-shaped Clusters Closer Together',
    'Circular Clusters with Varying Proximity',
    'Elongated Clusters Very Close Together',
    'G-shaped Clusters with Varying Sizes',
    'Circular Clusters with Mixed Sizes and Proximity',
]

plot_datasets(
    datasets_plt,
    titles
)

save_datasets(
    datasets_plt,
    [
        'circular',
        'elongated',
        'g-shaped',
        'elongated-diff_sizes',
        'g-shaped-closer',
        'circular-varying_prox',
        'elongated-very_close',
        'g-shaped-varying_sizes',
        'circular-mixed_sizes-proximity'
    ]
)
