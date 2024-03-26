import numpy as np


def kmeans(X, k, max_iters=100):
    indices = np.random.choice(X.shape[0], k, replace=False)

    centroids = X[indices]

    closest_cluster = []

    for _ in range(max_iters):
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        closest_cluster = np.argmin(distances, axis=0)

        new_centroids = np.array(
            [
                X[closest_cluster == k].mean(axis=0)
                for k in range(centroids.shape[0])
             ]
        )

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return closest_cluster, centroids
