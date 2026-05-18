"""
Utilities for estimating the optimal number of clusters and performing embedding-based clustering.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def estimate_n_clusters(
    embeddings, max_k=10, method: str = "elbow", plot: bool = False
) -> int:
    """Estimate the optimal number of clusters using Elbow or Silhouette methods.

    Computes KMeans clustering for `k=2` to `max_k` clusters and estimates the
    optimal number of clusters using the specified method. Optionally, plots the
    evaluation metric for visual inspection.

    Args:
        embeddings (np.ndarray): Data to cluster (shape: [n_samples, embedding_dim]).
        max_k (int): Maximum number of clusters to consider. Default is 10.
        method (str): Method to use for estimating clusters. Either "elbow" or "silhouette".
        plot (bool): If True, displays a plot of the metric versus number of clusters. Default False.

    Returns:
        int: Estimated optimal number of clusters.

    Notes:
        - "elbow": selects k with largest drop in inertia (sum of squared distances).
        - "silhouette": selects k with highest average silhouette score.
        - The optional plot helps visualize the chosen metric.
    """
    if method not in ("elbow", "silhouette"):
        raise ValueError("method must be either 'elbow' or 'silhouette'")

    if method == "elbow":
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(embeddings)
            inertias.append(kmeans.inertia_)

        if plot:
            plt.plot(range(1, max_k + 1), inertias, marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Inertia (sum of squared distances)")
            plt.title("Elbow Method for n_clusters")
            plt.show()

        deltas = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]
        n_clusters = deltas.index(max(deltas)) + 2

    else:  # silhouette
        best_score = -1
        n_clusters = 2
        scores = []
        for k in range(2, min(max_k, len(embeddings) - 1) + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            scores.append(score)
            if score > best_score:
                best_score = score
                n_clusters = k

        if plot:
            plt.plot(range(2, max_k + 1), scores, marker="o")
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Method for n_clusters")
            plt.show()

    logger.debug(f"Estimated n_clusters={n_clusters} using {method}")
    return n_clusters


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = None,
    method: str = "elbow",
    max_k: int = 10,
    plot: bool = False,
) -> np.ndarray:
    """Cluster embeddings using KMeans.

    If `n_clusters` is not provided, it is automatically estimated using the
    specified method ('elbow' or 'silhouette').

    Args:
        embeddings (np.ndarray): Array of embeddings to cluster (shape: [n_samples, embedding_dim]).
        n_clusters (int, optional): Number of clusters to form. If None, estimated automatically.
        method (str): Method to use for automatic cluster estimation. Either 'elbow' or 'silhouette'.
        max_k (int): Maximum number of clusters to consider when estimating.
        plot (bool): Whether to plot the metric used for automatic cluster estimation.

    Returns:
        tuple[np.ndarray, KMeans]:
            - Array of cluster labels for each embedding.
            - Fitted KMeans object.

    Raises:
        ValueError: If `embeddings` is None or empty.
    """
    if embeddings is None or len(embeddings) == 0:
        raise ValueError("Embeddings must be provided and non-empty.")

    if n_clusters is None:
        n_clusters = estimate_n_clusters(
            embeddings, max_k=max_k, method=method, plot=plot
        )

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels, kmeans
