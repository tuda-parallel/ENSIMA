"""
MoE PointNet System: A simple Mixture of Experts framework for point cloud classification and embedding using a PointNet-style encoder.

This module provides:
- A Mixture-of-Experts manager (MoEPointNetSystem) that supports both training modes.
- Dummy expert models that simulate downstream expert predictions.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import sys
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from ensima_optimize.classes.expert import Expert
from ensima_optimize.classes.geometry_nn import PointNetAutoEncoder, PointNetClassifier
from ensima_optimize.classes.logger import Logger
from ensima_optimize.helpers.clustering import cluster_embeddings
from ensima_optimize.helpers.read_geometry import (
    preprocess_point_cloud,
    read_coordinates_from_file,
    uniform_vector,
)


class MoEPointNetSystem:
    """Mixture of Experts (MoE) system for point cloud data using PointNet embeddings.

    Supports two modes:
        1. **Supervised classification**
           - Uses `PointNetClassifier` to predict labels for point clouds.
           - Learns **discriminative embeddings** that separate classes.
           - Builds an embedding library and automatically creates an expert per class.
           - Uses cosine similarity for gating new point clouds to the correct expert.
           - Fully supervised; requires labels in `points`.

        2. **Unsupervised clustering**
           - Uses `PointNetAutoEncoder` to learn embeddings from point clouds.
           - Embeddings capture geometric structure, not class separation.
           - Clusters embeddings with KMeans to define groups; each cluster gets an expert.
           - Labels are optional; the number of clusters (`n_clusters`) can be specified or
             automatically estimated using a heuristic (Elbow method or silhouette score).
           - This mode is fully unsupervised if `n_clusters` is inferred from embeddings;
             otherwise, specifying `n_clusters` imposes a partial supervision.
           - Uses cosine similarity or euclidean distance for gating new point clouds to the nearest cluster/expert.


    Args:
        mode (str): "supervised" or "unsupervised".
        train_points (dict[str, tuple[np.ndarray, np.ndarray]]): Dictionary mapping each part label to its training data. Each value is a tuple containing:
            - x (np.ndarray): Input features for the part, shape (n_points_per_part, n_features).
            - y (np.ndarray): Target values for the part, shape (n_points_per_part, n_targets).
        model_settings (dict[str, dict]):
            Dictionary mapping each part label to its model configuration
            (see `model.py` for details). Each inner dict may include keys such as:
                - "log_level" (str): Logging level for the expert model.
                - "normalize_y" (bool): Whether to normalize target values.
                - "model_filename" (str): File path for saving/loading the model.
                - "optimizer_restart" (int): Number of optimizer restarts.
                - "epochs" (int): Number of training epochs.
                - "save_and_load" (bool): If True, save model after training and
                  load it later instead of retraining.
        geo_points (list of tuple[np.ndarray, np.ndarray, np.ndarray, str]): Point cloud data.
            Each tuple is (x, y, z, label), where x, y, z are arrays of coordinates and
            label is a string (used in supervised mode; optional in unsupervised mode).
        n_points (int): Number of points to sample from each point cloud. If not provided
            the min length of the x axis in the parts is used.
        embedding_dim (int): Dimensionality of the PointNet embeddings. If not provided,
            it's computed as `n_points/10`.
        n_clusters (int, optional): Number of clusters for unsupervised mode. If None,
            inferred from labels if available, otherwise defaults to 4.
        args (Namespace, optional): Command line arguments passed. It Contains:
            - jobname (str, optional): Name of the optimization job. Defaults to "optimization".
            - output (str, optional): Directory or filename for storing optimization results. Defaults to "{jobname}_results".
        sample_mode (str, optional): Sample mode for the point cloud. Defaults to "upsample" ("downsample" or "upsample"). Upsamle is more accurate, but demands more memory. Downsample is faster but less accurate.
    """

    def __init__(
        self,
        mode: str,
        train_points: list[tuple[np.ndarray, np.ndarray, str]],
        model_settings: dict[str, dict],
        geo_points: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        n_points: int = None,
        embedding_dim: int = None,
        n_clusters: int = None,
        log_level="INFO",
        args: Namespace = None,
        sample_mode="upsample",
        save_and_load: bool = False,
    ):
        assert mode in [
            "supervised",
            "unsupervised",
        ], "mode must be 'supervised' or 'unsupervised'"
        self.log_level = log_level
        self.logger = Logger(__name__, level=self.log_level).get()
        self.mode = mode
        self.save_and_load = save_and_load  # not implemented yet
        self.args = args
        if self.args is not None:
            self.name = self.args.jobname
        else:
            self.name = "New_part"

        # set the device to use
        device = None
        for key, inner_dict in model_settings.items():
            device = inner_dict.get("device")
            if device is not None:
                break
        self.device = device or "cpu"

        self.sample_mode = sample_mode
        if self.sample_mode == "upsample":
            # Choose the largest point cloud → then smaller clouds will be upsampled
            if n_points is None:
                n_points = 0
                for geo in geo_points:
                    n_points = max(n_points, len(geo[0]))
                    self.logger.debug(
                        f"n_points updated to {n_points} for {geo[3]} (contains {len(geo[0])} points)"
                    )
                self.logger.info(f"Using n_points={n_points} for all point clouds")

        elif self.sample_mode == "downsample":
            # Choose the smallest point cloud → then larger clouds will be downsampled
            if n_points is None:
                n_points = np.inf
                for geo in geo_points:
                    n_points = min(n_points, len(geo[0]))
                    self.logger.debug(
                        f"n_points updated to {n_points} for {geo[3]} (contains {len(geo[0])} points)"
                    )
                self.logger.info(f"Using n_points={n_points} for all point clouds")

        else:
            raise ValueError("Unsupported sample mode.")

        self.n_points = n_points

        if embedding_dim is None:
            embedding_dim = int(n_points / 10)  # 128
            if embedding_dim < 1000 and n_points > 1000:
                embedding_dim = 1000
                self.logger.warning(
                    f"embedding_dim too small, adjusted embedding_dim={embedding_dim} "
                )
            limit = 5_000
            if embedding_dim > limit:
                self.logger.info(
                    f"embedding_dim is very large ({embedding_dim}), setting embedding_dim={limit} "
                )
                embedding_dim = limit
            self.logger.info(
                f"Using embedding_dim={embedding_dim} for all point clouds"
            )
        self.embedding_dim = embedding_dim

        self.train_points = train_points
        self.geo_points = geo_points
        self.n_clusters = n_clusters
        self.model_settings = model_settings
        self.reset_points = False
        # geo of new part:
        self.geo_for_prediction = None  # set this during prediction

        self.experts = {}  # label -> expert model (e.g., DummyGP)
        self.embeddings = None  # numpy array of embeddings
        self.labels = []  # list of labels per embedding
        self.kmeans = None  # Add this line to hold the trained KMeans model

        # only used in function prediction as wrappers
        self.geometry_path = None  # path to t52 file
        self.soft = None

        # for training, we need to know which experts should be trained
        self.recent_used_experts = None

        # used for new expert (indicates when the new expert is created
        self.counter = 0

        # Check if training data contains the new part. If this is the case, fall back
        for i, geo in enumerate(geo_points):
            if len(geo) > 0 and self.name in geo[3]:
                self.logger.info(f"Found {self.name} in training data")
                # check if enough points exist to fall back to default mode and avoid moe
                x = train_points[geo[3]][0]
                if (
                    len(x) > self.args.new_expert_start
                ):  # minimum number of points to consider the part as new
                    self.logger.info("Enough points exist to fall back to default mode")
                    self.mode = "fallback"
                    self.soft = False  # hard
                break

        self.create_experts()

    def create_experts(
        self,
    ) -> None:
        """Create expert models for each part or cluster.

        This function initializes expert models based on the given training points,
        model configuration, and geometric data. It is typically called from `__init__`.
        The behavior of this function depends on ``self.mode``:

        * **"supervised"**: One expert is created per part label using the
          provided ``train_points`` and ``model_settings``.
        * **"unsupervised"**: Experts are created by clustering the embeddings
          of the given ``geo_points`` into ``n_clusters`` groups.
        * **"fallback"**: Treated as a single-expert setup, ignoring labels
          or clusters.

        Args:
            None

        Returns:
            None
        """
        if self.mode == "fallback":
            self.logger.info("=== Fallback Mode ===")
            label = self.name
            x, y = self.train_points[label]
            self.logger.debug(f"Size x is {np.size(x)}, y is {np.size(y)}")
            self.experts[label] = Expert(
                label, x, y, **self.model_settings.get(label, {})
            )
            self.model = PointNetClassifier(
                num_classes=1, embedding_dim=self.embedding_dim
            ).to(
                self.device
            )  # dummy model no used

        elif self.mode == "supervised":
            self.logger.info("=== Supervised Mode ===")
            # map label -> class index
            label_to_idx = {
                label: i
                for i, label in enumerate(sorted({l for _, _, _, l in self.geo_points}))
            }
            # supervised mode, number of classes is computed from unique labels in points
            num_classes = len(label_to_idx)
            self.logger.info(f"Number of experts: {num_classes}")
            self.model = PointNetClassifier(
                num_classes=num_classes, embedding_dim=self.embedding_dim
            ).to(self.device)
            # Train classifier
            self.logger.info(
                f"Training classifier ({self.n_points} -> {self.embedding_dim} pts) on {self.device}"
            )
            self.train_classifier(self.geo_points, label_to_idx, n_epochs=5)
            # Build embeddings
            self.logger.info(
                f"Building embeddings for {num_classes} with {self.embedding_dim} pts"
            )
            self.build_embedding_library(self.geo_points)
            # Create dummy experts per label
            self.logger.info("Creating experts")
            for label in label_to_idx:
                self.logger.info(f"Creating expert '{label}'")
                x, y = self.train_points[label]
                self.logger.debug(f"Size x is {x.shape}, y is {y.shape}")
                self.experts[label] = Expert(
                    label, x, y, **self.model_settings.get(label, {})
                )

        elif self.mode == "unsupervised":
            self.logger.info("=== Unsupervised Mode ===")
            self.model = PointNetAutoEncoder(latent_dim=self.embedding_dim).to(
                self.device
            )

            # Train autoencoder (labels unused)
            # The autoencoder is trained to compress each point cloud into a low-dimensional latent vector
            self.logger.debug("Training autoencoder")
            self.train_autoencoder(self.geo_points, n_epochs=5)

            # Build embeddings
            self.logger.debug("Building Embedding Library")
            self.build_embedding_library(self.geo_points)

            # Determine max number of clusters
            # Extract labels if available (usually unsupervised method doesn't need
            # this, but if it is provided, benefit from it)
            try:
                labels_in_data = [l for *_, l in self.geo_points]
                max_k = (
                    len(set(labels_in_data)) if labels_in_data else 10
                )  # fallback max=10
            except Exception:
                max_k = None
            self.logger.debug(f"Max number of clusters: {max_k}")

            # Cluster embeddings (n_clusters automatically estimated if None)
            clusters, kmeans_model = cluster_embeddings(
                embeddings=self.embeddings,
                n_clusters=self.n_clusters,  # can be None
                method="elbow",  # or silhouette
                max_k=max_k,
                plot=False,
            )
            self.kmeans = kmeans_model  # store fitted KMeans model
            # Assign cluster labels
            self.labels = [f"cluster_{cid}" for cid in clusters]

            # Create experts per cluster
            self.logger.info(f"Creating {len(set(self.labels))} Experts")
            for clabel in sorted(set(self.labels)):
                # find which lables match
                indices = [i for i, lbl in enumerate(self.labels) if lbl == clabel]
                selected_labels = [labels_in_data[i] for i in indices]
                self.logger.info(
                    f"Expert {clabel} embeds the hidden labels {selected_labels}"
                )
                # extract the data according to the label
                x = np.concatenate(
                    [self.train_points[label][0] for label in selected_labels], axis=0
                )
                y = np.concatenate(
                    [self.train_points[label][1] for label in selected_labels], axis=0
                )
                settings = {}
                for label in selected_labels:
                    settings.update(self.model_settings.get(label, {}))
                # Finally create expert
                self.experts[clabel] = Expert(
                    clabel, x, y, **settings, hidden_labels=selected_labels
                )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def train_classifier(
        self,
        points: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        label_to_idx: dict[str, int],
        n_epochs: int = 20,
        lr: float = 1e-3,
    ) -> None:
        """Train the supervised classifier on labeled point clouds. Center each point
            cloud, so the classifier  is robust to spatial offsets (objects in different
            room locations).

        Args:
            points: List of tuples (x, y, z, label) where x,y,z are arrays of point coords.
            label_to_idx: Mapping from string labels to integer indices.
            n_epochs: Number of training epochs.
            lr: Learning rate.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        self.logger.debug(f"Training Classifier with {n_epochs} epochs")
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x, y, z, label in points:
                pc = uniform_vector(x, y, z, self.n_points, self.sample_mode)
                pc = preprocess_point_cloud(pc)

                pc_tensor = torch.tensor(
                    pc, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # (1, N, 3)
                target = torch.tensor(
                    [label_to_idx[label]], dtype=torch.long, device=self.device
                )

                optimizer.zero_grad()
                logits = self.model(pc_tensor)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            self.logger.debug(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(points):.4f}"
            )

    def train_autoencoder(
        self,
        points: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        n_epochs: int = 20,
        lr: float = 1e-3,
    ) -> None:
        """Train the unsupervised autoencoder to reconstruct point clouds.

        Args:
            points: List of tuples (x, y, z, label) - labels unused here.
            n_epochs: Number of training epochs.
            lr: Learning rate.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()

        self.logger.debug(f"Training autoencoder with {n_epochs}")
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x, y, z, _ in points:
                pc = uniform_vector(x, y, z, self.n_points, self.sample_mode)
                pc = preprocess_point_cloud(pc)
                pc_tensor = torch.tensor(
                    pc, dtype=torch.float32, device=self.device
                ).unsqueeze(
                    0
                )  # (1, N, 3)

                optimizer.zero_grad()
                recon = self.model(pc_tensor)
                loss = loss_fn(recon, pc_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            self.logger.debug(
                f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(points):.4f}"
            )

    def build_embedding_library(
        self, points: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ) -> None:
        """Compute and store embeddings for a list of point clouds.

        Args:
            points: List of tuples (x, y, z, label).
        """
        self.model.eval()
        embeddings = []
        labels = []

        with torch.no_grad():
            for x, y, z, label in points:
                pc = uniform_vector(x, y, z, self.n_points, self.sample_mode)
                pc = preprocess_point_cloud(pc)
                pc_tensor = torch.tensor(
                    pc, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                if self.mode == "supervised":
                    emb = self.model.get_embedding(pc_tensor).squeeze(0).cpu().numpy()
                else:
                    emb = self.model.encoder(pc_tensor).squeeze(0).cpu().numpy()
                embeddings.append(emb)
                labels.append(label)

        self.embeddings = np.stack(embeddings)
        self.labels = labels

    def assign_cluster_label(self, new_emb: np.ndarray) -> str:
        assert hasattr(self, "kmeans"), "KMeans model not fitted yet."
        cluster_idx = self.kmeans.predict(new_emb.reshape(1, -1))[0]
        return f"cluster_{cluster_idx}"

    def assign_to_expert(
        self,
        new_emb: np.ndarray,
        soft: bool = False,
        metric: str = "euclidean",  # "cosine"
    ) -> dict[str, float] | str:
        """Assign an embedding to expert(s) based on a chosen similarity/distance metric.

        Supports soft or hard gating. Can use either cosine similarity or Euclidean distance.

        Args:
            new_emb (np.ndarray): New embedding vector.
            soft (bool): If True, returns soft weights for all experts; if False, returns hard assignment.
            metric (str): Metric to use: "cosine" for cosine similarity, "euclidean" for Euclidean distance.

        Returns:
            dict[str, float] | str:
                - If soft=True, returns a dict of {label: weight} pairs.
                - If soft=False, returns the label of the assigned expert.

        Raises:
            ValueError: If `metric` is not "cosine" or "euclidean".
        """
        if metric == "cosine":
            scores = cosine_similarity([new_emb], self.embeddings)[0]
            self.logger.debug(f"Cosine similarities: {scores}")
            # For soft gating, higher score = higher weight
            best_idx_func = np.argmax
        elif metric == "euclidean":
            dists = euclidean_distances([new_emb], self.embeddings)[0]
            self.logger.debug(f"Euclidean distances: {dists}")
            # invert distances for soft gating so higher = better
            scores = 1 / (dists + 1e-8)

            def best_idx_func(_):
                return np.argmin(dists)

        else:
            raise ValueError("Invalid metric. Choose 'cosine' or 'euclidean'.")

        if soft:
            total = np.sum(scores)
            if total > 0:
                weights = scores / total
            else:
                weights = np.ones_like(scores) / len(scores)
            weighted = {self.labels[i]: weight for i, weight in enumerate(weights)}
            self.logger.debug(f"Soft gating resulted in {weighted} experts")
            return weighted
        else:
            best_idx = best_idx_func(scores)
            self.logger.debug(f"Hard gating with {self.labels[best_idx]}")
            return self.labels[best_idx]

    def predict_with_moe(
        self,
        x: np.ndarray,
        unknown_geometry: tuple[np.ndarray, np.ndarray, np.ndarray],
        soft: bool = False,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform Mixture-of-Experts prediction on new input(s).

        In supervised mode:
            - Soft gating: combine expert predictions using cosine similarity weights.
            - Hard gating: assign to a single expert based on nearest embedding.
        In unsupervised mode:
            - Hard assignment only: predict cluster label via stored k-means model
              and use the corresponding expert. Soft gating is not supported.

        Args:
            x (np.ndarray): Input feature(s) for prediction, shape (n_samples, n_features).
            unknown_geometry (tuple[np.ndarray, np.ndarray, np.ndarray]):
                Tuple of arrays representing the geometry (X, Y, Z coordinates).
            soft (bool, optional): If True, use soft gating (only valid in supervised mode).
                Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Mean and standard deviation of the prediction(s).
        """
        mean, std = None, None
        self.recent_used_experts = {}
        geo_x, geo_y, geo_z = unknown_geometry
        pc = uniform_vector(geo_x, geo_y, geo_z, self.n_points, self.sample_mode)
        pc = preprocess_point_cloud(pc)

        with torch.no_grad():
            pc_tensor = torch.tensor(
                pc, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        if self.mode == "supervised":
            with torch.no_grad():
                emb = self.model.get_embedding(pc_tensor).squeeze(0).cpu().numpy()

            if soft:
                self.logger.debug("Performing soft gating")
                assignment = self.assign_to_expert(emb, soft=True)
                self.logger.info(f"++ Soft Gating with {len(assignment)} experts ++")

                # filter experts by weight threshold and availability
                filtered_assignment = self.filter_experts(assignment, threshold=0.1)

                variance_accum = None  # accumulator for mixture variance
                for label, weight in filtered_assignment:
                    # getr the needed expert
                    expert = self.experts.get(label)
                    if expert is None:
                        self.logger.critical(f"No expert found for label '{label}'")
                        continue
                    # perform prediction with expert
                    expert_mean, expert_std = expert.predict(x, *args, **kwargs)
                    self.logger.info(
                        f"Expert '{label}' used with weight {weight*100:.2f}%"
                    )
                    # weight the results according to soft gating
                    if mean is None:
                        mean = expert_mean * weight
                        variance_accum = weight * (expert_std**2 + expert_mean**2)
                    else:
                        mean += expert_mean * weight
                        variance_accum += weight * (expert_std**2 + expert_mean**2)

                if variance_accum is not None:
                    mixture_var = variance_accum - mean**2
                    mixture_var = np.maximum(mixture_var, 0.0)  # ensure non-negative
                    std = np.sqrt(mixture_var)

            else:  # Hard gating
                self.logger.info("++ Hard gating with 1 expert ++")
                label = self.assign_to_expert(emb, soft=False)
                # save the label for training the expert later
                self.recent_used_experts[label] = 1
                # get the expert
                expert = self.experts.get(label)
                self.logger.info(f"Assigned to expert '{label}'")
                if expert is None:
                    self.logger.critical(f"No expert found for label {label}")
                else:
                    mean, std = expert.predict(x, *args, **kwargs)

        elif self.mode == "unsupervised":
            with torch.no_grad():
                emb = self.model.encoder(pc_tensor).squeeze(0).cpu().numpy()

            if self.kmeans is None:
                self.logger.critical(
                    "Error: KMeans model not available for unsupervised prediction."
                )
            else:
                if soft:
                    self.logger.debug(
                        "++ Performing soft gating based on cluster distances ++"
                    )
                    centroids = self.kmeans.cluster_centers_
                    distances = np.linalg.norm(centroids - emb.reshape(1, -1), axis=1)
                    tol = 0.1
                    variance_accum = None  # accumulator for mixture variance
                    valid_idx = np.where(distances < tol)[0]
                    if len(valid_idx) == 0:  # Fall back to hard gating
                        self.logger.critical(
                            f"No clusters within tolerance {tol}, falling back to hard gating."
                        )
                        expert_to_use = [self.kmeans.predict(emb.reshape(1, -1))[0]]
                        weights = [1.0]
                    else:
                        expert_to_use = valid_idx
                        weights = 1.0 / (distances[valid_idx] + 1e-8)
                        weights = weights / weights.sum()

                    for i in range(len(expert_to_use)):
                        label = f"cluster_{expert_to_use[i]}"
                        expert = self.experts.get(label)
                        if expert is None:
                            continue
                        # Save the expert used for training later
                        self.recent_used_experts[label] = weights[i]
                        expert_mean, expert_std = expert.predict(x, *args, **kwargs)
                        # hidden_labels is only logged in unsupervised mode for transparency
                        self.logger.info(
                            f"Expert {label}, which embeds {expert.hidden_labels}, used with weight {weights[i]*100:.2f}%"
                        )
                        # predict with the expert and weight the results
                        if mean is None:
                            mean = expert_mean * weights[i]
                            variance_accum = weights[i] * (
                                expert_std**2 + expert_mean**2
                            )
                        else:
                            mean += expert_mean * weights[i]
                            variance_accum += weights[i] * (
                                expert_std**2 + expert_mean**2
                            )

                    if variance_accum is not None:
                        mixture_var = variance_accum - mean**2
                        mixture_var = np.maximum(
                            mixture_var, 0.0
                        )  # ensure non-negative
                        std = np.sqrt(mixture_var)

                else:  # Hard gating
                    self.logger.info("++ Hard gating with 1 expert ++")
                    cluster_label_idx = self.kmeans.predict(emb.reshape(1, -1))[0]
                    label = f"cluster_{cluster_label_idx}"
                    expert = self.experts.get(label)
                    # hidden_labels is only logged in unsupervised mode for transparency
                    self.logger.info(
                        f"Assigned to expert '{label}' which embeds {expert.hidden_labels}"
                    )
                    if expert is None:
                        self.logger.critical(f"No expert found for cluster '{label}'")
                    else:
                        # save the expert used for training later
                        self.recent_used_experts[label] = 1
                        # predict with the expert
                        mean, std = expert.predict(x, *args, **kwargs)

        elif self.mode == "fallback":
            # no gating or embeddings needed
            label = self.name
            self.logger.info(f"++ Using the expert {label} ++")
            expert = self.experts.get(label)
            mean, std = expert.predict(x, *args, **kwargs)
            self.recent_used_experts[label] = 1
        else:
            raise ValueError(
                "Invalid mode. Choose 'supervised', 'unsupervised' or 'fallback'."
            )

        self.logger.debug("Prediction completed. Mean and STD computed.")
        return mean, std

    def filter_experts(self, assignment, threshold=0.1):
        """Filter experts by weight threshold and renormalize survivors.

        This method removes experts whose weight falls below a given threshold or
        whose model is not available. The remaining experts have their weights
        renormalized so that they sum to 1.

        Args:
            assignment (dict[str, float]):
                Mapping from expert label to its assigned weight. Assumes the
                weights sum to 1 before filtering.
            threshold (float, optional):
                Minimum weight required for an expert to be retained.
                Defaults to 0.1.

        Returns:
            list[tuple[str, float]]:
                A list of (expert_label, normalized_weight) pairs. The normalized
                weights always sum to 1. If no experts survive filtering, an
                empty list is returned.
        """
        survivors = []
        for label, weight in assignment.items():
            expert = self.experts.get(label)
            if expert is None:
                continue
            if weight > threshold:
                survivors.append((label, weight))
                self.recent_used_experts[label] = weight
            else:
                self.logger.info(
                    f"Expert {label} contribution is too low ({weight*100:.1f}% < {threshold*100:.1f}%). Removing it"
                )

        # fallback: take the max-weight expert if none survived filtering
        if not survivors:
            max_label = max(assignment, key=assignment.get)
            self.logger.warning(
                f"No experts survived filtering. Falling back to max-weight expert '{max_label}'."
            )
            return [(max_label, 1.0)]

        total = sum(w for _, w in survivors)
        return [(label, w / total) for (label, w) in survivors]

    def set_geometry_path(self, geometry_path):
        self.geometry_path = geometry_path

    def set_gating_mode(self, soft: bool) -> None:
        if "fallback" not in self.mode:
            self.soft = soft
        else:
            self.logger.warning(
                "Gating mode cannot be changed in 'fallback' mode. Skipping."
            )

    def predict(
        self,
        x: np.ndarray,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.geometry_path is None:
            raise ValueError("Geometry path not set.")
        if self.soft is None:
            raise ValueError("Gating mode not set.")
        if self.geo_for_prediction is None:
            coords = read_coordinates_from_file(
                self.geometry_path, log_level=self.log_level
            )
            if len(coords) > 0:
                geo_x = coords[:, 0]
                geo_y = coords[:, 1]
                geo_z = coords[:, 2]
                self.geo_for_prediction = (geo_x, geo_y, geo_z)
        return self.predict_with_moe(
            x, self.geo_for_prediction, *args, soft=self.soft, **kwargs
        )

    def train_expert(self, x, y, epochs):
        """Train recently used experts with the provided data.

        This method wraps the `train` method of the experts defined in `model.py`.
        Only experts listed in `self.recent_used_experts` will be trained.

        Args:
            x (np.ndarray): Input training data.
            y (np.ndarray): Output training data.
            epochs (int): Number of training epochs.
        """
        expert = self.experts.get(self.name)
        self.counter += 1
        # train a new expert in case it does not already exist once enough iterations
        # (new_expert_start) are completed.
        # Do this only once!
        if expert is None and self.counter >= self.args.new_expert_start:
            self.logger.info(f"Creating new expert '{self.name}'")
            self.logger.info(f"Extracting geometry for '{self.name}'")
            ## IMPORTANT: if the lines are enabled, all old experts are deleted and there
            # is only one (the new one)
            settings = self.model_settings[self.geo_points[-1][3]]

            self.geo_points = []
            self.train_points = {}
            self.model_settings[self.name] = settings
            coords = read_coordinates_from_file(
                self.geometry_path, log_level=self.log_level
            )
            self.geo_points.append(
                (coords[:, 0], coords[:, 1], coords[:, 2], self.name)
            )
            # Extract the relevant input output points from the large csv file
            self.logger.info(f"Creating training points for '{self.name}'")
            # Extract only the part of the point (shrinking is done outside)
            self.train_points[self.name] = (
                x[-self.args.new_expert_start :, :],
                y[-self.args.new_expert_start :, :],
            )
            self.logger.info(f"training points for '{self.name}'")
            self.reset_points = True
            # the expert will not bne used, unless the embeddings are recalculated
            self.logger.info("Retraining all experts and classificators")
            self.create_experts()
            # only use this expert from now one
            self.mode = "supervised"
            self.soft = "hard"
        # if too few points are available, train the other experts to improve predictions
        else:
            self.logger.info(f"Training {len(self.recent_used_experts)} experts")
            for i, label in enumerate(self.recent_used_experts):
                expert = self.experts.get(label)
                if expert is None:
                    self.logger.critical(f"No expert found for cluster '{label}'")
                    continue
                self.logger.info(
                    f"({i+1}/{len(self.recent_used_experts)}) Training expert '{label}'"
                )
                # we only need the last point for training plus what the points associated
                # to the parts
                x_train, y_train = self.train_points[label]
                x_train = np.vstack([x_train, x[-1]])
                y_train = np.vstack([y_train, y[-1]])
                self.logger.info(
                    f"Using {x_train.shape} input and  {y_train.shape} output arrays"
                )
                try:  # attempt to train the expert:
                    expert.train(x_train, y_train, epochs)
                    self.logger.debug(f"Finished training expert '{label}'")
                    # save for future use
                    self.train_points[label] = (x_train, y_train)
                except Exception as e:
                    self.logger.error(f"Training expert '{label}' failed: {e}")

                    # Check if we have enough experts left to continue
                    if len(self.experts) <= 2:
                        self.logger.critical(
                            f"Not enough experts remaining ({len(self.experts)}). Exiting."
                        )
                        sys.exit(1)
                    # Otherwise, remove the failed expert and continue
                    self.logger.warning(
                        f"Removing expert '{label}' (remaining: {len(self.experts)-1})"
                    )
                    self.experts.pop(label, None)
                    self.train_points.pop(label, None)

    def train(
        self, x: np.ndarray = None, y: np.ndarray = None, epochs: int = None
    ) -> None:
        """Train the GP experts using the selected backend.

        Delegates the training to `train_expert`, which trains only the experts
        listed in `self.recent_used_experts`.

        Args:
            x (np.ndarray, optional): Input training data. Defaults to None.
            y (np.ndarray, optional): Output training data. Defaults to None.
            epochs (int, optional): Number of training epochs. If not provided,
                the default value defined by the expert will be used.

        Returns:
            None
        """
        self.train_expert(x, y, epochs)

    def get_experts(self):
        return self.recent_used_experts


def example():
    # Generate some random parts geometry with labels
    geometry = []
    labels = ["gear", "shaft", "casing", "bracket"]
    n_points_per_part = 1000
    n_test_points = 2000

    scale = 5.0  # example scale factor, can be different per shape

    for label in labels:
        if label == "gear":
            # cube
            geo_x = np.random.rand(n_points_per_part) * scale
            geo_y = np.random.rand(n_points_per_part) * scale
            geo_z = np.random.rand(n_points_per_part) * scale
        elif label == "shaft":
            # cuboid (stretched in x)
            geo_x = np.random.rand(n_points_per_part) * 2 * scale
            geo_y = np.random.rand(n_points_per_part) * scale
            geo_z = np.random.rand(n_points_per_part) * scale
        elif label == "casing":
            # sphere
            phi = np.random.rand(n_points_per_part) * 2 * np.pi
            costheta = np.random.rand(n_points_per_part) * 2 - 1
            theta = np.arccos(costheta)
            r = (np.random.rand(n_points_per_part) ** (1 / 3)) * scale
            geo_x = r * np.sin(theta) * np.cos(phi)
            geo_y = r * np.sin(theta) * np.sin(phi)
            geo_z = r * np.cos(theta)
        elif label == "bracket":
            # cylinder
            theta = np.random.rand(n_points_per_part) * 2 * np.pi
            r = np.random.rand(n_points_per_part) * scale
            h = np.random.rand(n_points_per_part) * scale
            geo_x = r * np.cos(theta)
            geo_y = r * np.sin(theta)
            geo_z = h
        else:
            geo_x = np.random.rand(n_points_per_part) * scale
            geo_y = np.random.rand(n_points_per_part) * scale
            geo_z = np.random.rand(n_points_per_part) * scale

        geometry.append((geo_x, geo_y, geo_z, label))

    # Test prediction on a new random part
    # Generates pyramid along z-axis
    test_geo_z = np.random.rand(n_test_points) * scale
    # The max extent in x/y decreases linearly with height
    max_xy = scale * (1 - test_geo_z / scale)
    test_geo_x = (np.random.rand(n_test_points) * 2 - 1) * max_xy
    test_geo_y = (np.random.rand(n_test_points) * 2 - 1) * max_xy
    unknown_geometry = (test_geo_x, test_geo_y, test_geo_z)
    # Input vector fore new part
    new_x = np.random.rand(200, 1)

    functions = {
        "gear": lambda x: np.sin(2 * np.pi * x),
        "shaft": lambda x: np.cos(2 * np.pi * x),
        "casing": lambda x: x**2,
        "bracket": lambda x: np.sqrt(x),
    }
    # Create a dict to store x, y points per label
    train_points = {}
    model_settings = {}
    for label in labels:
        x = np.random.rand(n_points_per_part, 1)
        y = functions[label](x) + 0.1 * np.random.randn(n_points_per_part, 1)
        train_points[label] = (x, y)
        model_settings[label] = {
            "log_level": "INFO",
            "normalize_y": True,
            "optimizer_restart": 5,
            "epochs": 100,
            "save_and_load": True,
        }

    # ---- supervised Mode ----
    print("=== Supervised Mode ===")
    system_sup = MoEPointNetSystem(
        mode="supervised",
        model_settings=model_settings,
        train_points=train_points,
        geo_points=geometry,
        n_points=1024,
        log_level="DEBUG",
    )
    print("\n-- Hard gating prediction (supervised) --")
    system_sup.predict_with_moe(new_x, unknown_geometry, soft=False)
    print("\n-- Soft gating prediction (supervised) --")
    system_sup.predict_with_moe(new_x, unknown_geometry, soft=True)

    # ---- Unsupervised Mode ----
    system_unsup = MoEPointNetSystem(
        mode="unsupervised",
        model_settings=model_settings,
        train_points=train_points,
        geo_points=geometry,
        n_points=1024,
        log_level="DEBUG",
    )
    print("\n-- Hard gating prediction (unsupervised) --")
    system_unsup.predict_with_moe(new_x, unknown_geometry, soft=False)
    print("\n-- Soft gating prediction (unsupervised) --")
    system_unsup.predict_with_moe(new_x, unknown_geometry, soft=True)


# Example usage inside a main guard
if __name__ == "__main__":
    example()
