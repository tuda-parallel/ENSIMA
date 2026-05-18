"""
Expert model wrapping a Gaussian Process with an associated label for Mixture-of-Experts integration.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np

from ensima_optimize.classes.model import Model


class Expert(Model):
    """Expert model inheriting from Model with an associated label.

    This class wraps a Gaussian Process model with an expert label and
    formats the prediction output to include the label.

    Args:
        label (str): Identifier for the expert.
        x (np.ndarray): Input training data.
        y (np.ndarray): Output training data.
        backend (str): Backend to use ('torch' or 'sklearn'). Defaults to 'torch'.
        log_level (str): Logging verbosity level. Defaults to 'INFO'.
        normalize_y (bool): Whether to normalize the output. Defaults to True.
        model_filename (str): Path to store or load the model. Defaults to 'gp_model.pkl'.
        optimizer_restart (int): Number of optimizer restarts. Defaults to 5.
        epochs (int): Number of training epochs for torch backend. Defaults to 100.
        save_and_load (bool): Whether to save and load the model from disk. Defaults to False.
        hidden_labels (list[str]): List of hidden labels. Defaults to None.
        latent_input_dim: [PyTorch only] Latent input dimensionality for multitask GPs.
        latent_output_dim: [PyTorch only] Latent output dimensionality for multitask GPs.
    """

    def __init__(
        self,
        label: str,
        x: np.ndarray,
        y: np.ndarray,
        backend: str = "torch",
        log_level: str = "INFO",
        normalize_y: bool = True,
        model_filename: str = "gp_model.pkl",
        optimizer_restart: int = 5,
        epochs: int = 100,
        save_and_load: bool = False,
        hidden_labels=None,
        precision: int = 2,
        latent_input_dim: int = None,
        latent_output_dim: int = None,
        device: str = "cpu",
    ):
        self.label = label
        # hidden_labels stores the original part names contained in a cluster.
        # This is primarily used for diagnostic logging in unsupervised mode
        # to show which parts were grouped together.
        if hidden_labels is None:
            hidden_labels = [label]
        self.training_points = len(x)
        self.hidden_labels = hidden_labels
        super().__init__(
            x,
            y,
            backend=backend,
            log_level=log_level,
            normalize_y=normalize_y,
            model_filename=f"gp_{label}_model.pkl",
            optimizer_restart=optimizer_restart,
            epochs=epochs,
            save_and_load=save_and_load,
            precision=precision,
            latent_input_dim=latent_input_dim,
            latent_output_dim=latent_output_dim,
            device=device,
        )

    def predict(self, *args, **kwargs):
        """Generate prediction and prepend the expert label.

        Returns:
            Tuple of mean and standard deviation predictions.
        """

        return super().predict(*args, **kwargs)

    def train(self, *args, **kwargs):
        """train expert

        Returns:
           None
        """
        super().train(*args, **kwargs)
        self.training_points = len(self.x)
