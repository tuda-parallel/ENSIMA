"""
Unified Gaussian Process Regression (GPR) module supporting both scikit-learn and GPyTorch backends.

This module provides a flexible `Model` class for training, inference, and checkpointing of Gaussian Process
regression models. It supports both single-task regression using scikit-learn’s GaussianProcessRegressor
and multi-task regression using GPyTorch’s multitask framework with customizable kernels and mean functions.

Key features include:
- Single-task and multi-task GPR with task correlation modeling.
- Choice of backend: 'sklearn' for simpler cases, 'torch' for scalable multi-task models.
- Custom latent coupled mean function to capture shared latent structure across tasks.
- Model persistence with save/load functionality.
- Input and output normalization for numerical stability.
- Integrated logging for training and prediction metrics.
- Easy integration into Bayesian optimization and other ML pipelines.

Classes:
    GPyTorchMultitaskModel: Multi-task GP model using GPyTorch with Matern kernel and task correlations.
    LatentCoupledMean: Custom mean function modeling outputs as projections from a latent space.
    Model: Unified interface for training, inference, and persistence with scikit-learn or GPyTorch backends.

Args:
    x (np.ndarray): Training inputs, shape (n_samples, n_features).
    y (np.ndarray): Training targets, shape (n_samples,) for single-task or (n_samples, n_tasks) for multi-task.
    backend (str, optional): Backend to use, either 'sklearn' or 'torch'. Defaults to 'sklearn'.

Example:
    >>> model = Model(x, y, backend="torch")
    >>> model.fit()
    >>> mean, std = model.predict(x)
    >>> model._save_model(filepath="checkpoint.pkl")
    >>> model._load_model(filepath="checkpoint.pkl")

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import os
import pickle
import random
import time
import warnings

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.utils.warnings import GPInputWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ensima_optimize.classes.logger import Logger

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for GPU
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for CUDA >= 10.2
    os.environ["PYTHONHASHSEED"] = str(seed)


# for determinism
seed_everything(42)


class LatentInputEncoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class SimpleLCMKernel(gpytorch.kernels.Kernel):
    """
    Linear Model of Coregionalization kernel.

    Combines multiple latent kernels weighted by task-specific coefficients
    to model task correlations in a flexible way.
    """

    def __init__(self, base_kernels, num_tasks, rank):
        super().__init__()
        self.base_kernels = torch.nn.ModuleList(base_kernels)
        self.num_tasks = num_tasks
        self.rank = rank
        self.task_weights = torch.nn.Parameter(torch.randn(num_tasks, rank))

    def forward(self, x1, x2, diag=False, **params):
        covar = 0
        for r, base_kernel in enumerate(self.base_kernels):
            base_covar = base_kernel(x1, x2, diag=diag, **params)
            weight = self.task_weights[:, r].unsqueeze(1) @ self.task_weights[
                :, r
            ].unsqueeze(0)
            covar = covar + base_covar.mul(weight)
        return covar


class LatentCoupledMean(torch.nn.Module):
    """Latent coupled mean function for multi-task Gaussian Process regression.

    Models multi-task outputs as projections from a shared latent representation, enabling the model
    to capture correlations across tasks in the mean function space.

    Args:
        input_dim (int): Number of input features.
        num_tasks (int): Number of output tasks.
        rank (int, optional): Dimensionality of the latent space. Defaults to 5.

    Returns:
        torch.Tensor: Mean predictions of shape (batch_size, num_tasks).

    Example:
        >>> mean_module = LatentCoupledMean(input_dim=10, num_tasks=3)
        >>> x = torch.randn(5, 10)
        >>> mean_preds = mean_module(x)
    """

    def __init__(self, input_dim: int, num_tasks: int, rank: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, rank),
            torch.nn.ReLU(),
            torch.nn.Linear(rank, num_tasks),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GPyTorchMultitaskModel(gpytorch.models.ExactGP):
    """
    Multitask GP model supporting three configurations via the 'method' argument.

    Supports multiple correlated output tasks with flexible mean and covariance:

    Methods:
      - 'independent':
          Independent zero mean per task with MultitaskKernel covariance using a Matern base kernel.
      - 'coupled_mean':
          Coupled mean across tasks via a latent projection, covariance as in 'independent'.
      - 'lcm':
          Independent zero mean, covariance via Linear Model of Coregionalization (LCM) kernel
          combining multiple Matern kernels.

      | Method          | Mean function                         | Covariance kernel                          | Expressiveness                              | When to use                                      |
      |-----------------|----------------------------------------|-------------------------------------------|---------------------------------------------|--------------------------------------------------|
      | 'independent'   | Independent zero mean per task         | MultitaskKernel (with one Matern kernel)  | Moderate: models task covariance only       | Tasks share covariance, not mean                |
      | 'coupled_mean'  | Latent coupled mean across tasks       | MultitaskKernel (with one Matern kernel)  | Higher: models shared mean and covariance   | Tasks share structure in both mean & covariance |
      | 'lcm'           | Independent zero mean per task         | LCMKernel (linear sum of Matern kernels)  | Highest: flexible multi-kernel covariance   | Complex task covariance structure needed        |
      ---------------------
        Args:
        train_x (torch.Tensor): Training inputs, shape (n_samples, n_features).
        train_y (torch.Tensor): Training targets, shape (n_samples * num_tasks,) or (n_samples, num_tasks).
        likelihood (gpytorch.likelihoods.Likelihood): Likelihood module, e.g., MultitaskGaussianLikelihood.
        num_tasks (int): Number of output tasks.
        method (str, optional): One of {'independent', 'coupled_mean', 'lcm'}. Defaults to 'lcm'.
        latent_output_dim (int, optional): Dimensionality of latent projection for mean/covariance modules.
            Defaults to `num_tasks`.
        latent_input_dim (int, optional): Dimensionality of input features. If not provided, inferred from `train_x`.
        log_level (str, optional): Logging level for diagnostics. Defaults to 'INFO'.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.Likelihood,
        num_tasks: int,
        latent_output_dim: int = None,
        latent_input_dim: int = None,
        method: str = "lcm",
        log_level: str = "INFO",
    ):
        super().__init__(train_x, train_y, likelihood)
        self.logger = Logger(__name__, level=log_level).get()
        self.method = method
        self.dim_output = num_tasks
        if latent_output_dim is None:
            latent_output_dim = num_tasks
        self.logger.debug(f"Latent output dimension: {latent_output_dim}")

        if latent_input_dim is None:
            input_dim = train_x.shape[-1]
            self.logger.debug(f"Input dimension (default): {input_dim}")
        else:  # @GPyTorchMultitaskModelWithLatentInput overwrites this if needed
            input_dim = latent_input_dim
            self.logger.debug(f"Latent input dimension: {input_dim}")

        self.logger.debug(f"Method for mean/covariance configuration: {self.method}")
        if self.method == "coupled_mean":
            self.mean_module = LatentCoupledMean(
                input_dim=input_dim,
                num_tasks=num_tasks,
                rank=latent_output_dim,
            )
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
                lengthscale_constraint=gpytorch.constraints.Interval(1e-4, 1e2),
            )
            base_kernel.register_prior(
                "lengthscale_prior",
                gpytorch.priors.SmoothedBoxPrior(0.05, 2.0, sigma=0.1),
                "lengthscale",
            )
            base_kernel.lengthscale = torch.ones(input_dim) * 0.5
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=num_tasks, rank=latent_output_dim
            )

        elif self.method == "independent":
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=num_tasks
            )
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=input_dim,
                lengthscale_constraint=gpytorch.constraints.Interval(1e-4, 1e2),
            )
            base_kernel.register_prior(
                "lengthscale_prior",
                gpytorch.priors.SmoothedBoxPrior(0.05, 2.0, sigma=0.1),
                "lengthscale",
            )
            base_kernel.lengthscale = torch.ones(input_dim) * 0.5
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel, num_tasks=latent_output_dim, rank=latent_output_dim
            )

        elif self.method == "lcm":
            # Use LatentCoupledMean instead
            self.mean_module = LatentCoupledMean(
                input_dim=input_dim,
                num_tasks=num_tasks,
                rank=latent_output_dim,
            )
            # Use ZeroMean
            # self.mean_module = gpytorch.means.MultitaskMean(
            #     gpytorch.means.ConstantMean(),  # or custom task-specific means
            #     num_tasks=num_tasks,
            # )
            base_kernels = []
            for _ in range(latent_output_dim):
                kern = gpytorch.kernels.MaternKernel(
                    nu=2.5,
                    ard_num_dims=input_dim,
                    lengthscale_constraint=gpytorch.constraints.Interval(1e-4, 1e2),
                )
                kern.register_prior(
                    "lengthscale_prior",
                    gpytorch.priors.SmoothedBoxPrior(0.05, 2.0, sigma=0.1),
                    "lengthscale",
                )
                kern.lengthscale = torch.ones(input_dim) * 0.5
                base_kernels.append(kern)

            self.covar_module = gpytorch.kernels.LCMKernel(
                base_kernels, num_tasks=latent_output_dim
            )

        else:
            raise ValueError(
                f"Unknown method {self.method}, choose from 'independent', 'coupled_mean', 'lcm'."
            )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """Forward pass through the model.

        Args:
            x: Input tensor.

        Returns:
            MultitaskMultivariateNormal distribution.
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultitaskMultivariateNormal(mean, covar)


class GPyTorchMultitaskModelWithLatentInput(GPyTorchMultitaskModel):
    """
    Multi-task GP model extending GPyTorchMultitaskModel by adding a latent input encoder.

    This model first transforms inputs via a learned latent encoder network, then
    feeds the encoded latent representation to the GP mean and covariance modules.

    Args:
        *args: Positional arguments forwarded to GPyTorchMultitaskModel constructor.
        latent_input_dim (int): Dimension of the latent input space.
        **kwargs: Keyword arguments forwarded to GPyTorchMultitaskModel constructor.

    Attributes:
        latent_encoder (torch.nn.Module): Neural network encoding original inputs into latent inputs.
    """

    def __init__(self, *args, latent_input_dim: int = None, **kwargs):
        # Extract train_x from args[0] before super init
        train_x = args[0]  # First positional argument is train_x
        input_dim = train_x.shape[-1]

        if latent_input_dim is None:
            latent_input_dim = 2 * input_dim
            # latent_input_dim = input_dim

        # Store latent_input_dim in kwargs for parent class to use
        kwargs["latent_input_dim"] = latent_input_dim

        # Call parent constructor
        super().__init__(*args, **kwargs)

        # Define latent encoder network
        self.latent_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, latent_input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultitaskMultivariateNormal:
        """
        Forward pass through latent encoder and GP modules.

        Args:
            x (torch.Tensor): Original input tensor of shape (N, input_dim)

        Returns:
            MultitaskMultivariateNormal: GP prediction distribution over tasks.
        """
        latent_x = self.latent_encoder(x)
        return super().forward(latent_x)


class Model:
    """Unified interface for training and inference using sklearn or PyTorch Gaussian Process models."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        backend: str = "torch",
        log_level: str = "INFO",
        normalize_y: bool = True,
        model_filename: str = "gp_model.pkl",
        optimizer_restart: int = 5,
        epochs: int = 100,
        save_and_load: bool = False,
        precision: int = 2,
        latent_input_dim: int = None,
        latent_output_dim: int = None,
        device: str = "cpu",
    ):
        """Initialize the Model.

        Args:
            x: Input training data.
            y: Output training data.
            backend: Either 'torch' or 'sklearn'.
            log_level: Logging verbosity level.
            normalize_y: Whether to normalize the output.
            model_filename: Path to store or load the model.
            optimizer_restart: Number of restarts for optimizer.
            epochs: Number of training epochs for torch backend.
            save_and_load: If True, the model will be saved and loaded from disk.
            latent_input_dim: [PyTorch only] Latent input dimensionality for multitask GPs.
            latent_output_dim: [PyTorch only] Latent output dimensionality for multitask GPs.
            device: [PyTorch only] Device to use for training.
        """
        self.backend = backend
        self.x = x
        self.y = y
        self.normalize_y = normalize_y
        self.filename = os.path.abspath(model_filename)
        self.epochs = epochs
        self.save_and_load = save_and_load
        self.optimizer_restart = optimizer_restart
        self.log_level = log_level
        self.logger = Logger(__name__, level=self.log_level).get()
        self.initial_noise_variance = 0.1  # 1
        self.precision = precision
        self.device = device

        # PyTorch-only arguments
        if backend == "torch":
            self.latent_input_dim = latent_input_dim
            self.latent_output_dim = latent_output_dim
        else:
            if latent_input_dim is not None or latent_output_dim is not None:
                self.logger.warning(
                    "latent_input_dim and latent_output_dim are only used with the PyTorch backend. "
                    "They will be ignored for sklearn."
                )
            self.latent_input_dim = None
            self.latent_output_dim = None

        self.logger.info("Checking if a snapshot model exists")
        if len(self.x) > 0:
            model, need_retrain = self._load_model()
            if model is not None and not need_retrain:
                self.model = model
                self.logger.info("Loaded model from checkpoint.")
            else:
                self.logger.info("Training model from scratch.")
                self.train()
                self._save_model()
        else:  # skip training
            self.logger.info("Model not trained as not enough points are available.")

    def train(
        self, x: np.ndarray = None, y: np.ndarray = None, epochs: int = None
    ) -> None:
        """Train the GP model using the selected backend.

        Args:
            x: Optional new input training data.
            y: Optional new output training data.
            epochs (int): Optional number of training epochs. If not provided, the default value is used.
        """
        if x is not None:
            self.x = x
            # self.logger.debug("Using new x for training")
        if y is not None:
            self.y = y
            # self.logger.debug("Using new y for training")
        if self.backend == "sklearn":
            self._train_sklearn()
        elif self.backend == "torch":
            if epochs is not None:
                self.epochs = epochs
                self.logger.debug(f"Using new epochs: {self.epochs}")
            self._train_pytorch()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _train_sklearn(self) -> None:
        """Train using sklearn GaussianProcessRegressor."""
        kernel = ConstantKernel(10, (1e-4, 1e4)) * Matern(
            length_scale=0.1, length_scale_bounds=(1e-1, 1e3), nu=2.5
        ) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-4, 1e1))

        model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.optimizer_restart,
            normalize_y=self.normalize_y,
        )

        start = time.time()
        model.fit(self.x, self.y)
        self.logger.info(f"Finished GPR fitting in {time.time() - start:.2f} seconds")
        self.model = model

    def _train_pytorch(self) -> None:
        # GPU support
        self.logger.info(f"Using device: {self.device}")

        """Train using GPyTorch multitask model."""
        x = torch.tensor(self.x, dtype=torch.float32, device=self.device)
        y = torch.tensor(self.y, dtype=torch.float32, device=self.device)

        if y.ndim != 2:
            raise ValueError(
                "Multitask GP requires y to be a 2D tensor (num_samples x num_tasks)"
            )

        self.dim_output = y.shape[1]
        self.logger.debug(f"Output dim: {self.dim_output}")
        self.logger.debug(f"Input dim: {x.shape[1]}")
        if self.normalize_y:
            self.y_mean = y.mean(dim=0)
            self.y_std = y.std(dim=0)
            y = (y - self.y_mean) / self.y_std

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=self.dim_output,
            has_task_noise=True,
        )
        # self.likelihood.task_noises.data.fill_(self.initial_noise_variance)
        self.likelihood.register_prior(
            "noise_prior",
            gpytorch.priors.SmoothedBoxPrior(1e-3, 0.5, sigma=0.1),
            "task_noises",
        )

        # Constrain raw task noises
        # IMPORTANT: This is the most important parameter for fitting
        self.likelihood.register_constraint("raw_task_noises", Interval(1e-5, 1))
        # self.likelihood.register_constraint("raw_task_noises", Interval(1e-2, 2e0))

        with torch.no_grad():
            self.likelihood.raw_task_noises.fill_(
                np.log(np.exp(self.initial_noise_variance) - 1)
            )

        # check duplicated
        ##################
        x_cpu = x.cpu()

        # Find unique rows and indices
        uniq_x, inverse_indices, counts = torch.unique(
            x_cpu, dim=0, return_inverse=True, return_counts=True
        )
        total_duplicates = x_cpu.shape[0] - len(uniq_x)
        if total_duplicates > 0:
            self.logger.warning(f"Duplicate points detected: {total_duplicates}")
            self.logger.warning(f"Number of unique points: {len(uniq_x)}")

            # Remove duplicates
            # self.logger.warning(f"Old x: {x_cpu.shape}, Old y: {y_cpu.shape}")
            # self.logger.warning(
            #     f"Points will be reduced from {x_cpu.shape[0]} to {len(uniq_x)}"
            # )
            # # Aggregate y values for duplicates by taking the mean
            # y_unique = torch.zeros((len(uniq_x), y_cpu.shape[1]), dtype=y_cpu.dtype)
            # for i in range(len(uniq_x)):
            #     y_unique[i] = y_cpu[inverse_indices == i].mean(dim=0)
            #
            # x = uniq_x.to(self.device)
            # y = y_unique.to(self.device)
            # self.logger.warning(f"New x: {x.shape}, New y: {y.shape}")

        # self.model = GPyTorchMultitaskModel(
        #     x, y, self.likelihood, num_tasks=self.num_outputs, log_level=self.log_level,
        # )
        self.model = GPyTorchMultitaskModelWithLatentInput(
            x,
            y,
            self.likelihood,
            num_tasks=self.dim_output,
            log_level=self.log_level,
            latent_input_dim=self.latent_input_dim,
            latent_output_dim=self.latent_output_dim,
        ).to(self.device)
        self.likelihood = self.likelihood.to(self.device)

        # Move cleaned data to the device for training
        self.x_train = x
        self.y_train = y

        # train
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        print_every = max(1, self.epochs // 10)
        # Early stopping config
        prev_loss = None
        prev_noise = None
        tolerance = 1e-3  # Minimum required loss improvement
        noise_drop_limit = 0.5  # Max allowed noise drop ratio

        jitter_value = 10 ** -(self.precision - 2)
        start = time.time()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            # Apply Cholesky jitter for numerical stability
            with gpytorch.settings.cholesky_jitter(jitter_value):
                output = self.model(x)
                loss = -mll(output, y)
                loss.backward()
                optimizer.step()

            loss_val = loss.item()
            noise_val = self.likelihood.noise.item()

            # Logging
            if epoch % print_every == 0 or epoch == self.epochs - 1:
                self.logger.info(
                    f"[Epoch {epoch}/{self.epochs}] Loss: {loss_val:.4f}, Likelihood Noise: {noise_val:.8f}"
                )

            # Early stopping condition
            # 1) not converging
            if not torch.isfinite(loss):  # catches NaN/inf from non-PSD kernel
                self.logger.warning(
                    f"Stopping early at epoch {epoch} due to non-finite loss (numerical instability)."
                )
                break

            if prev_loss is not None and prev_noise is not None:
                loss_diff = prev_loss - loss_val
                noise_ratio = noise_val / prev_noise

                # 2) Loss increase
                if loss_val - prev_loss > 0.5:
                    self.logger.warning(
                        f"Stopping early at epoch {epoch} because loss increased "
                        f"(prev_loss={prev_loss:.6f}, current_loss={loss_val:.6f})"
                    )
                    break

                # 3) Small improvement + noise drop
                if loss_diff < tolerance and noise_ratio < noise_drop_limit:
                    self.logger.warning(
                        f"Stopping early at epoch {epoch} due to diminishing loss improvement "
                        f"and rapidly decreasing noise (Δloss={loss_diff:.4f}, noise ratio={noise_ratio:.4f})"
                    )
                    break

            prev_loss = loss_val
            prev_noise = noise_val

        self.logger.info(
            f"Finished multitask GPR fitting in {time.time() - start:.2f} seconds"
        )

        # --- Post-Training Diagnostics ---
        self.model.eval()
        self.likelihood.eval()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=GPInputWarning)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                preds = self.model(self.x_train)
                means = preds.mean.detach().cpu().numpy()
                variances = preds.variance.detach().cpu().numpy()

        self.logger.debug(
            f"Predicted mean: min={means.min():.4f}, max={means.max():.4f}, mean={means.mean():.4f}"
        )
        self.logger.debug(
            f"Predicted variance: min={variances.min():.4f}, max={variances.max():.4f}, mean={variances.mean():.4f}"
        )

    def predict(
        self, x_test: np.ndarray, batch_size: int = 1024, return_cov: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the trained GP model with optional full output covariance.

        Args:
            x_test: Test input data, shape (n_test, d)
            batch_size: For torch backend, number of points per batch.
            return_cov: If True, return full covariance matrices for each test point.

        Returns:
            Tuple (mean, std_or_cov)
            - mean: (n_test, m)
            - std_or_cov:
                - If return_cov=False: (n_test, m) marginal std
                - If return_cov=True: (n_test, m, m) full covariance per test point
        """
        x_test = np.atleast_2d(x_test)
        mean_list, cov_list = [], []

        if self.backend == "sklearn":
            mean = self.model.predict(x_test)
            if return_cov:
                # sklearn doesn't provide full multi-output covariance easily
                raise NotImplementedError(
                    "Full covariance not implemented for sklearn backend"
                )
            else:
                std = np.ones_like(mean)  # placeholder if needed
                return mean, std

        elif self.backend == "torch":
            self.model.eval()
            self.likelihood.eval()
            n_samples = x_test.shape[0]

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = torch.tensor(
                    x_test[start:end], dtype=torch.float32, device=self.device
                )

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = self.likelihood(self.model(x_batch))

                    batch_mean = pred.mean.cpu().numpy()  # (batch, m)
                    if return_cov:
                        full_cov = (
                            pred.covariance_matrix
                        )  # shape (batch_size*m, batch_size*m)
                        batch_size_current = x_batch.shape[0]
                        m = self.dim_output

                        # Extract per-point covariance: block-diagonal of m x m
                        batch_cov = torch.zeros(
                            (batch_size_current, m, m), device=full_cov.device
                        )
                        for i in range(batch_size_current):
                            idx = slice(i * m, (i + 1) * m)
                            batch_cov[i] = full_cov[idx, idx]

                        cov_list.append(batch_cov.cpu().numpy())
                    else:
                        batch_std = pred.variance.sqrt().cpu().numpy()
                        cov_list.append(batch_std)

                    mean_list.append(batch_mean)

            mean = np.concatenate(mean_list, axis=0)
            cov_or_std = np.concatenate(cov_list, axis=0)

            if self.normalize_y:
                mean = mean * self.y_std.cpu().numpy()
                if not return_cov:
                    cov_or_std = cov_or_std * self.y_std.cpu().numpy()
                else:
                    # scale full covariance: Cov(Y_scaled) = diag(std_y) * Cov(Y) * diag(std_y)
                    scale = np.diag(self.y_std.cpu().numpy())
                    cov_or_std = cov_or_std @ scale
                    cov_or_std = scale @ cov_or_std

            return mean, cov_or_std

    def _load_model(self) -> tuple[object, bool]:
        """Load model from file if the training data matches.

        Returns:
            A tuple of the model and a boolean indicating if retraining is needed.
        """
        self.logger.info(
            f"Trying to load model and training data from '{self.filename}'..."
        )
        model, need_retrain = None, True
        if self.save_and_load:
            try:
                if self.backend == "torch":
                    checkpoint = torch.load(self.filename, weights_only=False)
                    x_train = checkpoint["x_train"]
                    y_train = checkpoint["y_train"]

                    if np.array_equal(x_train, self.x) and np.array_equal(
                        y_train, self.y
                    ):
                        x_tensor = torch.tensor(self.x, dtype=torch.float32)
                        y_tensor = torch.tensor(self.y, dtype=torch.float32)

                        if checkpoint["normalize_y"]:
                            self.y_mean = checkpoint["y_mean"]
                            self.y_std = checkpoint["y_std"]
                            y_tensor = (y_tensor - self.y_mean) / self.y_std

                        self.dim_output = checkpoint["num_tasks"]
                        self.likelihood = (
                            gpytorch.likelihoods.MultitaskGaussianLikelihood(
                                num_tasks=self.dim_output,
                                has_task_noise=True,
                            )
                        )
                        self.likelihood.task_noises.data.fill_(
                            self.initial_noise_variance
                        )

                        # model = GPyTorchMultitaskModel(
                        #     x_tensor, y_tensor, self.likelihood, num_tasks=self.num_tasks
                        # )
                        model = GPyTorchMultitaskModelWithLatentInput(
                            x_tensor,
                            y_tensor,
                            self.likelihood,
                            num_tasks=self.dim_output,
                            latent_input_dim=None,
                            log_level=self.log_level,
                        )
                        model.load_state_dict(checkpoint["model_state_dict"])
                        self.likelihood.load_state_dict(
                            checkpoint["likelihood_state_dict"]
                        )

                        self.x_train = x_tensor
                        self.y_train = y_tensor
                        self.logger.info(
                            "Torch model loaded successfully with matching training data."
                        )
                        need_retrain = False

                elif self.backend == "sklearn":
                    with open(self.filename, "rb") as f:
                        data = pickle.load(f)

                    x_train = data["x_train"]
                    y_train = data["y_train"]

                    if np.array_equal(x_train, self.x) and np.array_equal(
                        y_train, self.y
                    ):
                        model = data["model"]
                        self.logger.info(
                            "Sklearn model loaded successfully with matching training data."
                        )
                        need_retrain = False

            except Exception as e:
                self.logger.info(f"Could not load model file '{self.filename}': {e}")

        return model, need_retrain

    def _save_model(self) -> None:
        """Save the model and training data to disk."""
        if self.save_and_load:
            if self.backend == "torch":
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "likelihood_state_dict": self.likelihood.state_dict(),
                        "x_train": self.x,
                        "y_train": self.y,
                        "normalize_y": self.normalize_y,
                        "y_mean": getattr(self, "y_mean", None),
                        "y_std": getattr(self, "y_std", None),
                        "num_tasks": getattr(self, "num_tasks", 1),
                    },
                    self.filename,
                )
            else:
                data = {"model": self.model, "x_train": self.x, "y_train": self.y}
                with open(self.filename, "wb") as f:
                    pickle.dump(data, f)
            self.logger.info(f"Model and training data saved to '{self.filename}'")
        else:
            self.logger.info("Model and training data not saved")

    def __repr__(self) -> str:
        """Return a human-readable representation of the model."""
        trained = False
        kernel = None

        if self.backend == "sklearn":
            trained = hasattr(self.model, "X_train_") and hasattr(
                self.model, "y_train_"
            )
            kernel = getattr(self.model, "kernel_", getattr(self.model, "kernel", None))
        elif self.backend == "torch":
            trained = hasattr(self, "x_train") and hasattr(self, "y_train")
            kernel = getattr(
                getattr(self.model, "covar_module", None), "data_covar_module", None
            )
            kernel = kernel.__repr__() if kernel is not None else "Unknown Kernel"

        obj_func = (
            "set"
            if hasattr(self, "objective_function") and self.objective_function
            else "None"
        )

        return (
            f"{self.__class__.__name__}(\n"
            f"  x_shape={self.x.shape if hasattr(self, 'x') else 'N/A'},\n"
            f"  y_shape={self.y.shape if hasattr(self, 'y') else 'N/A'},\n"
            f"  backend='{self.backend}',\n"
            f"  objective_function={obj_func},\n"
            f"  gp_kernel={kernel},\n"
            f"  model trained={trained}\n"
            f")"
        )
