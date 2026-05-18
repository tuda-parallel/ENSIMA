"""
Defines and manages the parameter search space for Bayesian optimization, supporting continuous and discrete variables.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import itertools

import numpy as np

from ensima_optimize.classes.logger import Logger


class SearchSpace:
    def __init__(
        self,
        x_fields: list[str],
        constrains: dict[str, tuple[float, float] | list[float]] = None,
        log_level: str = "WARNING",
    ):
        """
        Initializes the SearchSpace with variable names and their constraints.

        Args:
            x_fields (list[str]): Names of the variables in the search space.
            constrains (dict): Dictionary mapping variable names to either:
                - A tuple (lower, upper) for continuous variables.
                - A list of discrete allowed values.
            log_level (str, optional): Logging level (e.g., "INFO", "DEBUG"). Defaults to "WARNING".
        """
        self.logger = Logger(__name__, level=log_level).get()
        self.x_fields = x_fields
        self.constraints = self.set_constraints(constrains)

    def set_constraints(self, user_constraints=None):
        """
        Sets constraints for design variables with continuous bounds and discrete sets.

        Args:
            user_constraints: Optional dict to override default constraints.
                Keys: variable names.
                Values: either (low, high) for continuous variables or list of discrete allowed values.

        Returns:
            Dict with:
              - 'continuous' keys (np.ndarray shape (n_features,2)) and
              - 'discrete' (dict of variable to list of allowed discrete values).
        """
        default_constraints = {
            "Rp": [
                124,
                125,
                140,
                150,
                152,
                153,
                158,
                165,
                170,
                180,
                185,
                200,
                210,
                220,
                235,
                260,
                263,
                300,
                325,
                336,
                350,
                365,
                420.0,
            ],
            "Fr": (0.01, 10.0),
        }

        fallback = [-np.inf, np.inf]
        discrete_constraints = {}
        continuous_constraints = []

        if user_constraints is None:
            user_constraints = {}

        self.logger.info("Setting constraints for design variables.")
        for var in self.x_fields:
            constraint = user_constraints.get(
                var, default_constraints.get(var, fallback)
            )
            if (
                isinstance(constraint, (list, tuple))
                and len(constraint) == 2
                and all(isinstance(c, (int, float)) for c in constraint)
            ):
                self.logger.debug(
                    f"Variable '{var}' set as continuous with bounds {constraint}."
                )
                continuous_constraints.append(list(constraint))
            elif isinstance(constraint, list):
                self.logger.debug(
                    f"Variable '{var}' set as discrete with values {constraint}."
                )
                discrete_constraints[var] = constraint
                continuous_constraints.append(fallback)  # placeholder bounds
            else:
                self.logger.warning(
                    f"Invalid constraint format for '{var}', using fallback."
                )
                raise ValueError(
                    f"Invalid constraint format for variable '{var}': {constraint}"
                )

        return {
            "continuous": np.array(continuous_constraints),
            "discrete": discrete_constraints,
        }

    def create_sample_space(
        self,
        x: np.ndarray,
        n_samples: int | list[int] = 1000,
        method: str = "grid",
        expansion_factor: float = 0.1,
        precision: int | None = None,
        point_creation: str = "linear",
    ) -> np.ndarray:
        """
        Create sample space with expanded observed min/max per variable, clipped by constraints.
        Supports discrete variables and both grid and random sampling.

        Args:
            x: Observed samples, shape (n_points, n_features).
            n_samples: Number of samples per dimension (for grid) or total samples (for random).
                       Can also be a list specifying per-dimension counts (only for combination mode).
            method: "grid" or "random".
            expansion_factor: Fractional expansion around observed min/max for continuous variables.
            precision (int or None, optional): Number of decimal places to round continuous variables to.
            point_creation: "combination" to generate dense Cartesian product, "linear" for simple grid.
                "linear" mode is faster but suppresses a large portion of the state space (imagine a line vs rectangle).
                "combination" is better, but distorts the plot (several mean lines) and requires more memory.
        Returns:
            np.ndarray: Sampled points, shape (n_samples_total, n_features).
        """

        # IMPORTANT: the search space has the most significant influence on the Problem.
        # Random delivers good results with less points. As the number of points can become
        # a problem , the training handles batch sizes automatically.
        if method not in ("grid", "random"):
            raise ValueError("method must be 'grid' or 'random'")

        self.logger.info(
            f"Creating sample space using method: {method} and {point_creation}"
        )
        discrete_vars = self.constraints["discrete"]
        continuous_bounds = self.constraints["continuous"]
        n_min = 3  # mimum number of steps in combination mode
        if x.size == 0:
            x_dim = len(self.x_fields)
            x_min = np.zeros(x_dim)
            x_max = np.full(x_dim, np.inf)
        else:
            x_dim = x.shape[1]
            x_min = np.min(x, axis=0)
            x_max = np.max(x, axis=0)

        # --- compute effective number of samples per dimension ---
        n_eff = []
        # Support list of sample counts (per-dimension)
        if isinstance(n_samples, (list, tuple, np.ndarray)):
            if len(n_samples) != x_dim:
                raise ValueError(
                    f"Length of n_samples ({len(n_samples)}) must match number of dimensions ({x_dim})."
                )

            if point_creation != "combination":
                self.logger.warning(
                    "List-based n_samples is only used when point_creation='combination'. "
                    "Falling back to uniform sampling in 'linear' mode."
                )
            # sanity check
            for i, var in enumerate(self.x_fields):
                if var in discrete_vars:
                    if len(discrete_vars[var]) != n_samples[i]:
                        self.logger.critical(
                            f"Length of n_samples[{i}] ({n_samples[i]}) must match number of discrete values ({len(discrete_vars[var])} for {var})."
                        )
                        n_samples[i] = len(discrete_vars[var])
            n_eff = list(map(int, n_samples))
            self.logger.info(f"Using per-dimension n_samples: {n_eff}")

        elif point_creation == "linear" or precision is None:
            for dim in range(x_dim):
                n_eff.append(n_samples)
        else:  # combination mode
            for dim in range(x_dim):
                var = self.x_fields[dim]
                if var in discrete_vars:
                    n_eff.append(len(discrete_vars[var]))
                else:
                    c_lower, c_upper = continuous_bounds[dim]
                    x_range = x_max[dim] - x_min[dim]
                    lower = max(x_min[dim] - expansion_factor * x_range, c_lower)
                    upper = min(x_max[dim] + expansion_factor * x_range, c_upper)

                    self.check_precision(
                        var,
                        precision,
                        {
                            "xmin": x_min[dim],
                            "xmax": x_max[dim],
                            "c_lower": c_lower,
                            "c_upper": c_upper,
                        },
                    )

                    step_count = int(round((upper - lower) * (10**precision))) + 1
                    n_eff.append(step_count)
            # 1) reduce the point count if too many or use random selection
            if method == "grid":
                n_reduce = n_eff.copy()
                total_points = np.prod(n_reduce)
                self.logger.debug(
                    f"Original n_eff: {n_eff}, total points: {total_points}, n_samples: {n_samples}, target {x_dim*n_samples}"
                )
                while (
                    total_points > n_samples * x_dim and np.min(n_reduce) > n_min
                ):  # and x_dim <= 3:
                    largest_idx = np.argmax(n_reduce)
                    old_value = n_reduce[largest_idx]
                    n_reduce[largest_idx] = max(10, n_reduce[largest_idx] // 2)
                    total_points = np.prod(n_reduce)
                    self.logger.debug(
                        f"Reduced index {largest_idx} from {old_value} to {n_reduce[largest_idx]}, total_points: {total_points}"
                    )

                self.logger.debug(
                    f"Final reduced n_reduce: {n_reduce}, total_points: {total_points}"
                )

                n_eff = n_reduce
                self.logger.info(f"Updated n_eff: {n_eff}")

        # --- build per-dimension samples ---
        samples_per_dim = []
        for dim in range(x_dim):
            self.logger.info(
                f"Effective number of samples for {self.x_fields[dim]}: {n_eff[dim]}"
            )
            var = self.x_fields[dim]
            c_lower, c_upper = continuous_bounds[dim]
            x_range = x_max[dim] - x_min[dim]

            lower = (
                max(x_min[dim] - expansion_factor * x_range, c_lower)
                if x_range > 0
                else c_lower
            )
            upper = (
                min(x_max[dim] + expansion_factor * x_range, c_upper)
                if x_range > 0
                else (c_upper if c_upper < np.inf else lower + 1.0)
            )

            if var in discrete_vars:
                allowed_vals = np.array(discrete_vars[var])
                if point_creation == "combination":
                    linspace_vals = np.linspace(
                        allowed_vals.min(), allowed_vals.max(), n_eff[dim]
                    )
                    closest_vals = np.array(
                        [
                            allowed_vals[np.abs(allowed_vals - v).argmin()]
                            for v in linspace_vals
                        ]
                    )
                    samples_per_dim.append(closest_vals)
                else:  # linear mode
                    arr = allowed_vals
                    if len(arr) < n_eff[dim]:
                        repeat_count = n_eff[dim] // len(arr)
                        remainder = n_eff[dim] % len(arr)
                        arr = np.concatenate(
                            [np.tile(arr, repeat_count), arr[:remainder]]
                        )
                        self.logger.debug(
                            f"Repeating '{var}' from {len(discrete_vars[var])} -> {n_eff[dim]} points"
                        )
                    samples_per_dim.append(arr)
            else:
                vals = np.linspace(lower, upper, n_eff[dim])
                if precision is not None:
                    vals = np.round(vals, precision)
                    # vals = np.unique(vals)
                    # Ensure minimum resolution to prevent collapse (e.g., all identical values after rounding)
                    if np.unique(vals).size <= 1:
                        # Force at least 3 steps within [lower, upper]
                        vals = np.linspace(lower, upper, n_min)
                        vals = np.round(vals, precision)
                        self.logger.warning(
                            f"Adjusted sampling for '{var}' to avoid degenerate resolution "
                            f"(precision={precision}, lower={lower:.4f}, upper={upper:.4f}, steps={n_min})"
                            f"Alternatively, increase the precision of the x_space"
                        )
                samples_per_dim.append(vals)

        # --- assemble sample space ---
        if point_creation == "combination":
            x_sample_space_full = np.array(list(itertools.product(*samples_per_dim)))
        else:  # linear mode
            samples_array = np.array(samples_per_dim)
            # remove duplicated columns
            x_sample_space_full = np.unique(samples_array, axis=1).T
            self.logger.info(
                f"Removed duplicated samples: {samples_array.shape[1]} --> {x_sample_space_full.shape[0]} (increase precision if you want more non-redundant samples)"
            )
            # x_sample_space_full = samples_array.T

        self.logger.info(f"Size of X sample space: {x_sample_space_full.shape}")

        # --- fallback random sampling ---
        # 2) use random if #1 is commented out
        if method == "random" or x_sample_space_full.shape[0] > n_samples:
            if x_sample_space_full.shape[0] > n_samples:
                self.logger.warning(
                    f"X sample space exceeds points limit {n_samples}. "
                    f"Using random sampling with {n_samples} points."
                )
            self.logger.info(
                f"Using random subset: method={method}, "
                f"total points={x_sample_space_full.shape[0]}, n_samples={n_samples}"
            )
            replace = x_sample_space_full.shape[0] < n_samples
            indices = np.random.choice(
                x_sample_space_full.shape[0], size=n_samples, replace=replace
            )
            x_sample_space = x_sample_space_full[indices]
        else:
            x_sample_space = x_sample_space_full

        return x_sample_space

    def check_precision(self, var: str, precision: int, values: dict) -> None:
        """
        Check whether given values meet the required precision and log warnings if not.

        Args:
            var: Variable name.
            precision: Required precision in decimals.
            values: Dict mapping labels (e.g., 'xmin') to numeric values.
        """
        for label, val in values.items():
            decs = _decimal_places(val)
            if decs > precision:
                self.logger.warning(
                    f"Variable '{var}': {label}={val} has only {decs} decimals, "
                    f"more than requested precision={precision}. "
                    f"{precision} Resolution will be applied."
                )


def _decimal_places(val: float) -> int:
    """Return the number of decimal places of a float."""
    s = f"{val:.16f}".rstrip("0").rstrip(".")
    if "." in s:
        return len(s.split(".")[1])
    return 0


# === Example usage ===
if __name__ == "__main__":
    x_fields = ["Rp", "D", "p"]
    user_constraints = {
        "Rp": [
            124,
            140,
            150,
            165,
            170,
            185,
            200,
            263,
            300,
            325,
            420,
        ],  # discrete values
        "D": (0.1, 5.0),
    }
    logger = Logger(__name__, level="DEBUG").get()
    logger.info(f"Constrains are: {user_constraints}")
    search_space = SearchSpace(x_fields, user_constraints)

    # Assume observed data:
    x_observed = np.array([[270, 1.0, 0.05], [420, 2.0, 0.1], [890, 1.5, 0.07]])

    samples = search_space.create_sample_space(
        x=x_observed,
        n_samples=1000,
        method="grid",
        expansion_factor=0.1,
    )
    logger.info(f"Continuous samples are:\n {samples}")

    random_samples = search_space.create_sample_space(
        x=x_observed,
        n_samples=1000,
        method="random",
        expansion_factor=0.1,
    )
    logger.info(f"Discrete samples are:\n {random_samples}")
