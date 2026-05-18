"""
Identifies the best candidate from multi-objective output arrays using Pareto and attention-based strategies.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

import numpy as np


def find_min_or_max(y, x, args, logger=None, mode: str = "minimization") -> int:
    """
    Find the best row in `y` with a fallback strategy.
    If a user-defined target vector is provided, it overrides Pareto logic.

    Args:
        y (np.ndarray): 2D array of objective values.
        x (np.ndarray): 2D array of corresponding input variables.
        args (Namespace): Arguments containing attributes like 'target' and 'attention_coefficients'.
        logger (logging.Logger, optional): Logger instance for debug/info messages. Defaults to None.
        mode (str, optional): "minimization" (default) or "maximization".

    Returns:
        int: Index of the best row in `y`.
    """
    if mode not in {"minimization", "maximization"}:
        raise ValueError(f"Unsupported mode: {mode}")

    idx_best = None  # will be set in all cases
    if y.size == 0 or x.size == 0:
        if logger is not None:
            logger.warning("y or x is empty: cold start, returning None for idx_best")
        return idx_best

    # Handle target vector
    target_vec = None
    if getattr(args, "target", None) is not None:
        target_vec = np.array(args.target, dtype=float) * args.attention_coefficients
        if logger is not None:
            logger.debug(f"Target vector: {args.target}")
            logger.debug(f"Target vector inside model: {target_vec}")

    if logger is not None:
        logger.debug(
            f"finding y_best using {mode} with {y.shape[0]} rows and {y.shape[1]} dimensions"
        )

    if target_vec is not None:
        # Distance-based selection (no Pareto needed)
        if mode == "minimization":
            diffs = y - target_vec
        else:  # maximization
            diffs = target_vec - y

        # Optional neutral-dimension handling
        # diffs[:, target_vec == 100] = 0

        distances = np.linalg.norm(diffs, axis=1)
        idx_best = np.argmin(distances)
        if logger is not None:
            logger.debug(
                f"Best index based on target {idx_best}, y in model: {y[idx_best]}"
            )
            coeffs_safe = np.where(
                args.attention_coefficients == 0,
                1e12,
                args.attention_coefficients,
            )
            logger.info(
                f"Best index based on target {idx_best}, y: {y[idx_best]/coeffs_safe}"
            )

    elif mode == "minimization":
        # Step 1: strict component-wise minimum
        mins = np.min(y, axis=0)
        mask = np.all(y == mins, axis=1)
        if mask.any():
            idx_best = np.where(mask)[0][0]
            if logger is not None:
                logger.info(f"Found strict component-wise minimum at index {idx_best}")
        else:
            if logger is not None:
                logger.debug("No strict min found. Falling back to Pareto.")

            # Step 2: Pareto non-dominated set
            le = y[:, None, :] <= y[None, :, :]
            lt = y[:, None, :] < y[None, :, :]
            dominates = np.all(le, axis=2) & np.any(lt, axis=2)
            is_dominated = np.any(dominates, axis=0)
            non_dom_idxs = np.where(~is_dominated)[0]

            if non_dom_idxs.size == 0:
                idx_best = np.argmin(np.max(y, axis=1))
                if logger is not None:
                    logger.info(
                        f"No non-dominated rows; using global min-max row at index {idx_best}"
                    )
            else:
                idx_best = non_dom_idxs[np.argmin(np.max(y[non_dom_idxs], axis=1))]
                if logger is not None:
                    logger.info(f"Selected Pareto-optimal row at index {idx_best}")

    else:  # maximization
        # Step 1: strict component-wise maximum
        maxs = np.max(y, axis=0)
        mask = np.all(y == maxs, axis=1)
        if mask.any():
            idx_best = np.where(mask)[0][0]
            if logger is not None:
                logger.info(f"Found strict component-wise maximum at index {idx_best}")
        else:
            if logger is not None:
                logger.debug("No strict max found. Falling back to Pareto.")

            # Step 2: Pareto non-dominated set
            ge = y[:, None, :] >= y[None, :, :]
            gt = y[:, None, :] > y[None, :, :]
            dominates = np.all(ge, axis=2) & np.any(gt, axis=2)
            is_dominated = np.any(dominates, axis=0)
            non_dom_idxs = np.where(~is_dominated)[0]

            if non_dom_idxs.size == 0:
                idx_best = np.argmax(np.min(y, axis=1))
                if logger is not None:
                    logger.info(
                        f"No non-dominated rows; using global max-min row at index {idx_best}"
                    )
            else:
                idx_best = non_dom_idxs[np.argmax(np.min(y[non_dom_idxs], axis=1))]
                if logger is not None:
                    logger.info(f"Selected Pareto-optimal row at index {idx_best}")

    if logger is not None:
        logger.debug(f"Best row (y): {y[idx_best]}")
        logger.debug(f"Corresponding x: {x[idx_best]}")

    return idx_best
