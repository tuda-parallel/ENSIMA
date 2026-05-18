"""
Main entry point and orchestration logic for the ENSIMA Bayesian optimization loop.

Author: Ahmad Tarraf
Copyright (c) 2025-2026 TU Darmstadt, Germany
Version: 0.0.1
Date: May 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/ENSIMA/blob/main/LICENSE
"""

# import numpy as np
import numbers
import socket

from ensima.classes.bayesian_optimization import BayesianOptimization
from ensima.classes.license_server import LicenseServer
from ensima.classes.logger import Logger
from ensima.helpers.objective_function import dummy_init
from ensima.helpers.parse_args import parse_arguments
from ensima.helpers.plot import plot
from ensima.helpers.read_data import read_data_type


def main(
    args=None,
    type_filter: bool = True,
    parts: list[tuple[str, str]] = None,
    type_number: int = None,
) -> None:
    """main function for optimization. sets up initial data and executes bayesian optimization"""
    # Set default arguments when no args are passed
    if args is None:
        args = parse_arguments()

    # Dummy or real run
    if not args.jobname:
        print("dummy run with predefined optimization function")
        # create initial data set
        x, y, objective_function = dummy_init()

        # attention_coefficients = np.random.uniform(-1, 1, size=y.shape[1])
        # attention_coefficients = 0.5 * np.ones(y.shape[1])

        # Create BayesianOptimization instance
        bayes_opt = BayesianOptimization(
            args,
            x,
            y,
            objective_function=objective_function,
            # attention_coefficients=attention_coefficients,
        )
        bayes_opt.optimize(args.iterations, args.parallel_samples)
    else:
        # Init
        logger = Logger(__name__, level=args.log_level).get()
        logger.info(f"Logging to {args.log_file}")
        logger.info(f"Saving results to {args.result_folder}")
        # adjust cores for parallel samples
        args.cores = args.cores // args.parallel_samples

        # Print passed args
        logger.debug("Args are:")
        for key, value in args.__dict__.items():
            logger.debug(f"{key}: {value}")

        # Create initial data set
        logger.info("reading data")
        x, y, types = read_data_type(
            args.output, args.x_fields, args.y_fields, log_level=args.log_level
        )

        if parts is not None:
            # remove parts that are not in the dataset for moe
            for jobname, part in parts:
                if jobname not in types:
                    logger.warning(f"Skipping {part} because it is not in the dataset")
                    parts.remove((jobname, part))

        # take only one type or part and drop other (no moe)
        else:
            if type_filter:
                if len(types) > 0:
                    if isinstance(types[-1], numbers.Number):
                        # uses complexity to find the part number, does not require labeled csv data
                        if type_number is not None:
                            part = type_number
                            logger.info("Filtering by part complexity (type number)")
                        else:
                            raise ValueError(
                                "Part needs to be specified when using type_filter win unlabeled data"
                            )
                    else:
                        # filter by part, requires labeled csv data
                        part = args.jobname
                        logger.info("Filtering by part name")
                logger.warning(
                    f"Using only type {part} for optimization from {args.output}"
                )
                x = x[types == part]
                y = y[types == part]
                logger.debug(f"x.shape: {x.shape}, y.shape: {y.shape}")

        # Train the model
        logger.info("Training Model")
        bayes_opt = BayesianOptimization(
            args,
            x,
            y,
            type_filter=type_filter,  # to indicate the name on the **labeled** CSV file
            types=types,
            parts=parts,  # if parts is not None MOE is used
        )
        # logger.debug(repr(bayes_opt))
        if args.iterations > 0:
            # Start the license server:
            licenser_server = LicenseServer(args, True)
            licenser_server.start()
            # Execute the optimization
            logger.info(
                f"Optimization with {args.iterations} with {args.parallel_samples} parallel samples"
            )
            bayes_opt.optimize(
                args.iterations, args.parallel_samples, args.user_constraints
            )
            # Stop the license server:
            licenser_server.stop()

    # plot if needed
    if args.plot and args.input is None:
        if "dummy" in socket.gethostname().lower():
            bayes_opt.plot(fix_other_dims=False)
            bayes_opt.plot(fix_other_dims=True)
            plot(args.x_fields, args.y_fields, x, y, args.attention_coefficients)
        else:
            bayes_opt.plot(fix_other_dims=False, save_path=args.result_folder)
            bayes_opt.plot(fix_other_dims=True, save_path=args.result_folder)
            plot(
                args.x_fields,
                args.y_fields,
                x,
                y,
                args.attention_coefficients,
                save_path=args.result_folder,
            )


if __name__ == "__main__":
    main()
