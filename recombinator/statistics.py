import typing as tp

import numpy as np

from recombinator.types import ArrayFloat, ArrayInt


def estimate_bias_from_bootstrap(
    bootstrap_estimates: tp.Union[ArrayInt, ArrayFloat], original_estimate: float
) -> float:
    """
    This function estimates the bias of an estimator given an estimate of a
    statistic from the original data as well as variety of estimates of the same
    statistic from resampled data.

    Args:
        bootstrap_estimates:
            a NumPy array of dimension (number_of_replications, ) containing
            the statistic computed from resampled data
        original_estimate: the statistic computed from the original data
    """

    bootstrap_estimate_of_bias = np.mean(bootstrap_estimates) - original_estimate
    result: float = np.asscalar(bootstrap_estimate_of_bias)
    return result


def estimate_standard_error_from_bootstrap(
    bootstrap_estimates: tp.Union[ArrayInt, ArrayFloat],
    original_estimate: float,
    ddof: int = 0,
) -> float:
    """
    This function estimates the standard error of an estimator given an estimate
    of a statistic from the original data as well as variety of estimates of the
    same statistic from resampled data.

    Args:
        bootstrap_estimates:
            a NumPy array of dimension (number_of_replications, ) containing
            the statistic computed from resampled data
        original_estimate: the statistic computed from the original data
        ddof: Delta degrees of freedom
    """

    B = len(bootstrap_estimates)
    bootstrap_estimate_of_variance = np.sum(
        (bootstrap_estimates - original_estimate) ** 2
    ) / (B - ddof)
    bootstrap_estimate_of_standard_error = np.sqrt(bootstrap_estimate_of_variance)
    result: float = bootstrap_estimate_of_standard_error
    return result


def estimate_confidence_interval_from_bootstrap(
    bootstrap_estimates: tp.Union[ArrayInt, ArrayFloat], confidence_level: float = 95.0
) -> tp.Tuple[float, float]:
    """
    This function estimates a confidence interval of an estimator given a
    variety of estimates of the same statistic from resampled data.

    Args:
        bootstrap_estimates: a NumPy array of dimension (B, ) containing the statistic
                    computed from resampled data
        confidence_level: the confidence level associated with the confidence
                          interval in percent (i.e. between 0 and 100)
    """

    percent = 100.0 - confidence_level
    bootstrap_confidence_interval = (
        np.percentile(bootstrap_estimates, percent / 2.0),
        np.percentile(bootstrap_estimates, 100.0 - percent / 2.0),
    )

    return bootstrap_confidence_interval
