"""
Priors are specified for non-discrete parameters by the mean and standard deviation, on
the simulation scale. This is translated into a normal distribution in the varying scale
whose image in the simulation scale has the required mean and std.
"""

import math
from typing import Any
import numpy as np
import scipy.special  # type: ignore
from scipy.optimize import fsolve  # type: ignore

from .varying_scale_conversion import convert_to_varying_scale


def get_prior_dict(parameter_dict: dict[str, Any]) -> dict[str, Any]:
    if "prior_dict" in parameter_dict:
        assert parameter_dict["varying_scale"] == "linear"
        assert isinstance(parameter_dict["prior_dict"], dict)
        return parameter_dict["prior_dict"]
    if parameter_dict["varying_scale"] == "linear":
        return {
            "type": "norm",
            "kwargs": {
                "loc": parameter_dict["prior_mean"],
                "scale": parameter_dict["prior_std"],
            },
        }
    if parameter_dict["varying_scale"] == "log":
        # in this case we can directly solve
        mu = parameter_dict["prior_mean"]
        sigma = parameter_dict["prior_std"]
        varying_scale_prior_mean = math.log(mu) - 0.5 * math.log(1 + sigma**2 / mu**2)
        varying_scale_prior_std = math.sqrt(math.log(1 + sigma**2 / mu**2))
        return {
            "type": "norm",
            "kwargs": {
                "loc": varying_scale_prior_mean,
                "scale": varying_scale_prior_std,
            },
        }
    if parameter_dict["varying_scale"] == "logit":
        # here there is no analytic solution, so we use numerical methods
        mu = parameter_dict["prior_mean"]
        sigma = parameter_dict["prior_std"]
        (
            varying_scale_prior_mean,
            varying_scale_prior_std,
        ) = find_logit_normal_parameters(mu, sigma)
        return {
            "type": "norm",
            "kwargs": {
                "loc": varying_scale_prior_mean,
                "scale": varying_scale_prior_std,
            },
        }
    raise ValueError(f"Unknown varying scale: {parameter_dict['varying_scale']}")


def find_logit_normal_parameters(mu: float, sigma: float) -> tuple[float, float]:
    """
    Return the values mu_star, sigma_star such that if Y ~ N(mu_star, sigma_star)
    then expit(Y) ~ N(mu, sigma).
    """

    def error(mu_star: float, sigma_star: float) -> tuple[float, float]:
        expit_mean, expit_std, _ = get_expit_mean_std(mu_star, sigma_star)
        return expit_mean - mu, expit_std - sigma

    sigma_guess = (
        convert_to_varying_scale(mu + sigma, "logit")
        - convert_to_varying_scale(mu - sigma, "logit")
    ) / 2
    mu_star, sigma_star = fsolve(
        lambda x: error(x[0], x[1]),
        [
            1 / (1 + math.exp(-mu)),
            sigma_guess,
        ],
        xtol=1e-4,
    )
    return mu_star, abs(sigma_star)


def get_expit_mean_std(mu_star: float, sigma_star: float) -> tuple[float, float, int]:
    """
    Return the mean and standard deviation of the expit(Y) where
    Y ~ N(mu_star, sigma_star).
    Do this using the Monte Carlo approximation
    $E[expit(Y)^n] \approx \frac{1}{K-1} \sum_{i=1}^{K-1} expit(\Phi_{mu_star, sigma_star}^{-1}(i/K))^n$
    Iterate over K until convergence.
    """
    sigma_star = abs(sigma_star)
    tolerance = 1e-4
    K = 2
    expit_mean = 0.0
    expit_std = 0.0
    while True:
        prev_expit_mean = expit_mean
        prev_expit_std = expit_std
        inverse_cdf_values = mu_star + sigma_star * scipy.special.ndtri(
            [i / K for i in range(1, K)]
        )
        expit_inverse_cdf_values = 1 / (1 + np.exp(-inverse_cdf_values))
        expit_mean = np.mean(expit_inverse_cdf_values)
        expit_second_moment = np.mean(expit_inverse_cdf_values**2)
        expit_std = math.sqrt(expit_second_moment - expit_mean**2)
        if (
            abs(expit_mean - prev_expit_mean) < tolerance
            and abs(expit_std - prev_expit_std) < tolerance
        ):
            break
        K *= 2
        if K > 2**64 - 1:
            raise ValueError(
                f"Too many iterations in prior distribution calculations:"
                f"K={K}, current error ({abs(expit_mean - prev_expit_mean)},"
                f"{abs(expit_std - prev_expit_std)})"
            )
    return expit_mean, expit_std, K
