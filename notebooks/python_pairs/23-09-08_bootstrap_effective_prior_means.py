# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: clonesmodelling-US-Ycvdi-py3.10
#     language: python
#     name: python3
# ---

# %%
## Move to the correct directory
import os
os.chdir("/Users/hughselway/Documents/ClonesModelling/")

# %%
from ClonesModelling.parameters.parameter_class import get_parameters, Parameter
import numpy as np


def bootstrap_prior(
    parameter: Parameter, n_samples: int
) -> tuple[float, float, float, float]:
    """Bootstrap the prior distribution for a parameter.

    Args:
        parameter: The parameter to bootstrap.
        n_samples: The number of samples to take.

    Returns:
        The mean, lower and upper bounds, and standard deviation of the prior distribution.
    """
    if parameter.prior_dict["type"] == "rv_discrete":
        raise NotImplementedError
    assert parameter.prior_dict["type"] == "norm"

    varying_scale_prior_samples = np.random.normal(
        loc=parameter.prior_dict["kwargs"]["loc"],
        scale=parameter.prior_dict["kwargs"]["scale"],
        size=n_samples,
    )
    samples = [
        parameter.convert_from_varying_scale(sample)
        for sample in varying_scale_prior_samples
    ]

    mean = np.mean(samples)
    lower = np.percentile(samples, 2.5)
    upper = np.percentile(samples, 97.5)
    std = np.std(samples)
    return mean, lower, upper, std


# %%
# get the original prior means and standard deviations as specified in the json
import json

json_path = "ClonesModelling/parameters/hypothesis_module_parameters.json"
parameters_json = json.load(open(json_path, "r"))
parameter_specs = {
    # json_dict["name"]: (json_dict['prior_mean'], json_dict['prior_std'])
    json_dict["name"]: (
        f"{json_dict['prior_mean']} ± {json_dict['prior_std']} "
        f"({json_dict['prior_mean'] - 1.96 * json_dict['prior_std']}, "
        f"{json_dict['prior_mean'] + 1.96 * json_dict['prior_std']})"
    )
    for json_dict in parameters_json
    if "prior_mean" in json_dict.keys()
}
parameter_specs

# %%
from math import floor, log10


def round_to_sig_digits(value: float, significant_digits: int) -> float:
    return (
        value
        if value == 0
        else round(value, -int(floor(log10(abs(value)))) + significant_digits-1)
    )


# %%
from datetime import datetime
import time

start_time = time.perf_counter()
parameters = get_parameters()
print('get parameters time: ', time.perf_counter() - start_time)

for parameter in parameters:
    if parameter.prior_dict["type"] == "norm":
        # print(parameter.name, " ", parameter.prior_dict['kwargs'])

        mean, lower, upper, std = bootstrap_prior(parameter, 100000)
        mean = round_to_sig_digits(mean, 3)
        lower = round_to_sig_digits(lower, 3)
        upper = round_to_sig_digits(upper, 3)
        std = round_to_sig_digits(std, 3)
        print(
            f"{parameter.name}: ({parameter.varying_scale})"
            f"\n\tintended:\t{parameter_specs[parameter.name]}"
            f"\n\teffective:\t{mean} ± {std} ({lower}, {upper})"
        )

# %% [markdown]
# Fixed! (With the previous regime, the intended and effective moments were quite different).
