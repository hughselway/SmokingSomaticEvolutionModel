import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from ..parameters.parameter_class import Parameter, get_parameters


def add_prior_distribution(
    axis: plt.Axes,
    parameter: Parameter,
    **kwargs,
) -> None:
    if parameter.prior_dict["type"] == "rv_discrete":
        # add bar plot
        x = parameter.prior_dict["kwargs"]["values"][0]
        y = parameter.prior_dict["kwargs"]["values"][1]
        assert len(x) == len(y)
        assert sum(y) == 1
        axis.bar(x, y, **kwargs)
        return

    assert (
        parameter.prior_dict["type"] == "norm"
    ), f"unexpected prior type {parameter.prior_dict['type']}"

    distribution = stats.norm(
        loc=parameter.prior_dict["kwargs"]["loc"],
        scale=parameter.prior_dict["kwargs"]["scale"],
    )
    varying_scale_x = np.linspace(
        distribution.ppf(0.01),
        distribution.ppf(0.99),
        100,
    )
    x = [
        parameter.convert_from_varying_scale(varying_scale_x_i)
        for varying_scale_x_i in varying_scale_x
    ]
    y = distribution.pdf(varying_scale_x)
    axis.plot(x, y, **kwargs)


def plot_parameter_priors(
    plot_dir: str = "logs/parameter_priors",
    plot_grid: bool = True,
    **kwargs,
) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    parameters = get_parameters()
    for parameter in parameters:
        fig, axis = plt.subplots()
        add_prior_distribution(axis, parameter, **kwargs)
        axis.set_xlabel(parameter.name)
        axis.set_yticks([])
        axis.set_ylabel("")
        for side in ["top", "right", "left"]:
            axis.spines[side].set_visible(False)
        fig.savefig(f"{plot_dir}/{parameter.name}.png")
        plt.close(fig)
    if not plot_grid:
        return
    ncols = 4
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
    for i, parameter in enumerate(parameters):
        add_prior_distribution(axis.flat[i], parameter, **kwargs)
        axis.flat[i].set_xlabel(parameter.name)
        axis.flat[i].set_yticks([])
        axis.flat[i].set_ylabel("")
    fig.savefig(f"{plot_dir}/grid.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_parameter_priors()
