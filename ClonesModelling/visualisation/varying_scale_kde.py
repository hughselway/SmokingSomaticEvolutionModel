import matplotlib.pyplot as plt
import numpy as np
from scipy import stats  # type: ignore

from ..parameters.parameter_class import Parameter


def plot_kde_in_varying_scale(
    values: list[float],
    parameter: Parameter,
    axis: plt.Axes,
    **kwargs,
) -> None:
    """
    Plot a kernel density estimate of the simulation-scale values in the simulation
    scale, where the KDE is calculated in the varying scale
    """
    varying_scale_values = [
        parameter.convert_to_varying_scale(value) for value in values
    ]
    kde = stats.gaussian_kde(varying_scale_values)

    width = max(varying_scale_values) - min(varying_scale_values)
    varying_scale_x = np.linspace(
        min(varying_scale_values) - 0.25 * width,
        max(varying_scale_values) + 0.25 * width,
        1000,
    )
    axis.plot(
        [
            parameter.convert_from_varying_scale(varying_scale_x_i)
            for varying_scale_x_i in varying_scale_x
        ],
        kde(varying_scale_x),
        **kwargs,
    )
