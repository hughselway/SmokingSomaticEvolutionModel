import argparse
import json
import math
from typing import Any, Callable

from .varying_scale_conversion import (
    convert_from_varying_scale,
    convert_to_varying_scale,
)
from .unpack_priors import get_prior_dict


class Parameter:
    def __init__(
        self,
        name: str,
        module: str,
        default_value: float,
        inactive_value: float,  # value if that module is not activated
        varying_scale: str,
        bayesopt_type: str,
        bayesopt_bounds: tuple[float, float],
        prior_dict: dict[str, Any],
        colour: tuple[float, float, float],
        help_string: str,
    ) -> None:
        self.name = name
        self.module = module
        self.default_value = default_value
        self.inactive_value = inactive_value
        self.varying_scale = varying_scale
        self.bayesopt_type = bayesopt_type
        self.bayesopt_bounds = bayesopt_bounds
        self.prior_dict = prior_dict
        self.colour = colour
        self.help_string = help_string

    def __repr__(self) -> str:
        return (
            f"Parameter(name={self.name}, default_value={self.default_value}, "
            f"inactive_value={self.inactive_value}, "
            f"bayesopt_type={self.bayesopt_type}, "
            f"bayesopt_bounds={self.bayesopt_bounds}, prior_dict={self.prior_dict}, "
            f"colour={self.colour})"
        )

    def __str__(self) -> str:
        return (
            f"{self.name} parameter; "
            f"default_value={self.default_value}, "
            f"inactive_value={self.inactive_value}, "
            f"bayesopt_type={self.bayesopt_type}, "
            f"bayesopt_bounds={self.bayesopt_bounds}, prior_dict={self.prior_dict}"
        )

    @property
    def abbreviated_name(self) -> str:
        return abbreviate_name(self.name)

    def get_bayesopt_parameter_definition(self) -> dict[str, str | float]:
        assert self.bayesopt_type is not None and self.bayesopt_bounds is not None
        bayesopt_parameter_definition: dict[str, str | float] = {
            "name": self.name,
            "type": self.bayesopt_type,
            "lb": self.bayesopt_bounds[0],
            "ub": self.bayesopt_bounds[1],
        }
        if self.bayesopt_type == "pow":
            bayesopt_parameter_definition["base"] = 10
        return bayesopt_parameter_definition

    def add_argument(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            f"-{self.abbreviated_name}",
            f"--{self.name}",
            type=float,
            help=self.help_string,
        )

    def interpolate_value(self, interpolating_coefficient: float) -> float:
        assert 0 < interpolating_coefficient < 1
        if self.bayesopt_type == "num":
            return self.bayesopt_bounds[0] + interpolating_coefficient * (
                self.bayesopt_bounds[1] - self.bayesopt_bounds[0]
            )
        assert self.bayesopt_type == "pow"
        return 10 ** (
            math.log10(self.bayesopt_bounds[0])
            + interpolating_coefficient
            * (
                math.log10(self.bayesopt_bounds[1])
                - math.log10(self.bayesopt_bounds[0])
            )
        )

    def convert_from_varying_scale(self, value: float) -> float:
        return convert_from_varying_scale(value, self.varying_scale)

    def convert_to_varying_scale(self, value: float) -> float:
        return convert_to_varying_scale(value, self.varying_scale)

    def varying_scale_conversion_gradient(self, value: float) -> float:
        """Return the gradient of the conversion from the varying scale to the
        simulation scale at the given value."""
        if self.varying_scale == "log":
            return math.exp(value)
        if self.varying_scale == "logit":
            return math.exp(-value) / ((1 + math.exp(-value)) ** 2)
        if self.varying_scale == "linear":
            return 1
        raise ValueError(f"Unknown varying scale: {self.varying_scale}")


def abbreviate_name(name: str) -> str:
    return "".join([word[0] for word in name.split("_")])


def get_json_parameter_list() -> list[dict[str, Any]]:
    with open(
        "ClonesModelling/parameters/hypothesis_module_parameters.json",
        "r",
        encoding="utf-8",
    ) as json_file:
        parameter_list: list[dict[str, Any]] = json.load(json_file)
    return parameter_list


def replace_json_parameter_list(parameter_list: list[dict[str, Any]]) -> None:
    with open(
        "ClonesModelling/parameters/hypothesis_module_parameters.json",
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump(parameter_list, json_file, indent=4)


def should_skip_parameter(
    parameter_dict: dict[str, Any],
    skipped_parameters: list[str],
    module_name: str | None = None,
    spatial: bool = True,
) -> bool:
    if module_name is not None and parameter_dict["module"] != module_name:
        return True

    name = parameter_dict["name"]

    if not spatial and name in [
        "ambient_quiescent_divisions_per_year",
        "protected_region_radius",
    ]:
        return True

    if name in skipped_parameters:
        return True
    return False


def get_parameters(
    module_name: str | None = None,
    spatial: bool = True,
    skipped_parameters: list[str] | None = None,
) -> list[Parameter]:
    parameters_json = get_json_parameter_list()
    skipped_parameters = skipped_parameters if skipped_parameters is not None else []

    parameters: list[Parameter] = []

    for parameter_dict in parameters_json:
        if not should_skip_parameter(
            parameter_dict,
            skipped_parameters=skipped_parameters,
            module_name=module_name,
            spatial=spatial,
        ):
            parameters.append(
                Parameter(
                    name=parameter_dict["name"],
                    module=parameter_dict["module"],
                    default_value=parameter_dict["default_value"],
                    inactive_value=(
                        parameter_dict["inactive_value"]
                        if "inactive_value" in parameter_dict
                        else None
                    ),
                    varying_scale=parameter_dict["varying_scale"],
                    bayesopt_type=(
                        parameter_dict["bayesopt_type"]
                        if "bayesopt_type" in parameter_dict
                        else ""
                    ),
                    bayesopt_bounds=(
                        (
                            float(parameter_dict["bayesopt_bounds"][0]),
                            float(parameter_dict["bayesopt_bounds"][1]),
                        )
                        if "bayesopt_bounds" in parameter_dict
                        else (0, 0)
                    ),
                    prior_dict=get_prior_dict(parameter_dict),
                    colour=(
                        float(parameter_dict["colour"][0]),
                        float(parameter_dict["colour"][1]),
                        float(parameter_dict["colour"][2]),
                    ),
                    help_string=parameter_dict["help_string"],
                )
            )
    return parameters


def get_parameter_names(
    module_name: str | None = None,
    spatial: bool = True,
    skipped_parameters: list[str] | None = None,
) -> list[str]:
    return [
        parameter.name
        for parameter in get_parameters(module_name, spatial, skipped_parameters)
    ]


def get_conversion_from_varying_scale(
    varying_parameters: list[Parameter],
) -> Callable[[dict[str, float]], dict[str, float]]:
    # return a function which takes in a dictionary of parameter values in their ABC
    # varying scale, and returns a dictionary of parameter values in their simulation
    # scale
    return lambda parameter_values: {
        parameter.name: parameter.convert_from_varying_scale(
            parameter_values[parameter.name]
        )
        for parameter in varying_parameters
        if parameter.name in parameter_values
    }


def get_conversion_to_varying_scale(
    varying_parameters: list[Parameter],
) -> Callable[[dict[str, float]], dict[str, float]]:
    # return a function which takes in a dictionary of parameter values in their
    # simulation scale, and returns a dictionary of parameter values in their ABC
    # varying scale
    return lambda parameter_values: {
        parameter.name: parameter.convert_to_varying_scale(
            parameter_values[parameter.name]
        )
        for parameter in varying_parameters
        if parameter.name in parameter_values
    }


def get_skipped_parameter_values(parsed_args: argparse.Namespace) -> dict[str, float]:
    all_parameters = {parameter.name: parameter for parameter in get_parameters()}
    skipped_parameter_values: dict[str, float] = {
        parameter_name: getattr(parsed_args, parameter_name)
        for parameter_name in all_parameters.keys()
        if getattr(parsed_args, parameter_name) is not None
    }
    for parameter_name, parameter in all_parameters.items():
        if (
            not parsed_args.vary_division_rate
            and parameter_name
            in [
                "non_smoking_divisions_per_year",
                "smoking_division_rate_augmentation",
            ]
        ) or (
            not parsed_args.vary_mutation_rate
            and parameter_name == "mutation_rate_multiplier_shape"
        ):
            skipped_parameter_values[parameter_name] = parameter.inactive_value
    return skipped_parameter_values
