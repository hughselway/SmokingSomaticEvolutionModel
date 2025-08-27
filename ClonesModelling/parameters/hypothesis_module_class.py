import argparse

from .parameter_class import (
    Parameter,
    abbreviate_name,
    get_parameters,
)


class HypothesisModule:
    def __init__(
        self,
        name: str = "base",
        spatial: bool = False,
        skipped_parameters: list[str] | None = None,
    ):
        self.name = name
        self.parameters: list[Parameter] = get_parameters(
            module_name=self.name,
            spatial=spatial,
            skipped_parameters=skipped_parameters,
        )
        self.skipped_parameters = skipped_parameters
        self.dependencies: list[str] | None = (
            None if self.name != "quiescent_protected" else ["quiescent"]
        )

    def __repr__(self) -> str:
        return f"HypothesisModule(name={self.name})"

    def __str__(self) -> str:
        return f"{self.name} hypothesis module with parameters:\n" + "\n".join(
            [str(parameter) for parameter in self.parameters]
        )

    @property
    def abbreviated_name(self) -> str:
        return abbreviate_name(self.name)

    @property
    def parameter_names(self) -> list[str]:
        return [parameter.name for parameter in self.parameters]

    @property
    def abbreviated_parameter_names(self) -> list[str]:
        return [abbreviate_name(parameter.name) for parameter in self.parameters]

    def get_parameter(self, parameter_name) -> Parameter:
        return next(
            parameter
            for parameter in self.parameters
            if parameter.name == parameter_name
            or abbreviate_name(parameter.name) == parameter_name
        )

    def get_parameter_names(self, abbreviated: bool = False) -> list[str]:
        return [
            parameter.abbreviated_name if abbreviated else parameter.name
            for parameter in self.parameters
        ]

    def add_parameter_arguments(self, parser: argparse.ArgumentParser) -> None:
        for parameter in self.parameters:
            parameter.add_argument(parser)

    def get_module_colour(self) -> tuple[float, float, float]:
        # mean of the colours of the parameters
        mean_colour = (
            sum(colour) / len(colour)
            for colour in zip(
                *[
                    parameter.colour
                    for parameter in self.parameters
                    if parameter.name != "fitness_change_scale"
                ]
            )
        )
        return (next(mean_colour), next(mean_colour), next(mean_colour))


def get_module_names(
    parsed_args: argparse.Namespace | None = None,
    abbreviated: bool = False,
    include_base: bool = False,
) -> list[str]:
    module_names = set(parameter.module for parameter in get_parameters())
    return [
        (abbreviate_name(module_name) if abbreviated else module_name)
        for module_name in module_names
        if (module_name == "base" and include_base)
        or (
            module_name != "base"
            and (parsed_args is None or getattr(parsed_args, module_name + "_module"))
        )
    ]


def get_hypothesis_modules(
    parsed_args: argparse.Namespace | None = None,
    include_base: bool = True,
    spatial: bool = False,
    skipped_parameters: list[str] | None = None,
) -> list[HypothesisModule]:
    return [
        HypothesisModule(
            module_name,
            spatial,
            skipped_parameters,
        )
        for module_name in get_module_names(parsed_args, include_base=include_base)
    ]


def get_hypothesis_module_colours() -> dict[str, tuple[float, float, float]]:
    return {
        hypothesis_module.name: hypothesis_module.get_module_colour()
        for hypothesis_module in get_hypothesis_modules()
    }
