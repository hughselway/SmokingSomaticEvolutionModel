import argparse
import itertools

from pandas import DataFrame  # type: ignore

from .hypothesis_module_class import (
    HypothesisModule,
    get_hypothesis_modules,
    get_module_names,
)
from .parameter_class import Parameter, abbreviate_name, get_parameters

MODULE_ORDERING = [
    "base",
    "quiescent",
    "q",
    "quiescent_protected",
    "qp",
    "protected",
    "p",
    "immune_response",
    "ir",
    "smoking_driver",
    "sd",
]


class HypotheticalParadigm:
    def __init__(
        self,
        module_names: list[str] | None = None,
        spatial: bool = False,
        skipped_parameters: list[str] | None = None,
        hypothesis_modules: list[HypothesisModule] | None = None,
    ) -> None:
        assert hypothesis_modules is None or module_names is None
        self.spatial = spatial
        self.skipped_parameters = (
            skipped_parameters if skipped_parameters is not None else []
        )
        self.hypothesis_modules = sorted(
            (
                [
                    HypothesisModule(
                        module_name,
                        spatial,
                        skipped_parameters,
                    )
                    for module_name in (
                        ["base"]
                        + (
                            []
                            if module_names is None
                            else [x for x in module_names if x != "base"]
                        )
                    )
                ]
                if hypothesis_modules is None
                else hypothesis_modules
            ),
            key=(lambda x: MODULE_ORDERING.index(x.name)),
        )
        self.check_module_dependencies()

    def __repr__(self) -> str:
        return (
            f"HypotheticalParadigm(module_names="
            f"{self.get_module_names(include_base=False)}, "
            f"spatial={self.spatial}, "
            f"skipped_parameters={self.skipped_parameters},)"
        )

    def __str__(self) -> str:
        return (
            f"Hypothetical Paradigm: modules "
            f"{self.get_modules_string(abbreviated=True)}"
            f"{' (spatial)' if self.spatial else ''}"
            f"{' (skipped parameters: ' + ', '.join(self.skipped_parameters) + ')'}"
        )

    def get_parameter(self, parameter_name: str) -> Parameter:
        for module in self.hypothesis_modules:
            if (
                parameter_name in module.parameter_names
                or parameter_name in module.abbreviated_parameter_names
            ):
                return module.get_parameter(parameter_name)
        raise ValueError(f"Parameter {parameter_name} not found.")

    def includes_module(self, module_name: str) -> bool:
        return any(
            module_name in self.get_module_names(abbreviated, include_base=True)
            for abbreviated in [True, False]
        )

    def get_module_names(
        self, abbreviated: bool = False, include_base: bool = False
    ) -> list[str]:
        return [
            (module.abbreviated_name if abbreviated else module.name)
            for module in self.hypothesis_modules
            if include_base or module.name != "base"
        ]

    def get_modules_string(
        self, abbreviated: bool = False, include_base: bool = False
    ) -> str:
        return (
            "-".join(self.get_module_names(abbreviated, include_base))
            if include_base or len(self.hypothesis_modules) > 1
            else "base"
        )

    def get_module_parameter_names(
        self, abbreviated: bool = False
    ) -> dict[str, list[str]]:
        return {
            (
                abbreviate_name(module.name) if abbreviated else module.name
            ): module.get_parameter_names(abbreviated)
            for module in self.hypothesis_modules
        }

    def get_parameter_names(self, abbreviated: bool = False) -> list[str]:
        return list(
            itertools.chain.from_iterable(
                module.get_parameter_names(abbreviated)
                for module in self.hypothesis_modules
            )
        )

    @property
    def varying_parameters(self) -> list[Parameter]:
        return [
            parameter
            for module in self.hypothesis_modules
            for parameter in module.parameters
        ]

    @property
    def fixed_parameters(self) -> dict[str, float]:
        fixed_parameters = {
            parameter.name: parameter.inactive_value
            for parameter in get_parameters()
            if parameter.name not in self.get_parameter_names()
        }
        return fixed_parameters

    def check_module_dependencies(self) -> None:
        for module in self.hypothesis_modules:
            if module.dependencies is None:
                continue
            for dependency in module.dependencies:
                if not self.includes_module(dependency):
                    raise ValueError(
                        f"Module {module.name} requires module {dependency}."
                    )

    def modules_present(
        self, abbreviated: bool = False, include_base: bool = False
    ) -> list[tuple[str, bool]]:
        return [
            (module, module in self.get_module_names(include_base=include_base))
            for module in get_module_names(
                abbreviated=abbreviated, include_base=include_base
            )
        ]

    def convert_dataframe_from_varying_scale(
        self, varying_scale_df: DataFrame
    ) -> DataFrame:
        assert set(varying_scale_df.columns) == set(self.get_parameter_names()), (
            f"Parameter names in dataframe {set(varying_scale_df.columns)} do not "
            f"match parameter names in paradigm {set(self.get_parameter_names())}."
        )
        return varying_scale_df.apply(
            lambda column: [
                self.get_parameter(column.name).convert_from_varying_scale(value)
                for value in column
            ]
        )

    def convert_dataframe_to_varying_scale(
        self, simulation_scale_df: DataFrame
    ) -> DataFrame:
        assert set(simulation_scale_df.columns) == set(self.get_parameter_names()), (
            f"Parameter names in dataframe {set(simulation_scale_df.columns)} do not "
            f"match parameter names in paradigm {set(self.get_parameter_names())}."
        )
        return simulation_scale_df.apply(
            lambda column: [
                self.get_parameter(column.name).convert_to_varying_scale(value)
                for value in column
            ]
        )

    def convert_dict_from_varying_scale(
        self, varying_scale_dict: dict[str, float]
    ) -> dict[str, float]:
        assert set(varying_scale_dict.keys()) == set(self.get_parameter_names())
        return {
            parameter_name: self.get_parameter(
                parameter_name
            ).convert_from_varying_scale(value)
            for parameter_name, value in varying_scale_dict.items()
        }


def is_protection_selection_subset(module_names: list[str]) -> bool:
    if ("quiescent_protected" in module_names) or ("qp" in module_names):
        return False
    protection_module_count = sum(
        module_name in module_names for module_name in ["quiescent", "protected"]
    ) or sum(module_name in module_names for module_name in ["q", "p"])
    selection_module_count = sum(
        module_name in module_names
        for module_name in ["immune_response", "smoking_driver"]
    ) or sum(module_name in module_names for module_name in ["ir", "sd"])
    assert len(module_names) == protection_module_count + selection_module_count + 1
    return (protection_module_count <= 1) and (selection_module_count <= 1)


def get_hypothetical_paradigm_for_each_subset(
    parsed_args: argparse.Namespace | None = None,
    hypothesis_module_names: list[str] | None = None,
    max_modules: int | None = None,
    spatial: bool = False,
    skipped_parameters: list[str] | None = None,
) -> list[HypotheticalParadigm]:
    if parsed_args is not None and hypothesis_module_names is not None:
        raise ValueError("Cannot specify both parsed_args and hypothesis_module_names.")
    if parsed_args is None and hypothesis_module_names is None:
        raise ValueError("Must specify either parsed_args or hypothesis_module_names.")
    hypothesis_modules: dict[str, HypothesisModule] = (
        {
            module.name: module
            for module in get_hypothesis_modules(
                parsed_args,
                include_base=True,
                spatial=spatial,
                skipped_parameters=skipped_parameters,
            )
        }
        if hypothesis_module_names is None
        else {
            module_name: HypothesisModule(module_name, spatial, skipped_parameters)
            for module_name in set().union(hypothesis_module_names, ["base"])
        }
    )
    return sorted(
        [
            HypotheticalParadigm(
                spatial=spatial,
                skipped_parameters=skipped_parameters,
                hypothesis_modules=[
                    hypothesis_modules[module_name] for module_name in combination
                ],
            )
            for combination in filter(
                lambda modules_list: (  # type: ignore
                    ("q" in modules_list or "qp" not in modules_list)
                    and (
                        "quiescent" in modules_list
                        or "quiescent_protected" not in modules_list
                    )
                    and ("base" in modules_list)
                    and (
                        max_modules is None
                        or (len(modules_list) <= max_modules + 1)
                        or (
                            max_modules == -1
                            and is_protection_selection_subset(list(modules_list))
                        )
                    )
                ),
                itertools.chain.from_iterable(
                    itertools.combinations(
                        hypothesis_modules.keys(), combination_length
                    )
                    for combination_length in range(len(hypothesis_modules.keys()) + 1)
                ),
            )
        ],
        key=lambda x: (
            len(x.get_module_names()),
            *[MODULE_ORDERING.index(module) for module in x.get_module_names()],
        ),
    )


def get_hypothetical_paradigm(
    parsed_args: argparse.Namespace | None = None,
    hypothesis_module_names: list[str] | None = None,
    spatial: bool = False,
    skipped_parameters: list[str] | None = None,
) -> HypotheticalParadigm:
    if parsed_args is not None and hypothesis_module_names is not None:
        raise ValueError("Cannot specify both parsed_args and hypothesis_module_names.")
    hypothesis_module_names = (
        get_module_names(parsed_args)
        if parsed_args is not None
        else hypothesis_module_names
    )
    return HypotheticalParadigm(
        hypothesis_module_names,
        spatial,
        skipped_parameters,
    )


def get_abc_hypothetical_paradigms(
    parsed_args: argparse.Namespace, skipped_parameter_values: dict[str, float]
) -> list[HypotheticalParadigm]:
    hypothetical_paradigms = (
        get_hypothetical_paradigm_for_each_subset(
            parsed_args,
            max_modules=parsed_args.max_modules,
            spatial=parsed_args.spatial,
            skipped_parameters=list(skipped_parameter_values.keys()),
        )
        if not parsed_args.fix_paradigm
        else [
            get_hypothetical_paradigm(
                parsed_args,
                spatial=parsed_args.spatial,
                skipped_parameters=list(skipped_parameter_values.keys()),
            )
        ]
    )

    return hypothetical_paradigms


def get_matching_hypothetical_paradigm(
    paradigm_name: str,
    hypothetical_paradigms: list[HypotheticalParadigm],
) -> HypotheticalParadigm:
    same_modules_hypothetical_paradigms = [
        hypothetical_paradigm
        for hypothetical_paradigm in hypothetical_paradigms
        if set(hypothetical_paradigm.get_modules_string().split("-"))
        == set(paradigm_name.split("-"))
    ]
    assert len(same_modules_hypothetical_paradigms) == 1
    return same_modules_hypothetical_paradigms[0]
