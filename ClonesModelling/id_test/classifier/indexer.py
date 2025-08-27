from dataclasses import dataclass
from typing import Any
import numpy as np

from ..calculate_distances import DISTANCE_FUNCTIONS

from ...parameters.hypothetical_paradigm_class import (
    HypotheticalParadigm,
    get_hypothetical_paradigm_for_each_subset,
)
from ...parameters.hypothesis_module_class import get_hypothesis_modules


@dataclass
class Indexer:
    distance_function_names: list[str]
    hypothetical_paradigms: list[HypotheticalParadigm]
    simulation_replicates_per_paradigm: int
    include_true_data: bool

    def __init__(
        self,
        include_2d_wasserstein: bool,
        max_modules: int,
        simulation_replicates_per_paradigm: int,
        include_true_data: bool,
        distance_function_names: list[str] | None = None,
    ):
        self.distance_function_names = (
            [
                dist_fn_name
                for dist_fn_name in DISTANCE_FUNCTIONS
                if (include_2d_wasserstein or dist_fn_name != "2D_wasserstein")
                and dist_fn_name != "zero_control"
            ]
            if distance_function_names is None
            else distance_function_names
        )
        self.hypothetical_paradigms = get_hypothetical_paradigm_for_each_subset(
            hypothesis_module_names=[
                module.name for module in get_hypothesis_modules()
            ],
            max_modules=max_modules,
            spatial=True,
            skipped_parameters=["mutation_rate_multiplier_shape"],
        )
        self.simulation_replicates_per_paradigm = simulation_replicates_per_paradigm
        self.include_true_data = include_true_data

    def __eq__(self, other) -> bool:
        assert isinstance(other, Indexer), f"{type(other)}\n{other}"
        return (
            self._same_distance_function_names(other)
            and self.same_hypothetical_paradigms(other)
            and (
                self.simulation_replicates_per_paradigm
                == other.simulation_replicates_per_paradigm
            )
            and self.include_true_data == other.include_true_data
        )

    def _same_distance_function_names(self, other: "Indexer") -> bool:
        return sorted(self.distance_function_names) == sorted(
            other.distance_function_names
        )

    def same_hypothetical_paradigms(self, other: "Indexer") -> bool:
        return [hp.get_modules_string() for hp in self.hypothetical_paradigms] == [
            hp.get_modules_string() for hp in other.hypothetical_paradigms
        ]

    def subset_hypothetical_paradigms(self, other: "Indexer") -> bool:
        return set(
            hp.get_modules_string() for hp in self.hypothetical_paradigms
        ).issubset(set(hp.get_modules_string() for hp in other.hypothetical_paradigms))

    def is_df_subset(self, other: "Indexer") -> bool:
        return (
            (
                self.simulation_replicates_per_paradigm
                == other.simulation_replicates_per_paradigm
            )
            and self.subset_hypothetical_paradigms(other)
            and set(self.distance_function_names).issubset(
                set(other.distance_function_names)
            )
        )

    @property
    def n_distance_functions(self) -> int:
        return len(self.distance_function_names)

    @property
    def n_paradigms(self) -> int:
        return len(self.hypothetical_paradigms)

    @property
    def n_classes(self) -> int:
        return len(self.hypothetical_paradigms) + int(self.include_true_data)

    @property
    def n_datapoints_excluding_true_data(self) -> int:
        return self.n_paradigms * self.simulation_replicates_per_paradigm

    @property
    def n_datapoints(self) -> int:
        if self.include_true_data:
            return self.n_datapoints_excluding_true_data + 1
        return self.n_datapoints_excluding_true_data

    def get_module_names(self, abbreviated: bool = True) -> list[str]:
        return list(
            set().union(
                *[
                    paradigm.get_module_names(abbreviated)
                    for paradigm in self.hypothetical_paradigms
                ]
            )
        )

    def get_paradigm_agnostic_index(
        self, paradigm_index: int, replicate_index: int
    ) -> int:
        assert 0 <= paradigm_index < self.n_paradigms, paradigm_index
        assert (
            0 <= replicate_index < self.simulation_replicates_per_paradigm
        ), replicate_index
        return (
            paradigm_index * self.simulation_replicates_per_paradigm + replicate_index
        )

    @property
    def paradigm_index_labels(self) -> np.ndarray:
        """
        Paradigm indices by datapoint in the paradigm-agnostic distance matrix format
        """
        return np.repeat(
            np.arange(self.n_paradigms), self.simulation_replicates_per_paradigm
        )

    @property
    def replicate_indices(self) -> np.ndarray:
        """
        Replicate index by datapoint in the paradigm-agnostic distance matrix format
        """
        return np.pad(
            np.tile(
                np.arange(self.simulation_replicates_per_paradigm), self.n_paradigms
            ),
            (0, 1 if self.include_true_data else 0),
            constant_values=-1,
        )

    def get_paradigm_names(self, abbreviated=True) -> list[str]:
        return [
            paradigm.get_modules_string(abbreviated=abbreviated)
            for paradigm in self.hypothetical_paradigms
        ]

    def get_distance_matrix_shape(self) -> tuple[int, int, int, int, int]:
        return (
            self.n_distance_functions,
            self.n_paradigms,
            self.n_paradigms,
            self.simulation_replicates_per_paradigm,
            self.simulation_replicates_per_paradigm,
        )

    def get_true_data_distance_shape(self) -> tuple[int, int, int]:
        return (
            self.n_distance_functions,
            self.n_paradigms,
            self.simulation_replicates_per_paradigm,
        )

    @property
    def max_modules(self) -> int:
        if len(self.hypothetical_paradigms) == 9:
            # hacky hacky hard-coding - this is the protection-selection case
            return -1
        return max(len(hp.hypothesis_modules) for hp in self.hypothetical_paradigms) - 1

    def json_dict(self) -> dict[str, Any]:
        return {
            "include_2d_wasserstein": "2D_wasserstein" in self.distance_function_names,
            "max_modules": self.max_modules,
            "simulation_replicates_per_paradigm": (
                self.simulation_replicates_per_paradigm
            ),
            "include_true_data": self.include_true_data,
            "distance_function_names": self.distance_function_names,
        }
