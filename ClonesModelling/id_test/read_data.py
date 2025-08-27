from dataclasses import dataclass
from enum import Enum
import os
import random
from typing import Generator
import numpy as np
import pandas as pd  # type: ignore

from ..data.mixture_model_data import MixtureModelData, read_mixture_model_from_string
from ..parameters.parameter_class import abbreviate_name
from ..parameters.hypothetical_paradigm_class import MODULE_ORDERING
from ..distance import fit_mixture_model


class SimilarityLevel(Enum):
    REPLICATE = 1
    INTRA_PARADIGM = 2
    INTER_PARADIGM = 3


@dataclass
class PatientReplicateData:
    """
    Data for a single patient replicate; fails if branch lengths or J1 requested but
    not provided
    """

    mutational_burden: np.ndarray
    _branch_lengths: np.ndarray | None
    _j_one: float | None
    _mixture_model_data: MixtureModelData | None

    @property
    def branch_lengths(self) -> np.ndarray:
        if self._branch_lengths is None:
            raise ValueError("Branch lengths not provided")
        return self._branch_lengths

    @property
    def j_one(self) -> float:
        if self._j_one is None:
            raise ValueError("J1 not provided")
        return self._j_one

    @property
    def mixture_model_data(self) -> MixtureModelData:
        if self._mixture_model_data is None:
            raise ValueError("Mixture model not provided")
        return self._mixture_model_data


@dataclass
class ReplicateData:
    patient_mutational_burden: dict[str, np.ndarray]
    patient_branch_lengths: dict[str, np.ndarray | None]
    patient_j_one: dict[str, float | None]
    patient_mixture_models: dict[str, MixtureModelData | None]

    def __getitem__(self, patient: str) -> PatientReplicateData:
        return PatientReplicateData(
            self.patient_mutational_burden[patient],
            self.patient_branch_lengths[patient],
            self.patient_j_one[patient],
            self.patient_mixture_models[patient],
        )


def attach_error_info(func):
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except (FileNotFoundError, ValueError, AssertionError, IndexError) as e:
            error_message = str(self)
            if args:
                error_message += f"\nargs: {args}"
            if kwargs:
                error_message += f"\nkwargs: {kwargs}"
            raise type(e)(error_message) from e

    return wrapper


class LazyReplicateList:
    data_path: str

    def __init__(self, data_path: str):
        self.data_path = data_path

    def __str__(self) -> str:
        return f"Lazy replicate list from {self.data_path}"

    @attach_error_info
    def __getitem__(self, key: int) -> ReplicateData:
        mixture_models_added = False
        with open(self.data_path, "r", encoding="utf-8") as simulation_file:
            patient_mutational_burden: dict[str, np.ndarray] = {}
            patient_branch_lengths: dict[str, np.ndarray | None] = {}
            patient_j_one: dict[str, float | None] = {}
            patient_mixture_models: dict[str, MixtureModelData | None] = {}
            for line in simulation_file:
                if line.strip() == "" or "Error" in line:
                    continue
                entries = line.strip().split("--")
                if entries[0] != str(key + 1):
                    continue
                patient = entries[1]
                assert (
                    patient not in patient_mutational_burden
                ), f"Patient {patient} appears twice"
                patient_mutational_burden[patient] = np.array(
                    [
                        list(map(int, entries[2].split())),
                        list(map(int, entries[3].split())),
                    ]
                ).T
                assert len(patient_mutational_burden[patient].shape) == 2
                # for branch lengths and J1, we take that of the first (of 100) tree
                # subsample
                if entries[4] and entries[6]:
                    patient_branch_lengths[patient] = np.array(
                        list(map(float, entries[4].split(";")[0].split()))
                    )
                    patient_j_one[patient] = float(entries[6].split()[0])
                else:
                    patient_branch_lengths[patient] = None
                    patient_j_one[patient] = None
                if len(entries) > 8:
                    if len(entries) > 9:
                        assert all("MM" in entry for entry in entries[8:]), line
                    successfully_read = False
                    try:
                        read_mixture_model = read_mixture_model_from_string(entries[8])
                        patient_mixture_models[patient] = read_mixture_model
                        successfully_read = True
                    except AssertionError:  # doesn't start with MM
                        pass
                    if successfully_read:
                        continue  # no need to fit the mixture model
                mixture_models_added = True
                patient_mixture_models[patient] = fit_mixture_model(
                    patient_mutational_burden[patient][:, 0], log_transform=True
                )
            if not patient_mutational_burden:
                raise ValueError(f"Replicate {key} doesn't exist in {self.data_path}")
        if mixture_models_added:
            # then we need to add these mixture models to the file
            with open(self.data_path, "r", encoding="utf-8") as simulation_file:
                simulation_file_lines = simulation_file.readlines()
            with open(self.data_path, "w", encoding="utf-8") as simulation_file:
                for line in simulation_file_lines:
                    if (
                        line == ""
                        or "Error" in line
                        or line.split("--")[0] != str(key + 1)
                    ):
                        simulation_file.write(line)
                    else:
                        patient = line.split("--")[1]
                        assert patient in patient_mixture_models, (
                            f"Patient {patient} not found in mixture models; "
                            f"all mixture models: {patient_mixture_models}"
                        )
                        mixture_model = patient_mixture_models[patient]
                        if "MM" in line:
                            # assert it's in all entries past 8, then remove them
                            assert all("MM" in entry for entry in line.split("--")[8:])
                            line = "--".join(line.split("--")[:8])
                        simulation_file.write(
                            line.strip()
                            + (
                                f"--{mixture_model.write_to_string()}"
                                if mixture_model is not None
                                else ""
                            )
                            + "\n"
                        )
        return ReplicateData(
            patient_mutational_burden,
            patient_branch_lengths,
            patient_j_one,
            patient_mixture_models,
        )

    @attach_error_info
    def __len__(self) -> int:
        try:
            with open(self.data_path, "r", encoding="utf-8") as simulation_file:
                return max(
                    int(line.split("--")[0])
                    for line in simulation_file.readlines()
                    if line != "" and "Error" not in line and "Resizing" not in line
                )
        except FileNotFoundError:
            return 0
        except ValueError as e:
            with open(self.data_path, "r", encoding="utf-8") as simulation_file:
                first_line = simulation_file.readline()
            raise ValueError(
                f"Error in file {self.data_path}:\nFirst line:\n" + first_line
            ) from e

    @attach_error_info
    def __iter__(self) -> Generator[ReplicateData, None, None]:
        for i in range(len(self)):
            yield self[i]


@dataclass
class SimulationData:
    parameter_values: dict[str, float]
    replicates: LazyReplicateList

    @property
    def parameter_values_string(self) -> str:
        return "_".join(
            f"{abbreviate_name(name)}={value:.2f}"
            for name, value in self.parameter_values.items()
        )


class LazySimulationList:
    data_directory: str
    replicates_per_simulation: int
    _filenames: list[str] | None = None

    def __init__(self, data_directory: str, replicates_per_simulation: int):
        self.data_directory = data_directory
        self.replicates_per_simulation = replicates_per_simulation

    def __str__(self) -> str:
        return (
            f"Lazy simulation list from {self.data_directory} "
            f"({self.replicates_per_simulation} replicates per simulation)"
        )

    @attach_error_info
    def __getitem__(self, key: int) -> SimulationData:
        try:
            filename = self.filenames[key]
        except IndexError as e:
            raise IndexError(
                f"Error in directory {self.data_directory}:\n{key} from {len(self)}"
            ) from e
        return SimulationData(
            parameter_values={
                param.split("=")[0]: float(param.split("=")[1])
                for param in filename.split(".txt")[0].split("_")
            },
            replicates=LazyReplicateList(f"{self.data_directory}/{filename}"),
        )

    @property
    @attach_error_info
    def filenames(self) -> list[str]:
        if self._filenames is None:
            self._filenames = sorted(os.listdir(self.data_directory))
        return self._filenames

    @attach_error_info
    def __len__(self) -> int:
        return len(self.filenames)

    @attach_error_info
    def __iter__(self) -> Generator[SimulationData, None, None]:
        for i in range(len(self)):
            yield self[i]


class ReplicateIndexer:
    paradigms: list[str]
    simulations_per_paradigm: int
    replicates_per_simulation: int

    def __init__(
        self,
        paradigms: list[str],
        simulations_per_paradigm: int,
        replicates_per_simulation: int,
    ):
        self.paradigms = paradigms
        self.simulations_per_paradigm = simulations_per_paradigm
        self.replicates_per_simulation = replicates_per_simulation

    def __str__(self) -> str:
        return (
            f"Replicate indexer with paradigms {self.paradigms}, "
            f"{self.simulations_per_paradigm} simulations per paradigm, "
            f"{self.replicates_per_simulation} replicates per simulation"
        )

    def get_replicate_index(
        self, paradigm: str, simulation_index: int, replicate_index: int
    ) -> int:
        assert paradigm in self.paradigms, f"Paradigm {paradigm} not found in dataset"
        assert (
            0 <= simulation_index < self.simulations_per_paradigm
        ), f"Simulation index {simulation_index} out of range"
        assert (
            0 <= replicate_index < self.replicates_per_simulation
        ), f"Replicate index {replicate_index} out of range"
        return (
            # number of replicates in 'earlier' paradigms
            self.paradigms.index(paradigm)
            * self.simulations_per_paradigm
            * self.replicates_per_simulation
            # number of replicates in 'earlier' simulations
            + simulation_index * self.replicates_per_simulation
            # index of replicate in this simulation
            + replicate_index
        )

    def get_replicate_position(self, index: int) -> tuple[str, int, int]:
        paradigm, index_within_paradigm = self.paradigm_from_index(index)
        simulation_index = index_within_paradigm // self.replicates_per_simulation
        replicate_index = index_within_paradigm % self.replicates_per_simulation
        return paradigm, simulation_index, replicate_index

    def __len__(self) -> int:
        return (
            self.paradigm_count
            * self.simulations_per_paradigm
            * self.replicates_per_simulation
        )

    def __iter__(self) -> Generator[int, None, None]:
        yield from range(len(self))

    def paradigm_from_index(self, index: int) -> tuple[str, int]:
        replicates_per_paradigm = (
            self.simulations_per_paradigm * self.replicates_per_simulation
        )
        return (
            self.paradigms[int(index // replicates_per_paradigm)],
            index % replicates_per_paradigm,
        )

    @property
    def paradigms_by_index(self) -> list[str]:
        return [self.paradigm_from_index(i)[0] for i in self]

    @property
    def paradigm_count(self) -> int:
        return len(self.paradigms)

    def count_pairs(self, similarity_level: SimilarityLevel) -> int:
        if similarity_level == SimilarityLevel.REPLICATE:
            return (
                self.paradigm_count
                * self.simulations_per_paradigm
                * self.replicates_per_simulation
                * (self.replicates_per_simulation - 1)
            ) // 2
        if similarity_level == SimilarityLevel.INTRA_PARADIGM:
            return (
                self.paradigm_count
                * self.simulations_per_paradigm
                * (self.simulations_per_paradigm - 1)
                * self.replicates_per_simulation**2
            ) // 2
        if similarity_level == SimilarityLevel.INTER_PARADIGM:
            return (
                self.paradigm_count
                * (self.paradigm_count - 1)
                * self.simulations_per_paradigm**2
                * self.replicates_per_simulation**2
            ) // 2
        raise ValueError("Invalid similarity level")

    def get_all_pairs(self, similarity_level: SimilarityLevel) -> list[tuple[int, int]]:
        if similarity_level == SimilarityLevel.REPLICATE:
            return [
                (
                    self.get_replicate_index(
                        paradigm, simulation_index, replicate_index_1
                    ),
                    self.get_replicate_index(
                        paradigm, simulation_index, replicate_index_2
                    ),
                )
                for paradigm in self.paradigms
                for simulation_index in range(self.simulations_per_paradigm)
                for replicate_index_1 in range(self.replicates_per_simulation)
                for replicate_index_2 in range(
                    replicate_index_1 + 1, self.replicates_per_simulation
                )
            ]
        if similarity_level == SimilarityLevel.INTRA_PARADIGM:
            return [
                (
                    self.get_replicate_index(
                        paradigm, simulation_index_1, replicate_index_1
                    ),
                    self.get_replicate_index(
                        paradigm, simulation_index_2, replicate_index_2
                    ),
                )
                for paradigm in self.paradigms
                for simulation_index_1 in range(self.simulations_per_paradigm)
                for simulation_index_2 in range(
                    simulation_index_1 + 1, self.simulations_per_paradigm
                )
                for replicate_index_1 in range(self.replicates_per_simulation)
                for replicate_index_2 in range(self.replicates_per_simulation)
            ]
        if similarity_level == SimilarityLevel.INTER_PARADIGM:
            return [
                (
                    self.get_replicate_index(
                        paradigm_1, simulation_index_1, replicate_index_1
                    ),
                    self.get_replicate_index(
                        paradigm_2, simulation_index_2, replicate_index_2
                    ),
                )
                for paradigm_1 in self.paradigms
                for paradigm_2 in self.paradigms
                for simulation_index_1 in range(self.simulations_per_paradigm)
                for simulation_index_2 in range(self.simulations_per_paradigm)
                for replicate_index_1 in range(self.replicates_per_simulation)
                for replicate_index_2 in range(self.replicates_per_simulation)
                if paradigm_1 < paradigm_2
            ]

    def get_group_pairs(
        self, similarity_level: SimilarityLevel, group_number: int, group_size: int
    ) -> list[tuple[int, int]]:
        pairs = self.get_all_pairs(similarity_level)
        return pairs[(group_number - 1) * group_size : group_number * group_size]

    def randomly_sample_pair(
        self, similarity_level: SimilarityLevel
    ) -> tuple[int, int]:
        if similarity_level == SimilarityLevel.REPLICATE:
            paradigm = random.choice(self.paradigms)
            simulation_index = random.randint(0, self.simulations_per_paradigm - 1)
            replicate_index_1 = random.randint(0, self.replicates_per_simulation - 1)
            replicate_index_2 = replicate_index_1
            while replicate_index_2 == replicate_index_1:
                replicate_index_2 = random.randint(
                    0, self.replicates_per_simulation - 1
                )
            return (
                self.get_replicate_index(paradigm, simulation_index, replicate_index_1),
                self.get_replicate_index(paradigm, simulation_index, replicate_index_2),
            )
        if similarity_level == SimilarityLevel.INTRA_PARADIGM:
            paradigm = random.choice(self.paradigms)
            simulation_index_1 = random.randint(0, self.simulations_per_paradigm - 1)
            simulation_index_2 = random.randint(0, self.simulations_per_paradigm - 1)
            return (
                self.get_replicate_index(
                    paradigm,
                    simulation_index_1,
                    random.randint(0, self.replicates_per_simulation - 1),
                ),
                self.get_replicate_index(
                    paradigm,
                    simulation_index_2,
                    random.randint(0, self.replicates_per_simulation - 1),
                ),
            )
        if similarity_level == SimilarityLevel.INTER_PARADIGM:
            paradigm_1 = random.choice(self.paradigms)
            paradigm_2 = random.choice(self.paradigms)
            simulation_index_1 = random.randint(0, self.simulations_per_paradigm - 1)
            simulation_index_2 = random.randint(0, self.simulations_per_paradigm - 1)
            return (
                self.get_replicate_index(
                    paradigm_1,
                    simulation_index_1,
                    random.randint(0, self.replicates_per_simulation - 1),
                ),
                self.get_replicate_index(
                    paradigm_2,
                    simulation_index_2,
                    random.randint(0, self.replicates_per_simulation - 1),
                ),
            )


class IdentifiabilityDataset:
    paradigm_simulation_lists: dict[str, LazySimulationList]
    replicate_indexer: ReplicateIndexer
    _filecounts_checked: bool = False

    def __init__(self, logging_directory: str, run_id: str):
        self.paradigm_simulation_lists = {}
        paradigms = sorted(
            [
                x
                for x in os.listdir(f"{logging_directory}/{run_id}/simulation_outputs/")
                if os.path.isdir(f"{logging_directory}/{run_id}/simulation_outputs/{x}")
            ],
            key=lambda x: (
                len(x.split("-")),
                *sorted(MODULE_ORDERING.index(y) for y in x.split("-")),
            ),
        )
        self.replicate_indexer = ReplicateIndexer(
            paradigms=paradigms,
            simulations_per_paradigm=len(
                os.listdir(
                    f"{logging_directory}/{run_id}/simulation_outputs/{paradigms[0]}"
                )
            ),
            replicates_per_simulation=get_replicates_per_simulation(
                f"{logging_directory}/{run_id}/simulation_outputs/{paradigms[0]}"
            ),
        )
        for paradigm in paradigms:
            self.paradigm_simulation_lists[paradigm] = LazySimulationList(
                f"{logging_directory}/{run_id}/simulation_outputs/{paradigm}",
                replicates_per_simulation=(
                    self.replicate_indexer.replicates_per_simulation
                ),
            )

    def check_filecounts(self) -> None:
        if self._filecounts_checked:
            return
        assert all(
            len(simulations) == self.replicate_indexer.simulations_per_paradigm
            for simulations in self.paradigm_simulation_lists.values()
        ), "Different number of simulations per paradigm"
        assert all(
            len(simulation.replicates)
            == self.replicate_indexer.replicates_per_simulation
            for paradigm in self.paradigms
            for simulation in self[paradigm]
        ), (
            f"Different number of replicates per simulation; "
            f"{self.replicate_indexer.replicates_per_simulation}\n\t"
            + "\n\t".join(
                f"{paradigm},{simulation_index},{len(simulation.replicates)}"
                for paradigm in self.paradigms
                for simulation_index, simulation in enumerate(self[paradigm])
            )
        )
        self._filecounts_checked = True

    def __str__(self) -> str:
        return (
            f"Identifiability dataset with {self.paradigm_count} paradigms, "
            f"{self.simulations_per_paradigm} simulations per paradigm, "
            f"{self.replicates_per_simulation} replicates per simulation"
        )

    def __getitem__(self, paradigm: str) -> LazySimulationList:
        return self.paradigm_simulation_lists[paradigm]

    @property
    def paradigm_count(self) -> int:
        return len(self.paradigm_simulation_lists)

    @property
    def paradigms(self) -> list[str]:
        return self.replicate_indexer.paradigms

    @property
    def simulations_per_paradigm(self) -> int:
        return self.replicate_indexer.simulations_per_paradigm

    @property
    def replicates_per_simulation(self) -> int:
        return self.replicate_indexer.replicates_per_simulation

    def get_replicate(self, index: int) -> ReplicateData:
        self.check_filecounts()
        (
            paradigm,
            simulation_index,
            replicate_index,
        ) = self.replicate_indexer.get_replicate_position(index)
        return self.paradigm_simulation_lists[paradigm][simulation_index].replicates[
            replicate_index
        ]

    @property
    def patient_list(self) -> list[str]:
        # assumes that all replicates have the same set of patients
        return list(self.get_replicate(0).patient_mutational_burden.keys())

    def get_parameter_values(self, index: int) -> dict[str, float]:
        paradigm, simulation_index, _ = self.replicate_indexer.get_replicate_position(
            index
        )
        return self.paradigm_simulation_lists[paradigm][
            simulation_index
        ].parameter_values

    def to_dataframe(self) -> pd.DataFrame:
        data = []
        for paradigm, paradigm_simulations in self.paradigm_simulation_lists.items():
            for simulation_index, simulation in enumerate(paradigm_simulations):
                parameter_values = simulation.parameter_values
                for replicate_index, replicate in enumerate(simulation.replicates):
                    for patient in replicate.patient_mutational_burden:
                        patient_mutational_burden = replicate.patient_mutational_burden[
                            patient
                        ]
                        patient_branch_lengths = replicate.patient_branch_lengths[
                            patient
                        ]
                        patient_j_one = replicate.patient_j_one[patient]
                        patient_mixture_models = replicate.patient_mixture_models[
                            patient
                        ]

                        data.append(
                            {
                                "paradigm": paradigm,
                                "simulation_index": simulation_index,
                                "replicate_index": replicate_index,
                                "patient": patient,
                                "mutational_burden": patient_mutational_burden[
                                    :, 0
                                ].tolist(),
                                "smoking_signature_mutational_burden": patient_mutational_burden[
                                    :, 1
                                ].tolist(),
                                "branch_lengths": (
                                    patient_branch_lengths.tolist()
                                    if patient_branch_lengths is not None
                                    else None
                                ),
                                "j_one": (
                                    patient_j_one if patient_j_one is not None else None
                                ),
                                "mixture_model_larger_mean": (
                                    patient_mixture_models.larger_mean
                                    if patient_mixture_models is not None
                                    else None
                                ),
                                "mixture_model_smaller_mean": (
                                    patient_mixture_models.smaller_mean
                                    if patient_mixture_models is not None
                                    else None
                                ),
                                "mixture_model_larger_weight": (
                                    patient_mixture_models.larger_weight
                                    if patient_mixture_models is not None
                                    else None
                                ),
                                "mixture_model_n_components": (
                                    patient_mixture_models.n_components
                                    if patient_mixture_models is not None
                                    else None
                                ),
                                "mixture_model_larger_mean_weight": (
                                    patient_mixture_models.larger_mean_weight
                                    if patient_mixture_models is not None
                                    else None
                                ),
                                **parameter_values,
                            }
                        )
        return pd.DataFrame(data)


def get_replicates_per_simulation(simulation_directory: str) -> int:
    file = os.listdir(simulation_directory)[0]
    with open(f"{simulation_directory}/{file}", "r", encoding="utf-8") as f:
        return max(
            int(line.split("--")[0])
            for line in f.readlines()
            if line != "" and "Error" not in line
        )
