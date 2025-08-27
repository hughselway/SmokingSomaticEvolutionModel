import json
from dataclasses import dataclass
import numpy as np


@dataclass
class PatientSimulationOutput:
    mutational_burden: list[int]
    smoking_signature_mutational_burden: list[int]
    phylogeny_branch_lengths: np.ndarray | None
    simulation_time: float
    tree_balance_indices: np.ndarray | None
    tree_calculation_time: float | None
    zero_population_error: bool
    final_cell_count: int
    min_cell_count: int
    max_cell_count: int

    # Ignores simulation time to aid reproducibility tests
    def __str__(self):
        attributes = [
            f"mutational_burden: {self.mutational_burden}",
            f"smoking_signature_mutational_burden: "
            f"{self.smoking_signature_mutational_burden}",
            f"phylogeny_branch_lengths: {self.phylogeny_branch_lengths}",
            f"tree_balance_indices: {self.tree_balance_indices}",
            f"zero_population_error: {self.zero_population_error}",
            f"final_cell_count: {self.final_cell_count}",
            f"min_cell_count: {self.min_cell_count}",
            f"max_cell_count: {self.max_cell_count}",
        ]
        return f"PatientSimulationOutput({', '.join(attributes)})"

    __repr__ = __str__

    def json_dict(self) -> dict:
        return {
            "mutational_burden": self.mutational_burden,
            "smoking_signature_mutational_burden": self.smoking_signature_mutational_burden,
            "phylogeny_branch_lengths": (
                self.phylogeny_branch_lengths.tolist()
                if self.phylogeny_branch_lengths is not None
                else None
            ),
            "simulation_time": self.simulation_time,
            "tree_balance_indices": (
                self.tree_balance_indices.tolist()
                if self.tree_balance_indices is not None
                else None
            ),
            "tree_calculation_time": self.tree_calculation_time,
            "zero_population_error": self.zero_population_error,
            "final_cell_count": self.final_cell_count,
            "min_cell_count": self.min_cell_count,
            "max_cell_count": self.max_cell_count,
        }


@dataclass
class SpatialPatientSimulationOutput:
    mutational_burden: list[int]
    smoking_signature_mutational_burden: list[int]
    phylogeny_branch_lengths: np.ndarray | None
    simulation_time: float
    tree_balance_indices: np.ndarray | None
    tree_calculation_time: float | None

    # Ignores simulation time to aid reproducibility tests
    def __str__(self):
        attributes = [
            f"mutational_burden: {self.mutational_burden}",
            f"smoking_signature_mutational_burden: "
            f"{self.smoking_signature_mutational_burden}",
            f"phylogeny_branch_lengths: {self.phylogeny_branch_lengths}",
            f"tree_balance_indices: {self.tree_balance_indices}",
        ]
        return f"SpatialPatientSimulationOutput({', '.join(attributes)})"

    __repr__ = __str__

    def json_dict(self) -> dict:
        return {
            "mutational_burden": self.mutational_burden,
            "smoking_signature_mutational_burden": self.smoking_signature_mutational_burden,
            "phylogeny_branch_lengths": (
                self.phylogeny_branch_lengths.tolist()
                if self.phylogeny_branch_lengths is not None
                else None
            ),
            "simulation_time": self.simulation_time,
            "tree_balance_indices": (
                self.tree_balance_indices.tolist()
                if self.tree_balance_indices is not None
                else None
            ),
            "tree_calculation_time": self.tree_calculation_time,
        }


def read_simulation_output_from_json(
    file_path: str, spatial: bool
) -> (
    list[dict[str, PatientSimulationOutput]]
    | list[dict[str, SpatialPatientSimulationOutput]]
):
    with open(file_path, "r", encoding="utf-8") as f:
        simulation_output = json.load(f)
    if spatial:
        return [
            {
                key: SpatialPatientSimulationOutput(
                    mutational_burden=patient["mutational_burden"],
                    smoking_signature_mutational_burden=patient[
                        "smoking_signature_mutational_burden"
                    ],
                    phylogeny_branch_lengths=(
                        np.array(patient["phylogeny_branch_lengths"])
                        if patient["phylogeny_branch_lengths"]
                        else None
                    ),
                    simulation_time=patient["simulation_time"],
                    tree_balance_indices=(
                        np.array(patient["tree_balance_indices"])
                        if patient["tree_balance_indices"]
                        else None
                    ),
                    tree_calculation_time=patient["tree_calculation_time"],
                )
                for key, patient in replicate.items()
            }
            for replicate in simulation_output
        ]
    return [
        {
            key: PatientSimulationOutput(
                mutational_burden=patient["mutational_burden"],
                smoking_signature_mutational_burden=patient[
                    "smoking_signature_mutational_burden"
                ],
                phylogeny_branch_lengths=(
                    np.array(patient["phylogeny_branch_lengths"])
                    if patient["phylogeny_branch_lengths"]
                    else None
                ),
                simulation_time=patient["simulation_time"],
                tree_balance_indices=(
                    np.array(patient["tree_balance_indices"])
                    if patient["tree_balance_indices"]
                    else None
                ),
                tree_calculation_time=patient["tree_calculation_time"],
                zero_population_error=patient["zero_population_error"],
                final_cell_count=patient["final_cell_count"],
                min_cell_count=patient["min_cell_count"],
                max_cell_count=patient["max_cell_count"],
            )
            for key, patient in replicate.items()
        }
        for replicate in simulation_output
    ]


def extract_mutational_burden(
    patient_simulation_output: PatientSimulationOutput | SpatialPatientSimulationOutput,
    compare_smoking_signature_mutations: bool,
) -> np.ndarray:
    if compare_smoking_signature_mutations:
        return np.column_stack(
            (
                patient_simulation_output.mutational_burden,
                patient_simulation_output.smoking_signature_mutational_burden,
            )
        )
    return np.array(patient_simulation_output.mutational_burden)


def parse_spatial_subprocess_output(
    subprocess_stdout: str, replicate_count: int
) -> list[dict[str, SpatialPatientSimulationOutput]]:
    assert_consistent_replicate_numbers(subprocess_stdout, replicate_count)
    return [
        {
            line.split("--")[1]: SpatialPatientSimulationOutput(
                mutational_burden=[int(x) for x in line.split("--")[2].split()],
                smoking_signature_mutational_burden=[
                    int(x) for x in line.split("--")[3].split()
                ],
                phylogeny_branch_lengths=(
                    np.array(
                        [
                            [int(x) for x in y.split()]
                            for y in line.split("--")[4].split(";")
                        ]
                    )
                    if line.split("--")[4]
                    else None
                ),
                simulation_time=float(line.split("--")[5]),
                tree_balance_indices=(
                    np.array([float(x) for x in line.split("--")[6].split()])
                    if line.split("--")[6]
                    else None
                ),
                tree_calculation_time=(
                    float(line.split("--")[7]) if line.split("--")[7] else None
                ),
            )
            for line in subprocess_stdout.splitlines()
            if "Warning:" not in line
            and "Resizing" not in line
            and int(line.split("--")[0]) == replicate_number
        }
        for replicate_number in range(1, replicate_count + 1)
    ]


def parse_non_spatial_subprocess_output(
    subprocess_stdout: str, replicate_count: int
) -> list[dict[str, PatientSimulationOutput]]:
    assert_consistent_replicate_numbers(subprocess_stdout, replicate_count)
    return [
        {
            line.split("--")[1]: PatientSimulationOutput(
                mutational_burden=[int(x) for x in line.split("--")[2].split()],
                smoking_signature_mutational_burden=[
                    int(x) for x in line.split("--")[3].split()
                ],
                phylogeny_branch_lengths=(
                    np.array(
                        [
                            [int(x) for x in y.split()]
                            for y in line.split("--")[4].split(";")
                        ]
                    )
                    if line.split("--")[4]
                    else None
                ),
                simulation_time=float(line.split("--")[5]),
                tree_balance_indices=(
                    np.array([float(x) for x in line.split("--")[6].split()])
                    if line.split("--")[6]
                    else None
                ),
                tree_calculation_time=(
                    float(line.split("--")[7]) if line.split("--")[7] else None
                ),
                zero_population_error=bool(line.split("--")[7]),
                final_cell_count=int(line.split("--")[8]),
                min_cell_count=int(line.split("--")[9]),
                max_cell_count=int(line.split("--")[10]),
            )
            for line in subprocess_stdout.splitlines()
            if (
                "Warning:" not in line
                and "Resizing" not in line
                and int(line.split("--")[0]) == replicate_number
            )
        }
        for replicate_number in range(1, replicate_count + 1)
    ]


def assert_consistent_replicate_numbers(subprocess_stdout, replicate_count):
    stdout_replicate_numbers = {
        int(line.split("--")[0])
        for line in subprocess_stdout.splitlines()
        if "Warning:" not in line and "Resizing" not in line
    }
    assert stdout_replicate_numbers == set(range(1, replicate_count + 1)), (
        f"simulation number mismatch: {set(range(1, replicate_count + 1))} vs "
        f"{stdout_replicate_numbers}"
    )
