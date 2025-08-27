import os
import subprocess
import time
from logging import Logger

from .output_class import (
    PatientSimulationOutput,
    SpatialPatientSimulationOutput,
    parse_non_spatial_subprocess_output,
    parse_spatial_subprocess_output,
)


class TooManyMutationsError(Exception):
    pass


class TooManyImmuneDeathsError(Exception):
    pass


class RejectionSamplingError(Exception):
    pass


class LowDivisionRateError(Exception):
    pass


def run_fast_simulation_abc(
    parameter_values_dict: dict[str, float],
    fixed_parameters_dict: dict[str, float],
    spatial: bool,
    seed: int,
    initial_basal_cell_number: int,
    steps_per_year: int,
    this_probe_logging_directory: str | None,
    sysimage_directory: str,
    replicate_count: int,
    include_infants: bool,
    first_patient_test: bool,
    exclude_nature_genetics: bool,
    allow_early_stopping: bool,
    dynamic_normalisation_power: float,
    record_frequency: int,
    console_logging_level: int | None,
    file_logging_level: int | None,
    calculate_tree_balance_index: bool,
    tree_subsample_count: int | None,
    csv_logging: bool,
    record_phylogenies: bool,
    status_representative_test: bool = False,
    timeout: int = 60 * 60,  # 1 hour
) -> (
    list[dict[str, PatientSimulationOutput]]
    | list[dict[str, SpatialPatientSimulationOutput]]
):
    julia_command = get_julia_simulation_command(
        parameter_values_dict,
        fixed_parameters_dict,
        spatial,
        seed,
        initial_basal_cell_number,
        steps_per_year,
        this_probe_logging_directory,
        sysimage_directory,
        replicate_count,
        include_infants,
        first_patient_test,
        exclude_nature_genetics,
        allow_early_stopping,
        dynamic_normalisation_power,
        record_frequency,
        console_logging_level,
        file_logging_level,
        calculate_tree_balance_index,
        tree_subsample_count,
        csv_logging,
        record_phylogenies,
        status_representative_test,
        heap_size_hint_gb=6,
    )
    subprocess_stdout = run_fast_simulation(julia_command, timeout)
    reprint_special_outputs(subprocess_stdout)
    if this_probe_logging_directory is not None:
        with open(
            f"{this_probe_logging_directory}/julia_cmd.txt", "w", encoding="utf-8"
        ) as f:
            f.write(" ".join(julia_command))
    try:
        if spatial:
            return parse_spatial_subprocess_output(subprocess_stdout, replicate_count)
        return parse_non_spatial_subprocess_output(subprocess_stdout, replicate_count)
    except (IndexError, ValueError) as error:
        print(f"{type(error).__name__} parsing subprocess output:")
        print(subprocess_stdout)
        raise


def reprint_special_outputs(subprocess_stdout):
    if "Warning:" in subprocess_stdout:
        for line in subprocess_stdout.splitlines():
            if "Warning:" in line:
                print(line)
    if "Resizing" in subprocess_stdout:
        resizing_count = 0
        for line in subprocess_stdout.splitlines():
            if "Resizing" in line:
                resizing_count += 1
        print(f"Resized phylogenies {resizing_count} times")


def get_julia_simulation_command(
    parameter_values_dict: dict[str, float],
    fixed_parameters_dict: dict[str, float],
    spatial: bool,
    seed: int,
    initial_basal_cell_number: int,
    steps_per_year: int,
    this_probe_logging_directory: str | None,
    sysimage_directory: str,
    replicate_count: int,
    include_infants: bool,
    first_patient_test: bool,
    exclude_nature_genetics: bool,
    allow_early_stopping: bool,
    dynamic_normalisation_power: float,
    record_frequency: int,
    console_logging_level: int | None,
    file_logging_level: int | None,
    calculate_tree_balance_index: bool,
    tree_subsample_count: int | None,
    csv_logging: bool,
    record_phylogenies: bool,
    status_representative_test: bool,
    supersample_patient_cohort: bool = False,
    heap_size_hint_gb: int | None = None,
) -> list[str]:
    # macOS sysimage fails so only use on cluster
    on_cluster = ("cluster" in sysimage_directory) or ("SAN" in sysimage_directory)
    return (
        ["julia"]
        + ([f"--sysimage={sysimage_directory}/sysimage.so"] * on_cluster)
        + ([f"--heap-size={heap_size_hint_gb}G"] * (heap_size_hint_gb is not None))
        + [
            "--project=ClonesModelling/FastSimulation",
            "ClonesModelling/FastSimulation/run_simulation_abc.jl",
            "--seed=" + str(int(seed)),
            "--initial_basal_cell_number=" + str(initial_basal_cell_number),
            "--steps_per_year=" + str(steps_per_year),
            "--dynamic_normalisation_power=" + str(dynamic_normalisation_power),
            "--record_frequency=" + str(record_frequency),
        ]
        + ["--this_probe_logging_directory=" + str(this_probe_logging_directory)]
        * (this_probe_logging_directory is not None)
        + ["--spatial"] * spatial
        + ["--replicate_count=" + str(replicate_count)]
        + ["--include_infants"] * include_infants
        + ["--first_patient_test"] * first_patient_test
        + ["--exclude_nature_genetics"] * exclude_nature_genetics
        + ["--supersample_patient_cohort"] * supersample_patient_cohort
        + ["--allow_early_stopping"] * allow_early_stopping
        + ["--console_logging_level=" + str(console_logging_level)]
        * (console_logging_level is not None)
        + ["--file_logging_level=" + str(file_logging_level)]
        * (file_logging_level is not None)
        + ["--calculate_tree_balance_index"] * calculate_tree_balance_index
        + [f"--tree_subsample_count={tree_subsample_count}"]
        * (tree_subsample_count is not None)
        + ["--record_phylogenies"] * record_phylogenies
        + ["--csv_logging"] * csv_logging
        + ["--status_representative_test"] * status_representative_test
        + [
            f"--{arg}="
            + str(
                (
                    parameter_values_dict[arg]
                    if arg in parameter_values_dict
                    else fixed_parameters_dict[arg]
                )
                if arg != "protected_region_radius"
                else (
                    int(
                        parameter_values_dict[arg]
                        if arg in parameter_values_dict
                        else fixed_parameters_dict[arg]
                    )
                )
            )
            for arg in [
                "smoking_mutation_rate_augmentation",
                "non_smoking_mutations_per_year",
                "fitness_change_scale",
                "smoking_division_rate_augmentation",
                "non_smoking_divisions_per_year",
                "mutation_rate_multiplier_shape",
                "quiescent_fraction",
                "quiescent_divisions_per_year",
                "ambient_quiescent_divisions_per_year",
                "quiescent_gland_cell_count",
                "quiescent_protection_coefficient",
                "protected_fraction",
                "protection_coefficient",
                "protected_region_radius",
                "immune_death_rate",
                "smoking_immune_coeff",
                "smoking_driver_fitness_augmentation",
            ]
            if (
                arg not in fixed_parameters_dict
                or fixed_parameters_dict[arg] is not None
            )
            and (
                arg not in parameter_values_dict
                or parameter_values_dict[arg] is not None
            )
        ]
    )


def run_fast_simulation(julia_command: list[str], timeout: int) -> str:
    subprocess_output = subprocess.run(
        julia_command, check=False, capture_output=True, timeout=timeout
    )
    subprocess_stdout = subprocess_output.stdout.decode("utf-8")
    if subprocess_output.stderr:
        stderr_text = subprocess_output.stderr.decode("utf-8")
        if "Too many mutations" in stderr_text:
            raise TooManyMutationsError()
        if "TooManyImmuneDeathsError" in stderr_text:
            raise TooManyImmuneDeathsError()
        if "RejectionSamplingError" in stderr_text:
            raise RejectionSamplingError()
        if "LowDivisionRateError" in stderr_text:
            raise LowDivisionRateError()
        raise RuntimeError(
            f"error in julia simulation with julia_command {' '.join(julia_command)}:\n"
            f"{subprocess_output.stderr.decode('utf-8')}\nJulia stdout:\n"
            f"{subprocess_stdout}\n---End of Julia stdout---"
        )
    return subprocess_stdout


def ensure_sysimage_exists(directory: str, logger: Logger | None) -> None:
    if os.path.exists(f"{directory}/sysimage.so"):
        return
    start_time = time.perf_counter()
    subprocess.run(
        [
            "julia",
            "--project=ClonesModelling/FastSimulation",
            "ClonesModelling/FastSimulation/create_sysimage.jl",
            "--this_probe_logging_directory=" + directory,
        ],
        check=True,
    )
    if logger is not None:
        logger.info(
            "Created sysimage; time %s",
            time.strftime("%H:%M:%S", time.gmtime(time.perf_counter() - start_time)),
        )
