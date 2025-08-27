from argparse import Namespace
import os
from typing import Generator
import sys
import pyabc  # type: ignore


from ..abc.run_abc import (
    get_abc_hypothetical_paradigms,
    get_abc_priors,
    get_skipped_parameter_values,
)
from ..hpc.abc_batch import make_submissions_log
from ..parameters.hypothetical_paradigm_class import HypotheticalParadigm
from ..parameters.parameter_class import abbreviate_name
from ..parse_cmd_line_args import parse_id_test_arguments
from ..simulation.run_fast_simulation import (
    get_julia_simulation_command,
    ensure_sysimage_exists,
)


def get_simulation_commands(
    parsed_args: Namespace,
    idt_dir: str | None = None,
) -> Generator[tuple[list[str], str], None, None] | None:
    if idt_dir is None:
        assert parsed_args.high_performance_cluster
        idt_dir = f"{parsed_args.logging_directory}/{parsed_args.run_id}"
        time_string = parsed_args.logging_directory.split("_")[-1]
        make_submissions_log("id_test", " ".join(sys.argv[1:]), time_string)
        ensure_sysimage_exists(idt_dir, logger=None)
    if os.path.exists(f"{idt_dir}/dataset/simulation_commands.txt"):
        print(
            f"Simulation commands already exist in {idt_dir}/dataset/simulation_commands.txt"
        )
        if not parsed_args.high_performance_cluster:
            with open(
                f"{idt_dir}/dataset/simulation_commands.txt", "r", encoding="utf-8"
            ) as commands_file:
                for line in commands_file:
                    if line.strip():
                        command = line.strip().split()
                        output_filepath = next(commands_file).strip()
                        yield (command, output_filepath)
        return None

    hypothetical_paradigms = get_abc_hypothetical_paradigms(
        parsed_args, get_skipped_parameter_values(parsed_args)
    )
    parameter_priors, _ = get_abc_priors(hypothetical_paradigms, weight_paradigms=True)
    ensure_directories(idt_dir, hypothetical_paradigms)
    simulation_commands = list(
        randomise_simulation_commands(
            parsed_args,
            hypothetical_paradigms,
            parameter_priors,
            idt_dir,
        )
    )
    os.makedirs(f"{idt_dir}/dataset", exist_ok=True)
    with open(
        f"{idt_dir}/dataset/simulation_commands.txt", "w", encoding="utf-8"
    ) as commands_file:
        for command, output_filepath in simulation_commands:
            commands_file.write(" ".join(command) + "\n")
            commands_file.write(f"{output_filepath}\n\n")
    if not parsed_args.high_performance_cluster:
        return simulation_commands
    return None


def randomise_simulation_commands(
    parsed_args: Namespace,
    hypothetical_paradigms: list[HypotheticalParadigm],
    parameter_priors: list[pyabc.RV],
    idt_dir: str,
) -> Generator[tuple[list[str], str], None, None]:
    for hypothetical_paradigm, parameter_prior in zip(
        hypothetical_paradigms, parameter_priors
    ):
        for simulation_number in range(parsed_args.simulations_per_paradigm):
            parameter_values = get_random_parameter_values(
                hypothetical_paradigm, parameter_prior
            )
            simulation_command = get_id_test_simulation_command(
                parameter_values,
                hypothetical_paradigm.fixed_parameters,
                parsed_args.replicate_count,
                parsed_args,
                simulation_number,
            )
            output_filepath = (
                f"{idt_dir}/simulation_outputs/"
                + hypothetical_paradigm.get_modules_string(abbreviated=True)
                + "/"
                + "_".join(
                    [
                        f"{abbreviate_name(key)}={round(value, 5)}"
                        for key, value in parameter_values.items()
                    ]
                )
                + ".txt"
            )
            if os.path.exists(output_filepath):
                print(f"File already exists: {output_filepath}")
                continue
            if simulation_command is not None:
                yield (simulation_command, output_filepath)
            else:
                print(
                    "Failed to generate simulation command for "
                    + hypothetical_paradigm.get_modules_string(abbreviated=True)
                )


def ensure_directories(
    idt_dir: str,
    hypothetical_paradigms: list[HypotheticalParadigm],
) -> None:
    for hypothetical_paradigm in hypothetical_paradigms:
        os.makedirs(
            f"{idt_dir}/simulation_outputs/"
            + hypothetical_paradigm.get_modules_string(abbreviated=True),
            exist_ok=True,
        )


def get_random_parameter_values(
    hypothetical_paradigm: HypotheticalParadigm,
    parameter_prior: pyabc.RV,
) -> dict[str, float]:
    parameter_values: dict[str, float] = (
        hypothetical_paradigm.convert_dict_from_varying_scale(
            dict(parameter_prior.rvs())
        )
    )
    return parameter_values


def get_id_test_simulation_command(
    parameter_values: dict[str, float],
    fixed_parameters: dict[str, float],
    replicate_count: int,
    parsed_args: Namespace,
    seed: int = 0,
) -> list[str] | None:
    try:
        return get_julia_simulation_command(
            parameter_values,
            fixed_parameters,
            True,  # spatial
            seed,
            parsed_args.initial_basal_cell_number,
            parsed_args.steps_per_year,
            None,  # this_probe_logging_directory
            f"{parsed_args.logging_directory}/{parsed_args.run_id}",  # sysimage_directory
            replicate_count,
            parsed_args.include_infants,
            parsed_args.first_patient_test,
            parsed_args.exclude_nature_genetics,
            True,  # allow_early_stopping
            parsed_args.dynamic_normalisation_power,
            0,  # record_frequency
            None,  # console_logging_level
            None,  # file_logging_level
            True,  # calculate_tree_balance_index
            None,  # tree_subsample_count
            False,  # csv_logging
            True,  # record_phylogenies
            False,  # status_representative_test
            parsed_args.supersample_patient_cohort,
            heap_size_hint_gb=6,
        )
    except Exception as e:  # pylint: disable=broad-except
        print(e)
        return None


if __name__ == "__main__":
    get_simulation_commands(parse_id_test_arguments())
