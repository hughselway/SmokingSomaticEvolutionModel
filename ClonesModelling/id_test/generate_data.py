from argparse import Namespace
from datetime import datetime
from multiprocessing import Pool
import os
import subprocess
import sys

from .simulation_commands import get_simulation_commands

from ..parameters.hypothetical_paradigm_class import get_abc_hypothetical_paradigms
from ..parameters.parameter_class import get_skipped_parameter_values
from ..simulation.run_fast_simulation import run_fast_simulation
from ..parse_cmd_line_args import parse_id_test_arguments



def generate_identifiability_dataset(parsed_args: Namespace):
    current_time = datetime.now()
    run_id = parsed_args.run_id or f'idt_{current_time.strftime("%Y-%m-%d_%H-%M-%S")}'
    idt_dir = f"{parsed_args.logging_directory}/{run_id}"
    hypothetical_paradigms = get_abc_hypothetical_paradigms(
        parsed_args, get_skipped_parameter_values(parsed_args)
    )

    if parsed_args.high_performance_cluster:
        os.makedirs(f"{idt_dir}/dataset/hpc_logs", exist_ok=True)
        new_args = sys.argv[1:]
        if parsed_args.run_id is None:
            new_args.extend(["-rid", run_id])
        command_generation_job_id = (
            subprocess.run(
                f"qsub -N idtcomm_{current_time.strftime('%S-%M-%H_%d-%m-%Y')} -terse "
                f"-wd /cluster/project2/clones_modelling/ClonesModelling "
                f"-o {idt_dir}/dataset/hpc_logs/generate_commands.log -j y "
                "/cluster/project2/clones_modelling/ClonesModelling/ClonesModelling/"
                "hpc/generate_id_test_commands.sh".split() + new_args,
                capture_output=True,
                check=True,
            )
            .stdout.decode("utf-8")
            .split(".")[0]
            .strip()
        )
        simulation_count = (
            len(hypothetical_paradigms) * parsed_args.simulations_per_paradigm
        )
        os.system(
            f"qsub -hold_jid {command_generation_job_id} "
            f"-N idt_{current_time.strftime('%S-%M-%H_%d-%m-%Y')} "
            f"-t 1-{simulation_count} -wd {idt_dir}/dataset "
            "/cluster/project2/clones_modelling/ClonesModelling/ClonesModelling/"
            "hpc/generate_id_test_dataset.sh"
        )
    else:
        simulation_commands = get_simulation_commands(parsed_args, idt_dir)
        assert simulation_commands is not None
        # for command, output_filepath in simulation_commands:
        #     record_simulation(command, output_filepath)
        with Pool(parsed_args.number_of_processes) as pool:
            pool.starmap(record_simulation, simulation_commands)


def record_simulation(simulation_command: list[str], output_filepath: str) -> None:
    if os.path.exists(output_filepath):
        # print(f"Output file already exists: {output_filepath}")
        with open(output_filepath, "r", encoding="utf-8") as file:
            file_lines = file.readlines()
        # print(f"File has {len(file_lines)} lines; {output_filepath}")
        done = not all(line.startswith("Resizing") for line in file_lines)
        # print(f"boolean calculation completed; done = {done}; {output_filepath}")
        # if all(line.startswith("Resizing") for line in file_lines):
        if not done:
            # print(f"Removing empty file: {output_filepath}")
            os.remove(output_filepath)
        else:
            # 49 is the cohort size; this means all patients were recorded
            assert len(file_lines) == 49, (output_filepath, len(file_lines))
            # print(f"File already exists and is not empty: {output_filepath}")
            return
    # print(f"Running simulation command for {output_filepath}", flush=True)
    subprocess_stdout = run_fast_simulation(simulation_command, timeout=60 * 60)
    # print(f"Simulation completed: {output_filepath}")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "w", encoding="utf-8") as file:
        file.write(subprocess_stdout)


if __name__ == "__main__":
    generate_identifiability_dataset(parse_id_test_arguments())
