import os
from multiprocessing import Pool
import subprocess

from .calculate_distances import (
    calculate_distances,
    SAMPLES_PER_GROUP,
    get_groups_per_similarity_level,
)
from .plot import plot_id_test
from .read_data import SimilarityLevel

from ..parse_cmd_line_args import parse_id_test_comparison_arguments


def compare_id_dataset(
    logging_directory: str,
    run_id: str,
    n_pairs_to_sample: int | None,
    high_performance_cluster: bool,
    hold_jid: int | None,
    include_2d_wasserstein: bool,
    number_of_processes: int = 8,
):
    groups_per_similarity_level = get_groups_per_similarity_level(
        n_pairs_to_sample, logging_directory, run_id
    )
    print(
        "Groups per similarity level:\n\t"
        + "\n\t".join(
            f"{similarity_level.name}: {n_groups_by_sm_sig[include_2d_wasserstein]}"
            for similarity_level, n_groups_by_sm_sig in groups_per_similarity_level.items()
        ),
    )
    if high_performance_cluster:
        comparison_batch_job_ids = submit_comparison_jobs(
            logging_directory,
            run_id,
            n_pairs_to_sample,
            hold_jid,
            include_2d_wasserstein,
            groups_per_similarity_level,
        )
        submit_plot_job(
            logging_directory,
            run_id,
            n_pairs_to_sample,
            include_2d_wasserstein,
            comparison_batch_job_ids,
        )
    else:
        with Pool(number_of_processes) as pool:
            pool.starmap(
                calculate_distances,
                [
                    (
                        logging_directory,
                        run_id,
                        similarity_level,
                        (
                            SAMPLES_PER_GROUP[include_2d_wasserstein][similarity_level]
                            if n_pairs_to_sample is not None
                            else None
                        ),
                        include_2d_wasserstein,
                        group_index + 1,
                    )
                    for similarity_level, n_groups_by_sm_sig in groups_per_similarity_level.items()
                    for group_index in range(n_groups_by_sm_sig[include_2d_wasserstein])
                ],
            )
        plot_id_test(
            logging_directory, run_id, n_pairs_to_sample, include_2d_wasserstein
        )


def submit_comparison_jobs(
    logging_directory: str,
    run_id: str,
    n_pairs_to_sample: int | None,
    hold_jid: int | None,
    include_2d_wasserstein: bool,
    groups_per_similarity_level: dict[SimilarityLevel, dict[bool, int]],
) -> list[str]:
    comparison_batch_job_ids = []
    for similarity_level, n_groups_by_sm_sig in groups_per_similarity_level.items():
        if n_groups_by_sm_sig[include_2d_wasserstein] == 0:
            continue
        samples_this_group = SAMPLES_PER_GROUP[include_2d_wasserstein][similarity_level]
        hpc_output_dir = (
            f"{logging_directory}/{run_id}/distance/hpc/{similarity_level.name.lower()}"
        )
        os.makedirs(hpc_output_dir, exist_ok=True)
        comparison_batch_job_ids.append(
            subprocess.run(
                (
                    f"qsub -N idtc_{similarity_level.name}_{run_id} -j y "
                    f"-terse -wd {hpc_output_dir} "
                    f"-t 1-{n_groups_by_sm_sig[include_2d_wasserstein]} "
                    + (f"-hold_jid {hold_jid} " * (hold_jid is not None))
                    + "/cluster/project2/clones_modelling/ClonesModelling/"
                    f"ClonesModelling/hpc/compare_id_test_dataset.sh "
                    f"-ld {logging_directory} -rid {run_id} -hpc "
                    f"-sl {similarity_level.value} "
                    + (f"-nps {samples_this_group} " * (n_pairs_to_sample is not None))
                    + ("-iss" * (not include_2d_wasserstein))
                ).split(),
                capture_output=True,
                check=True,
            )
            .stdout.decode("utf-8")
            .split(".")[0]
            .strip()
        )
    return comparison_batch_job_ids


def submit_plot_job(
    logging_directory: str,
    run_id: str,
    n_pairs_to_sample: int | None,
    include_2d_wasserstein: bool,
    comparison_batch_job_ids: list[str] | None,
) -> None:
    os.system(
        f"qsub -N idtplot_{run_id} "
        + (
            f"-hold_jid {','.join(comparison_batch_job_ids)} "
            if (
                comparison_batch_job_ids is not None
                and len(comparison_batch_job_ids) > 0
            )
            else ""
        )
        + f"-wd {logging_directory}/{run_id}/distance/hpc "
        "/cluster/project2/clones_modelling/ClonesModelling/ClonesModelling/"
        "hpc/plot_id_test.sh "
        f"-ld {logging_directory} -rid {run_id} -hpc "
        + (f"-nps {n_pairs_to_sample} " * (n_pairs_to_sample is not None))
        + ("-iss" * (not include_2d_wasserstein))
    )


if __name__ == "__main__":
    parsed_args = parse_id_test_comparison_arguments()
    compare_id_dataset(
        parsed_args.logging_directory,
        parsed_args.run_id,
        parsed_args.n_pairs_to_sample,
        parsed_args.high_performance_cluster,
        parsed_args.hold_jid,
        parsed_args.compare_smoking_signature_mutations,
    )
