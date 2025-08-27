import datetime
import os
import numpy as np

from .run_classifiers import run_classifiers

from ..calculate_distances import DISTANCE_FUNCTIONS
from ..read_data import IdentifiabilityDataset

from ...parse_cmd_line_args import parse_id_test_arguments


def create_simulation_mask(
    logging_directory: str,
    run_id: str,
    subset_replicate_index: int,
    subset_simulations_per_paradigm: int,
) -> None:
    if os.path.exists(
        f"{logging_directory}/{run_id}/classifiers/"
        f"simulation_subset_masks/subset_replicate_{subset_replicate_index}.npz"
    ):
        print(
            f"Simulation mask for subset {subset_replicate_index} already exists; "
            f"skipping creation"
        )
        return
    dataset = IdentifiabilityDataset(logging_directory, run_id)
    paradigm_selected_simulations = {
        paradigm: np.random.choice(
            dataset.simulations_per_paradigm,
            subset_simulations_per_paradigm,
            replace=False,
        )
        for paradigm in dataset.paradigms
    }
    os.makedirs(
        f"{logging_directory}/{run_id}/classifiers/simulation_subset_masks",
        exist_ok=True,
    )
    np.savez(
        f"{logging_directory}/{run_id}/classifiers/simulation_subset_masks/"
        f"subset_replicate_{subset_replicate_index}.npz",
        **paradigm_selected_simulations,
    )


def run_simulation_subset_classifiers(
    subset_simulations_per_paradigm_values: list[int],
    subset_replicate_count: int,
    logging_directory: str,
    run_id: str,
    max_modules: int,
    include_2d_wasserstein: bool,
    simulation_replicates_per_paradigm: int,
    n_features_per_dist_fn_options: list[int],
    mds_replicate_count: int,
    disance_function_names: list[str],
    restrict_to_first_replicate: bool = False,
    restrict_to_n_module_paradigms: int | None = None,
    include_true_data: bool = False,
) -> None:
    subset_replicate_index = 0
    for subset_simulations_per_paradigm in subset_simulations_per_paradigm_values:
        for i in range(subset_replicate_count):
            if (
                subset_simulations_per_paradigm == simulation_replicates_per_paradigm
                and i > 0
            ):
                # simulating the same number of simulations as the full dataset, only
                # need to one subset replicate
                break
            try:
                print(
                    f"Running classifiers for simulation subset {subset_replicate_index} "
                    f"({i+1}/{subset_replicate_count} at {subset_simulations_per_paradigm} sims); "
                    f"starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                create_simulation_mask(
                    logging_directory,
                    run_id,
                    subset_replicate_index,
                    subset_simulations_per_paradigm,
                )
                run_classifiers(
                    logging_directory=logging_directory,
                    run_id=run_id,
                    max_modules=max_modules,
                    include_2d_wasserstein=include_2d_wasserstein,
                    distance_function_names=disance_function_names,
                    simulation_replicates_per_paradigm=simulation_replicates_per_paradigm,
                    n_features_per_dist_fn_options=n_features_per_dist_fn_options,
                    mds_replicate_count=mds_replicate_count,
                    restrict_to_first_replicate=restrict_to_first_replicate,
                    restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
                    simulation_subset_index=subset_replicate_index,
                    include_true_data=include_true_data,
                )
            except TypeError:
                pass
            subset_replicate_index += 1


if __name__ == "__main__":
    args = parse_id_test_arguments()
    for restrict_to_n_module_paradigms_ in [-1, None]:
        run_simulation_subset_classifiers(
            subset_simulations_per_paradigm_values=[5, 10, 20, 50, 100, 150, 200],
            subset_replicate_count=3,
            logging_directory=args.logging_directory,
            run_id=args.run_id,
            max_modules=args.max_modules,
            include_2d_wasserstein=args.compare_smoking_signature_mutations,
            simulation_replicates_per_paradigm=(
                args.simulations_per_paradigm * args.replicate_count
            ),
            n_features_per_dist_fn_options=[2, 3, 5, 10, 20],
            mds_replicate_count=3,
            disance_function_names=[
                df_name
                for df_name in DISTANCE_FUNCTIONS
                if df_name
                not in [
                    "total_branch_length_squared",
                    "sum_control",
                    "zero_control",
                    # "mean_subtracted_2D_simplified",
                    # "l2_j_one",
                    "2D_wasserstein",
                    # ## two DFs removed because they didn't map properly in the idt_2025-02-05_14-53-34 run
                    # "mean_subtracted",
                    # "total_branch_length",
                    #
                    ## extra DFs removed for true data comparison analysis
                    # "random_control",
                    # # 0th percentile idt_2025-04-07_10-16-59
                    # # - none
                    # # 10th percentile idt_2025-04-07_10-16-59
                    # "mean_subtracted",
                    # "mean_subtracted_2D_simplified",
                    # "abs_j_one",
                    # "l2_j_one",
                    # # 20th percentile (as well as above) idt_2025-04-07_10-16-59
                    # "z_values",
                    # "branch_length_wasserstein",
                    # # 25th percentile (as well as above) idt_2025-04-07_10-16-59
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "mm_larger_means_sq_diff",
                    # "total_branch_length",
                    #
                    # # 5th percentile idt_2025-02-05_14-53-34
                    # "mean_subtracted_2D_simplified",
                    # "l2_j_one",
                    # # 10th percentile (as well as 5th) idt_2025-02-05_14-53-34
                    # "abs_j_one",
                    # # 25th percentile (as well as above) idt_2025-02-05_14-53-34
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "mm_larger_means_sq_diff",
                    # "branch_length_wasserstein",
                    #
                    # # 5th percentile idt_2024-10-03_19-04-47
                    # "abs_j_one",
                    # "l2_j_one",
                    # # 10th percentile (as well as 5th) idt_2024-10-03_19-04-47
                    # "mean_subtracted",
                    # "mean_subtracted_2D_simplified",
                    # # 25th percentile (as well as above) idt_2024-10-03_19-04-47
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "z_values",
                    # "mm_larger_means_sq_diff",
                ]
                # and "control" not in df_name
            ],
            restrict_to_first_replicate=False,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms_,
            include_true_data=True,
        )
