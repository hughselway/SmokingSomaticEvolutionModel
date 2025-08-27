import datetime
import numpy as np

from .run_classifiers import run_classifiers
from ..calculate_distances import DISTANCE_FUNCTIONS
from ...parse_cmd_line_args import parse_id_test_arguments


def run_pt_subset_classifiers(
    logging_directory: str,
    run_id: str,
    max_modules: int,
    include_2d_wasserstein: bool,
    simulation_replicates_per_paradigm: int,
    n_features_per_dist_fn_options: list[int],
    mds_replicate_count: int,
    distance_function_names: list[str] | None,
    subset_replicates_to_consider: int | None = None,
    restrict_to_first_replicate: bool = False,
    restrict_to_n_module_paradigms: int | None = None,
    include_true_data: bool = False,
):
    # subsamples_npz = np.load(f"{logging_directory}/{run_id}/subsamples.npz")
    # subsample_replicate_count = subsamples_npz["subsamples"].shape[1]
    # total_subsample_count = subsamples_npz["subsamples"].shape[0] * min(
    #     subsample_replicate_count, subset_replicates_to_consider
    # )
    # idx = 0
    # for subsample_size in subsamples_npz["subsample_sizes"]:
    #     for subsample_replicate_index in range(
    #         min(subsample_replicate_count, subset_replicates_to_consider)
    #     ):
    #         idx += 1
    #         print(
    #             f"Running classifiers for subsample size {subsample_size}, replicate {subsample_replicate_index}; "
    #             f"(subsample {idx}/{total_subsample_count}); "
    #             f"starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    #         )
    #         subsample_name = f"{subsample_size}_r{subsample_replicate_index}"
    for subsample_name in [
        "status_representatives_distance",
        "nature_patients_distance",
        "total_distance",
        "nature_genetics_patients_distance",
    ]:
        print(
            f"Running classifiers for subsample {subsample_name}; starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        run_classifiers(
            logging_directory=logging_directory,
            run_id=run_id,
            max_modules=max_modules,
            include_2d_wasserstein=include_2d_wasserstein,
            distance_function_names=(
                distance_function_names
                if subsample_name != "nature_genetics_patients_distance"
                else [
                    df_name
                    for df_name in distance_function_names
                    if df_name
                    not in [
                        "total_branch_length",
                        "total_branch_length_squared",
                        "branch_length_wasserstein",
                        "abs_j_one",
                        "l2_j_one",
                    ]
                ]
                # else [
                #     # No phylogeny distances, NatGen don't have phylogenies
                #     "wasserstein",
                #     "2D_wasserstein_simplified",
                #     "smoking_sig_only",
                #     "z_values",
                #     "mean_subtracted",
                #     "mean_subtracted_2D_simplified",
                #     "mm_larger_weight_sq_diff",
                #     "mm_larger_weight_abs_diff",
                #     "mm_dominant_means_sq_diff",
                #     "mm_larger_means_sq_diff",
                #     "mm_smaller_means_sq_diff",
                #     "mm_weighted_means_by_dominance",
                #     "random_control",
                # ]
            ),
            # distance_function_names=[
            #     "wasserstein",
            #     "2D_wasserstein_simplified",
            #     "smoking_sig_only",
            #     "z_values",
            #     "mean_subtracted",
            #     "mean_subtracted_2D_simplified",
            #     "mixture_model",
            #     "mm_larger_weight_sq_diff",
            #     "mm_larger_weight_abs_diff",
            #     "mm_dominant_means_sq_diff",
            #     "mm_larger_means_sq_diff",
            #     "mm_smaller_means_sq_diff",
            #     "mm_weighted_means_by_dominance",
            #     "total_branch_length",
            #     "total_branch_length_squared",
            #     "branch_length_wasserstein",
            #     "abs_j_one",
            #     "l2_j_one",
            #     "random_control",
            #     "sum_control",
            # ],
            simulation_replicates_per_paradigm=simulation_replicates_per_paradigm,
            n_features_per_dist_fn_options=n_features_per_dist_fn_options,
            mds_replicate_count=mds_replicate_count,
            patient_subset=subsample_name,
            restrict_to_first_replicate=restrict_to_first_replicate,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
            include_true_data=include_true_data,
        )


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    for restrict_to_n_module_paradigms in [None, -1]:
        run_pt_subset_classifiers(
            logging_directory=parsed_args.logging_directory,
            run_id=parsed_args.run_id,
            max_modules=parsed_args.max_modules,
            include_2d_wasserstein=parsed_args.compare_smoking_signature_mutations,
            simulation_replicates_per_paradigm=(
                parsed_args.simulations_per_paradigm * parsed_args.replicate_count
            ),
            n_features_per_dist_fn_options=[2, 3, 5, 10, 20],
            mds_replicate_count=3,
            distance_function_names=[
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
                ]
                # and "control" not in df_name
            ],
            # subset_replicates_to_consider=3,  # up to 3 next
            restrict_to_first_replicate=False,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
            include_true_data=True,
        )
