import datetime
from itertools import combinations

from .run_classifiers import run_classifiers
from ..calculate_distances import DISTANCE_FUNCTIONS
from ...parse_cmd_line_args import parse_id_test_arguments


def run_df_subsets_classifiers(
    logging_directory: str,
    run_id: str,
    max_modules: int,
    include_2d_wasserstein: bool,
    simulation_replicates_per_paradigm: int,
    n_features_per_dist_fn_options: list[int],
    mds_replicate_count: int = 10,
    restrict_to_first_replicate: bool = False,
    restrict_to_n_module_paradigms: int | None = None,
):
    for distance_function_names_subset in reversed(
        get_distance_function_names_subsets(
            max_subset_size=3, include_subtractions=True
        )
    ):
        print(
            f"Running classifiers for {distance_function_names_subset}; "
            f"starting at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        run_classifiers(
            logging_directory=logging_directory,
            run_id=run_id,
            max_modules=max_modules,
            include_2d_wasserstein=include_2d_wasserstein,
            distance_function_names=distance_function_names_subset,
            simulation_replicates_per_paradigm=simulation_replicates_per_paradigm,
            n_features_per_dist_fn_options=n_features_per_dist_fn_options,
            mds_replicate_count=mds_replicate_count,
            restrict_to_first_replicate=restrict_to_first_replicate,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
            include_true_data=True,
        )


def get_distance_function_names_subsets(
    max_subset_size: int, include_subtractions: bool = False
) -> list[list[str]]:
    comparing_distance_function_names = [
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
        ]
        # and "control" not in df_name
    ]
    #     "wasserstein",
    #     "smoking_sig_only",
    #     "z_values",
    #     # "mixture_model",
    #     "mm_larger_weight_sq_diff",
    #     "mm_larger_weight_abs_diff",
    #     "mm_dominant_means_sq_diff",
    #     "mm_larger_means_sq_diff",
    #     "mm_smaller_means_sq_diff",
    #     "mm_weighted_means_by_dominance",
    #     "mm_weighted_means_by_position",
    #     "branch_length_wasserstein",
    #     "abs_j_one",
    #     "random_control",
    #     # "sum_control",
    # ]
    subsets = [
        list(combo)
        for r in range(1, max_subset_size + 1)
        for combo in combinations(comparing_distance_function_names, r)
    ]
    if include_subtractions:
        subsets += [
            list(combo)
            for r in range(
                len(comparing_distance_function_names) - max_subset_size,
                len(comparing_distance_function_names) + 1,
            )
            for combo in combinations(comparing_distance_function_names, r)
        ]
    return subsets


def get_n_whole_mm_df_groups(
    distance_function_names_subset: list[str], mm_df_groups: list[list[str]]
) -> int | None:
    n_whole_mm_df_groups = 0
    full_group_mm_dfs = set()
    for mm_df_group in mm_df_groups:
        if set(mm_df_group).issubset(distance_function_names_subset):
            n_whole_mm_df_groups += 1
            full_group_mm_dfs.update(mm_df_group)
    all_mm_dfs = set().union(*mm_df_groups)
    mm_dfs_in_subset = set(distance_function_names_subset).intersection(all_mm_dfs)
    if (len(mm_dfs_in_subset) != 0) and (full_group_mm_dfs != mm_dfs_in_subset):
        return None
    return n_whole_mm_df_groups


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    for restrict_to_n_module_paradigms in [None]:  # [-1, 5]:
        run_df_subsets_classifiers(
            logging_directory=parsed_args.logging_directory,
            run_id=parsed_args.run_id,
            max_modules=parsed_args.max_modules,
            include_2d_wasserstein=parsed_args.compare_smoking_signature_mutations,
            simulation_replicates_per_paradigm=(
                parsed_args.simulations_per_paradigm * parsed_args.replicate_count
            ),
            n_features_per_dist_fn_options=[2, 3, 5, 10, 20],
            mds_replicate_count=3,
            # restrict_to_first_replicate=True,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
        )
