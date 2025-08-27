import os
import statsmodels.stats.outliers_influence  # type: ignore
import numpy as np

import matplotlib.pyplot as plt

from ..classifier.generate_distance_matrix import read_idt_distances
from ..classifier.indexer import Indexer
from ..classifier.run_classifiers import get_classifier_directory

from ..calculate_distances import DISTANCE_FUNCTIONS

from ...parse_cmd_line_args import parse_id_test_arguments


def subset_distance_functions(
    all_distances_by_function: np.ndarray,
    indexer: Indexer,
    verbose: bool,
    classifier_directory: str,
    vif_threshold: float = 10,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Greedy search for independent distance functions. For each distance function, find
    the variance inflation factor (VIF). Remove the distance function with the highest
    VIF until all distance functions have VIF below the threshold.
    """
    save_directory = f"{classifier_directory}/independent_distance_functions/vif_threshold_{vif_threshold}"
    if os.path.exists(f"{save_directory}/iteration_0.npz"):
        return load_existing_distance_functions(save_directory)
    print(f"file {save_directory}/iteration_0.npz not found, calculating VIFs")

    distance_function_indices = np.arange(indexer.n_distance_functions)
    subset_distance_function_indices: list[np.ndarray] = []
    greedy_search_df_vifs: list[np.ndarray] = []

    while True:
        greedy_search_df_vifs.append(np.empty(len(distance_function_indices)))
        subset_distance_function_indices.append(distance_function_indices)
        if verbose:
            print("distance_function_indices: ", distance_function_indices)
        for df_index_position, df_index in enumerate(distance_function_indices):
            if verbose:
                print(df_index, end=" ", flush=True)
            greedy_search_df_vifs[-1][df_index_position] = (
                statsmodels.stats.outliers_influence.variance_inflation_factor(
                    all_distances_by_function.T, df_index_position
                )
            )
        max_vif_index = np.argmax(greedy_search_df_vifs[-1])
        if greedy_search_df_vifs[-1][max_vif_index] < vif_threshold:
            break
        if verbose:
            removed_df = indexer.distance_function_names[
                distance_function_indices[max_vif_index]
            ]
            print(
                f"Removing distance function {removed_df} with VIF "
                f"{greedy_search_df_vifs[-1][max_vif_index]} (index {max_vif_index})"
                f" from {greedy_search_df_vifs[-1]}"
            )
        all_distances_by_function = np.delete(
            all_distances_by_function, max_vif_index, axis=0
        )
        distance_function_indices = np.delete(distance_function_indices, max_vif_index)
    print(
        "independent distance functions: ",
        [indexer.distance_function_names[i] for i in distance_function_indices],
        distance_function_indices,
    )

    assert len(subset_distance_function_indices) == len(greedy_search_df_vifs)
    os.makedirs(save_directory, exist_ok=True)
    for ix, (iteration_df_indices, iteration_vifs) in enumerate(
        zip(subset_distance_function_indices, greedy_search_df_vifs)
    ):
        np.savez(
            f"{save_directory}/iteration_{ix}.npz",
            distance_function_indices=iteration_df_indices,
            greedy_search_df_vifs=iteration_vifs,
        )

    return (subset_distance_function_indices, greedy_search_df_vifs)


def load_existing_distance_functions(
    save_directory: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    subset_distance_function_indices: list[np.ndarray] = []
    greedy_search_df_vifs: list[np.ndarray] = []
    print(f"Loading existing independent distance functions from {save_directory}")
    i = 0
    while True:
        try:
            with np.load(f"{save_directory}/iteration_{i}.npz") as data:
                subset_distance_function_indices.append(
                    data["distance_function_indices"]
                )
                greedy_search_df_vifs.append(data["greedy_search_df_vifs"])
                i += 1
        except FileNotFoundError:
            break
    return subset_distance_function_indices, greedy_search_df_vifs


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    distance_function_names = [
        df_name
        for df_name in DISTANCE_FUNCTIONS
        if (
            "control" not in df_name
            and df_name
            not in [
                "2D_wasserstein",
                "2D_wasserstein_simplified",
                # "total_branch_length",
                "total_branch_length_squared",
            ]
        )
    ]
    print(distance_function_names)
    # protection_selection_paradigms = False
    # simulation_subset_index = None  # range(20)
    for protection_selection_paradigms in [False, True]:
        for simulation_subset_index in [None]:  # , *range(20)]:
            print(
                "\nprotection_selection_paradigms:",
                protection_selection_paradigms,
                "simulation_subset_index:",
                simulation_subset_index,
            )
            indexer = Indexer(
                False,
                parsed_args.max_modules,
                parsed_args.simulations_per_paradigm * parsed_args.replicate_count,
                include_true_data=True,
                distance_function_names=distance_function_names,
            )
            mds_directory, classifier_directory = get_classifier_directory(
                parsed_args.logging_directory,
                parsed_args.run_id,
                distance_function_names,
                patient_subset=None,
                restrict_to_first_replicate=False,
                restrict_to_n_module_paradigms=(
                    -1 if protection_selection_paradigms else None
                ),
                simulation_subset_index=simulation_subset_index,
                include_true_data=True,
            )
            print("classifier_directory: ", classifier_directory)
            distance_matrix = read_idt_distances(
                parsed_args.logging_directory,
                parsed_args.run_id,
                indexer,
                mds_directory,
                patient_subset=None,
            )
            if protection_selection_paradigms:
                distance_matrix = distance_matrix.subset_to_n_module_paradigms(-1)
            if simulation_subset_index is not None:
                distance_matrix = distance_matrix.subset_to_simulation_subset(
                    parsed_args.logging_directory,
                    parsed_args.run_id,
                    simulation_subset_index,
                )

            (
                distance_function_indices,
                greedy_search_df_vifs,
            ) = subset_distance_functions(
                np.array(
                    [
                        distance_matrix.get_paradigm_agnostic_distances(df_index)[
                            np.triu_indices(
                                distance_matrix.indexer.n_datapoints_excluding_true_data,
                                1,
                            )
                        ]
                        for df_index in range(
                            distance_matrix.indexer.n_distance_functions
                        )
                    ]
                ),
                indexer,
                verbose=(simulation_subset_index is None),
                classifier_directory=classifier_directory,
                vif_threshold=5,
            )
            # print(distance_function_indices)
            # print("distance function vifs by iteration: ", greedy_search_df_vifs)

            # os.makedirs(
            #     f"{classifier_directory}/independent_distance_functions", exist_ok=True
            # )
            # assert len(distance_function_indices) == len(greedy_search_df_vifs)
            # for ix, (iteration_df_indices, iteration_vifs) in enumerate(
            #     zip(distance_function_indices, greedy_search_df_vifs)
            # ):
            #     np.savez(
            #         f"{classifier_directory}/independent_distance_functions/iteration_{ix}.npz",
            #         distance_function_indices=iteration_df_indices,
            #         greedy_search_df_vifs=iteration_vifs,
            #     )
