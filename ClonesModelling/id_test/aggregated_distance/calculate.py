import os
import numpy as np

from .independent_distance_functions import subset_distance_functions

from ..classifier.generate_distance_matrix import read_idt_distances, DistanceMatrix
from ..classifier.indexer import Indexer
from ..classifier.run_classifiers import get_classifier_directory

from ...parse_cmd_line_args import parse_id_test_arguments

from ..calculate_distances import DISTANCE_FUNCTIONS
from ...parameters.hypothetical_paradigm_class import get_hypothetical_paradigm


def calculate_aggregated_distances(
    logging_directory: str,
    run_id: str,
    max_modules: int,
    simulation_replicates_per_paradigm: int,
    distance_function_names: list[str] | None,
    protection_selection_paradigms: bool,
    simulation_subset_index: int | None,
    include_true_data: bool,
    vif_threshold: float | None,
    normalisation_method: str,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mds_directory, classifier_directory = get_classifier_directory(
        logging_directory,
        run_id,
        distance_function_names,
        patient_subset=None,
        restrict_to_first_replicate=False,
        restrict_to_n_module_paradigms=-1 if protection_selection_paradigms else None,
        simulation_subset_index=simulation_subset_index,
        include_true_data=include_true_data,
    )
    save_path = (
        f"{classifier_directory}/classifier_outputs/aggregated_distances/"
        f"{normalisation_method}_normalised/vif_threshold_{vif_threshold}.npz"
    )
    if os.path.exists(save_path):
        if verbose:
            print(f"Loading aggregated distances from {save_path}")
        saved_data = np.load(save_path)
        if len(saved_data["aggregated_distances"]) > 0:
            if not include_true_data:
                return (
                    saved_data["aggregated_distances"],
                    None,
                    saved_data["normalisation_factors"],
                    saved_data["included_df_indices"],
                )
            if len(saved_data["true_data_aggregated_distances"]) > 0:
                return (
                    saved_data["aggregated_distances"],
                    saved_data["true_data_aggregated_distances"],
                    saved_data["normalisation_factors"],
                    saved_data["included_df_indices"],
                )
        # otherwise, recalculate
    ## new code for fixed paradigm case (for param_fit_2025-06-06)
    indexer = Indexer.__new__(Indexer)
    indexer.distance_function_names = distance_function_names or [
        dist_fn_name
        for dist_fn_name in DISTANCE_FUNCTIONS
        if (dist_fn_name != "2D_wasserstein") and dist_fn_name != "zero_control"
    ]
    indexer.hypothetical_paradigms = [
        get_hypothetical_paradigm(
            hypothesis_module_names=["q", "ir"],
            spatial=True,
            skipped_parameters=["mutation_rate_multiplier_shape"],
        )
    ]
    indexer.simulation_replicates_per_paradigm = simulation_replicates_per_paradigm
    indexer.include_true_data = include_true_data

    distance_matrix = read_idt_distances(
        logging_directory,
        run_id,
        indexer,
        # Indexer(
        #     False,
        #     max_modules,
        #     simulation_replicates_per_paradigm,
        #     include_true_data,
        #     distance_function_names,
        # ),
        mds_directory,
        patient_subset=None,
    )
    if protection_selection_paradigms:
        distance_matrix = distance_matrix.subset_to_n_module_paradigms(-1)
    if simulation_subset_index is not None:
        distance_matrix = distance_matrix.subset_to_simulation_subset(
            logging_directory, run_id, simulation_subset_index
        )
    subset_distance_function_indices = (
        subset_distance_functions(
            np.array(
                [
                    distance_matrix.get_paradigm_agnostic_distances(df_index)[
                        np.triu_indices(
                            distance_matrix.indexer.n_datapoints_excluding_true_data,
                            1,
                        )
                    ]
                    for df_index in range(distance_matrix.indexer.n_distance_functions)
                ]
            ),
            distance_matrix.indexer,
            verbose=verbose,
            classifier_directory=classifier_directory,
            vif_threshold=vif_threshold,
        )[0][-1]
        if vif_threshold is not None
        else np.arange(distance_matrix.indexer.n_distance_functions)
    )
    df_subset_distance_matrix = distance_matrix.subset_distance_functions(
        # Indexer(
        #     False,
        #     -1 if protection_selection_paradigms else max_modules,
        #     simulation_replicates_per_paradigm,
        #     include_true_data,
        #     [
        #         str(x)
        #         for x in np.array(distance_matrix.indexer.distance_function_names)[
        #             subset_distance_function_indices
        #         ]
        #     ],
        # )
        indexer
    )
    unnormalised_symmetric_distance_matrix = get_symmetric_distance_matrix(
        df_subset_distance_matrix
    )
    if verbose:
        print(
            "unnormed symmetric distance matrix shape: ",
            unnormalised_symmetric_distance_matrix.shape,
        )
    symmetric_distance_matrix, normalisation_factors = normalise_distances(
        unnormalised_symmetric_distance_matrix, normalisation_method
    )
    if verbose:
        print("symmetric distance matrix shape: ", symmetric_distance_matrix.shape)

    aggregated_distance = symmetric_distance_matrix.mean(axis=0)
    if verbose:
        print("aggregated distance shape: ", aggregated_distance.shape)

    if include_true_data:
        assert df_subset_distance_matrix.true_data_distance is not None
        if verbose:
            print(
                "true data distance shape: ",
                df_subset_distance_matrix.true_data_distance.shape,
            )
        # same normalisation factors as for the simulated data
        normalised_true_data_distance = (
            df_subset_distance_matrix.true_data_distance
            / normalisation_factors[:, None, None]
        )
        true_data_aggregated_distance = normalised_true_data_distance.mean(axis=0)
    else:
        true_data_aggregated_distance = None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        aggregated_distances=aggregated_distance,
        true_data_aggregated_distances=(
            true_data_aggregated_distance
            if true_data_aggregated_distance is not None
            else []
        ),
        normalisation_factors=normalisation_factors,
        included_df_indices=subset_distance_function_indices,
    )
    return (
        aggregated_distance,
        true_data_aggregated_distance,
        normalisation_factors,
        subset_distance_function_indices,
    )


def get_symmetric_distance_matrix(distance_matrix: DistanceMatrix) -> np.ndarray:
    symmetric_distance_matrix = np.zeros_like(distance_matrix.distance_matrix)
    added_flag_matrix = np.zeros_like(distance_matrix.distance_matrix, dtype=bool)

    for paradigm1_index in range(distance_matrix.indexer.n_paradigms):
        diagonal = distance_matrix.distance_matrix[:, paradigm1_index, paradigm1_index]
        assert np.tril(diagonal).flatten().sum() == 0
        symmetric_distance_matrix[:, paradigm1_index, paradigm1_index] = (
            diagonal + diagonal.transpose(0, 2, 1)
        )
        added_flag_matrix[:, paradigm1_index, paradigm1_index] = True
        for paradigm2_index in range(
            paradigm1_index + 1, distance_matrix.indexer.n_paradigms
        ):
            first_paradigm_ix, second_paradigm_ix = paradigm1_index, paradigm2_index
            if (
                distance_matrix.distance_matrix[:, paradigm1_index, paradigm2_index]
                .flatten()
                .sum()
                == 0
            ):
                first_paradigm_ix, second_paradigm_ix = paradigm2_index, paradigm1_index
            assert (
                distance_matrix.distance_matrix[
                    :, second_paradigm_ix, first_paradigm_ix
                ]
                .flatten()
                .sum()
                == 0
            )
            symmetric_distance_matrix[:, first_paradigm_ix, second_paradigm_ix] = (
                distance_matrix.distance_matrix[
                    :, first_paradigm_ix, second_paradigm_ix
                ]
            )
            symmetric_distance_matrix[:, second_paradigm_ix, first_paradigm_ix] = (
                distance_matrix.distance_matrix[
                    :, first_paradigm_ix, second_paradigm_ix
                ].transpose(0, 2, 1)
            )
            added_flag_matrix[:, first_paradigm_ix, second_paradigm_ix] = True
            added_flag_matrix[:, second_paradigm_ix, first_paradigm_ix] = True
    assert added_flag_matrix.all()
    return symmetric_distance_matrix


def normalise_distances(
    unnormalised_symmetric_distance_matrix: np.ndarray, normalisation_method: str
) -> tuple[np.ndarray, np.ndarray]:
    normalisation_factors = np.zeros(unnormalised_symmetric_distance_matrix.shape[0])
    symmetric_distance_matrix = np.empty_like(unnormalised_symmetric_distance_matrix)
    for df_index in range(unnormalised_symmetric_distance_matrix.shape[0]):
        normalisation_factors[df_index] = get_normalisation_factor(
            unnormalised_symmetric_distance_matrix[df_index], normalisation_method
        )
        symmetric_distance_matrix[df_index] = (
            unnormalised_symmetric_distance_matrix[df_index]
            / normalisation_factors[df_index]
        )
    return symmetric_distance_matrix, normalisation_factors


def get_normalisation_factor(distances: np.ndarray, normalisation_method: str) -> float:
    if normalisation_method == "overall_mean":
        return distances.mean()
    if normalisation_method == "range":
        return distances.max() - distances.min()
    if normalisation_method.startswith("robust_range_"):
        percentile_excluded_from_range = float(normalisation_method.split("_")[-1])
        return np.percentile(
            distances, (100 - percentile_excluded_from_range / 2)
        ) - np.percentile(distances, percentile_excluded_from_range / 2)
    raise ValueError(f"Unknown normalisation method: {normalisation_method}")


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    normalisation_methods = [
        "overall_mean",
        "range",
        "robust_range_5",
        "robust_range_10",
    ]
    for normalisation_method_ in normalisation_methods:
        (
            aggregated_distance_,
            true_data_aggregated_distance_,
            normalisation_factors_,
            included_df_indices_,
        ) = calculate_aggregated_distances(
            logging_directory=parsed_args.logging_directory,
            run_id=parsed_args.run_id,
            max_modules=parsed_args.max_modules,
            simulation_replicates_per_paradigm=(
                parsed_args.simulations_per_paradigm * parsed_args.replicate_count
            ),
            distance_function_names=None,
            protection_selection_paradigms=False,
            simulation_subset_index=None,
            include_true_data=True,
            vif_threshold=10,
            normalisation_method=normalisation_method_,
            verbose=True,
        )
        print("aggregated distance shape: ", aggregated_distance_.shape)
        print(
            "true data aggregated distance shape: ",
            true_data_aggregated_distance_.shape,
        )
        print("normalisation factors: ", normalisation_factors_)
        print("included df indices: ", included_df_indices_)
