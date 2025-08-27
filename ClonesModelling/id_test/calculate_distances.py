from itertools import repeat
from multiprocessing import Pool
import os
import time
from typing import Callable, Generator
import numpy as np
import pandas as pd  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore

from .read_data import (
    IdentifiabilityDataset,
    SimilarityLevel,
    ReplicateData,
    PatientReplicateData,
)

from ..data.mutations_data import get_total_mutations_data_per_patient
from ..parse_cmd_line_args import parse_id_test_comparison_arguments
from ..distance import two_dim_wasserstein_distance


# Group so each group takes approx 1h to run, separated by whether slow 2DWD is included
SAMPLES_PER_GROUP: dict[bool, dict[SimilarityLevel, int]] = {
    True: {
        SimilarityLevel.REPLICATE: 327,
        SimilarityLevel.INTRA_PARADIGM: 55,
        SimilarityLevel.INTER_PARADIGM: 71,
    },
    False: {
        SimilarityLevel.REPLICATE: 30000,
        SimilarityLevel.INTRA_PARADIGM: 30000,
        SimilarityLevel.INTER_PARADIGM: 30000,
    },
}


DISTANCE_FUNCTIONS: dict[
    str, Callable[[PatientReplicateData, PatientReplicateData], float]
] = {
    # tuple is (mutational burden (incl sm sig), phylogeny, j1)
    ## Mutational burden functions
    "wasserstein": lambda x, y: wasserstein_distance(
        x.mutational_burden[:, 0], y.mutational_burden[:, 0]
    ),
    "2D_wasserstein": lambda x, y: two_dim_wasserstein_distance(
        x.mutational_burden, y.mutational_burden
    ),
    "2D_wasserstein_simplified": lambda x, y: two_dim_wasserstein_distance(
        x.mutational_burden, y.mutational_burden, simplified=True
    ),
    "smoking_sig_only": lambda x, y: wasserstein_distance(
        x.mutational_burden[:, 1], y.mutational_burden[:, 1]
    ),
    "z_values": lambda x, y: wasserstein_distance(
        np.abs(x.mutational_burden[:, 0] - np.mean(x.mutational_burden[:, 0]))
        / np.std(x.mutational_burden[:, 0]),
        np.abs(y.mutational_burden[:, 0] - np.mean(y.mutational_burden[:, 0]))
        / np.std(y.mutational_burden[:, 0]),
    ),
    "mean_subtracted": lambda x, y: wasserstein_distance(
        x.mutational_burden[:, 0] - np.mean(x.mutational_burden[:, 0]),
        y.mutational_burden[:, 0] - np.mean(y.mutational_burden[:, 0]),
    ),
    "mean_subtracted_2D_simplified": lambda x, y: two_dim_wasserstein_distance(
        x.mutational_burden - np.mean(x.mutational_burden, axis=0),
        y.mutational_burden - np.mean(y.mutational_burden, axis=0),
        simplified=True,
    ),
    # "mm_larger_weight_sq_diff": lambda x, y: (
    #     x.mixture_model_data.larger_weight - y.mixture_model_data.larger_weight
    # )
    # ** 2,
    "mm_larger_weight_abs_diff": lambda x, y: np.abs(
        x.mixture_model_data.larger_weight - y.mixture_model_data.larger_weight
    ),
    "mm_dominant_means_sq_diff": lambda x, y: (
        x.mixture_model_data.dominant_mean - y.mixture_model_data.dominant_mean
    )
    ** 2,
    "mm_larger_means_sq_diff": lambda x, y: (
        x.mixture_model_data.larger_mean - y.mixture_model_data.larger_mean
    )
    ** 2,
    "mm_smaller_means_sq_diff": lambda x, y: (
        x.mixture_model_data.smaller_mean - y.mixture_model_data.smaller_mean
    )
    ** 2,
    "mm_weighted_means_by_dominance": lambda x, y: (
        ((x.mixture_model_data.dominant_mean - y.mixture_model_data.dominant_mean) ** 2)
        * x.mixture_model_data.larger_weight
        * y.mixture_model_data.larger_weight
        + (
            (x.mixture_model_data.other_mean or x.mixture_model_data.dominant_mean)
            - (y.mixture_model_data.other_mean or y.mixture_model_data.dominant_mean)
        )
        ** 2
        * ((1 - x.mixture_model_data.larger_weight) or 1)
        * ((1 - y.mixture_model_data.larger_weight) or 1)
    ),
    "mm_weighted_means_by_position": lambda x, y: (
        (x.mixture_model_data.larger_mean - y.mixture_model_data.larger_mean) ** 2
        * x.mixture_model_data.larger_mean_weight
        * y.mixture_model_data.larger_mean_weight
        + (x.mixture_model_data.smaller_mean - y.mixture_model_data.smaller_mean) ** 2
        * x.mixture_model_data.smaller_mean_weight
        * y.mixture_model_data.smaller_mean_weight
    ),
    ## Phylogeny functions
    # "total_branch_length": lambda x, y: np.abs(
    #     x.branch_lengths.sum() - y.branch_lengths.sum()
    # ),
    "branch_length_wasserstein": lambda x, y: wasserstein_distance(
        x.branch_lengths, y.branch_lengths
    ),
    "abs_j_one": lambda x, y: np.abs(x.j_one - y.j_one),
    # "l2_j_one": lambda x, y: (x.j_one - y.j_one) ** 2,
    ## Control functions
    "zero_control": lambda x, y: 0,
    "random_control": lambda x, y: np.random.rand(),
}


def calculate_distances(
    logging_directory: str,
    run_id: str,
    similarity_level: SimilarityLevel,
    n_pairs_to_sample: int | None,
    include_2d_wasserstein: bool,
    group_number: int,
) -> None:
    on_cluster = ("cluster" in logging_directory) or ("SAN" in logging_directory)
    start_time = time.time()
    data = IdentifiabilityDataset(logging_directory, run_id)
    patient_weights = get_patient_weights()
    if n_pairs_to_sample is not None:
        distances_length = n_pairs_to_sample
        pairs_iterator: (
            list[tuple[int, int]] | Generator[tuple[int, int], None, None]
        ) = (
            data.replicate_indexer.randomly_sample_pair(similarity_level)
            for _ in range(n_pairs_to_sample)
        )
    else:
        pairs_iterator = data.replicate_indexer.get_group_pairs(
            similarity_level,
            group_number,
            SAMPLES_PER_GROUP[include_2d_wasserstein][similarity_level],
        )
        distances_length = len(pairs_iterator)

    patient_list = list(data.patient_list)
    distances = {
        function_name: pd.DataFrame(
            np.nan,
            index=range(distances_length),
            columns=["replicate_1", "replicate_2", "distance"] + patient_list,
        )
        for function_name in DISTANCE_FUNCTIONS
        if not (function_name == "2D_wasserstein" and not include_2d_wasserstein)
        and not is_already_completed(
            logging_directory,
            run_id,
            function_name,
            similarity_level,
            group_number,
            distances_length,
        )
    }
    if len(distances) == 0:
        print(
            f"Group {group_number} of {similarity_level.name.lower()} distances already calculated"
        )
        return
    print(
        f"Calculating {distances_length} {similarity_level.name.lower()} distances "
        f"for group {group_number}, distance functions {list(distances.keys())}"
    )
    if on_cluster:
        print("-" * 80)
    for sample_index, (replicate_index_1, replicate_index_2) in enumerate(
        pairs_iterator
    ):
        if on_cluster and (sample_index % (distances_length // 80) == 0):
            print(".", end="", flush=True)
        replicate_1 = data.get_replicate(replicate_index_1)
        replicate_2 = data.get_replicate(replicate_index_2)
        for function_name, distances_df in distances.items():
            distance, patient_distances = calculate_distance(
                replicate_1,
                replicate_2,
                patient_weights,
                function_name,
            )
            distances_df.loc[sample_index] = [
                replicate_index_1,
                replicate_index_2,
                distance,
            ] + [patient_distances.get(patient, np.nan) for patient in patient_list]
    for function_name, distances_df in distances.items():
        output_directory = (
            f"{logging_directory}/{run_id}/distance/{function_name}/"
            f"{similarity_level.name.lower()}/"
        )
        os.makedirs(output_directory, exist_ok=True)
        distances_df.to_csv(f"{output_directory}/group_{group_number}.csv", index=False)
    print(
        f"\nGroup {group_number} of {similarity_level.name.lower()} distances calculated "
        f"in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}"
    )


def get_patient_weights():
    return {
        patient: np.log(1 + len(patient_data))
        for patient, patient_data in get_total_mutations_data_per_patient(True).items()
    }


def is_already_completed(
    logging_directory: str,
    run_id: str,
    function_name: str,
    similarity_level: SimilarityLevel,
    group_number: int,
    distances_length: int,
) -> bool:
    if os.path.exists(
        f"{logging_directory}/{run_id}/distance/{function_name}/"
        f"pairwise_distances.csv"
    ):
        group_numbers_in_csv = pd.read_csv(
            f"{logging_directory}/{run_id}/distance/{function_name}/pairwise_distances.csv",
            usecols=["group_index", "similarity_level"],
        ).loc[lambda x: x["similarity_level"] == similarity_level.name]["group_index"]
        if sum(group_numbers_in_csv == group_number) == distances_length:
            return True
    if os.path.exists(
        f"{logging_directory}/{run_id}/distance/{function_name}/"
        f"{similarity_level.name.lower()}/group_{group_number}.csv"
    ):
        # check it's the right length
        with open(
            f"{logging_directory}/{run_id}/distance/{function_name}/"
            f"{similarity_level.name.lower()}/group_{group_number}.csv",
            "r",
            encoding="utf-8",
        ) as f:
            return len(f.readlines()) == distances_length + 1
    return False


def calculate_distance(
    replicate_1: ReplicateData,
    replicate_2: ReplicateData,
    patient_weights: dict[str, float],
    function_name: str,
) -> tuple[float, dict[str, float]]:
    patient_distances = {}
    for patient in set(replicate_1.patient_mutational_burden.keys()).intersection(
        replicate_2.patient_mutational_burden.keys()
    ):
        try:
            patient_distances[patient] = DISTANCE_FUNCTIONS[function_name](
                replicate_1[patient], replicate_2[patient]
            )
        except ValueError:  # tried to access eg J1 when it doesn't exist
            pass
    if any(patient in patient_weights for patient in patient_distances):
        assert all(patient in patient_weights for patient in patient_distances)
        distance = sum(
            patient_distances[patient] * patient_weights[patient]
            for patient in patient_distances
        )
    else:
        distance = sum(patient_distances.values())
    return float(distance), patient_distances


def get_groups_per_similarity_level(
    n_pairs_to_sample: int | None, logging_directory: str, run_id: str
) -> dict[SimilarityLevel, dict[bool, int]]:
    if n_pairs_to_sample is None:
        replicate_indexer = IdentifiabilityDataset(
            logging_directory, run_id
        ).replicate_indexer
        print(f"Using replicate indexer to count groups:\n\t{str(replicate_indexer)}")
        total_pairs = {
            similarity_level: replicate_indexer.count_pairs(similarity_level)
            for similarity_level in SimilarityLevel
        }
    else:
        total_pairs = {
            similarity_level: n_pairs_to_sample for similarity_level in SimilarityLevel
        }
    return {
        similarity_level: {
            include_2d_wasserstein: int(
                np.ceil(
                    total_pairs[similarity_level]
                    / SAMPLES_PER_GROUP[include_2d_wasserstein][similarity_level]
                )
            )
            for include_2d_wasserstein in [True, False]
        }
        for similarity_level in SimilarityLevel
    }


if __name__ == "__main__":
    parsed_args = parse_id_test_comparison_arguments()
    calculate_distances(
        parsed_args.logging_directory,
        parsed_args.run_id,
        SimilarityLevel(parsed_args.similarity_level),
        parsed_args.n_pairs_to_sample,
        parsed_args.compare_smoking_signature_mutations,
        parsed_args.group_number,
    )
