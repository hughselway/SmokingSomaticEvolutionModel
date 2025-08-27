import os
import pandas as pd  # type: ignore

from .calculate_distances import get_groups_per_similarity_level, DISTANCE_FUNCTIONS
from .read_data import SimilarityLevel, IdentifiabilityDataset


def process_distance_comparison_outputs(
    logging_directory: str,
    run_id: str,
    n_pairs_to_sample: int | None,
    include_2d_wasserstein: bool,
) -> list[str]:
    groups_per_similarity_level = get_groups_per_similarity_level(
        n_pairs_to_sample, logging_directory, run_id
    )
    distance_functions = [
        function_name
        for function_name in DISTANCE_FUNCTIONS
        if function_name in os.listdir(f"{logging_directory}/{run_id}/distance")
        and len(os.listdir(f"{logging_directory}/{run_id}/distance/{function_name}"))
        > 0
    ]
    for function_name in distance_functions:
        combine_id_test_datasets(
            logging_directory,
            run_id,
            groups_per_similarity_level,
            function_name,
            include_2d_wasserstein,
        )
    for function_name in distance_functions:
        calculate_paradigm_distances(logging_directory, run_id, function_name)
    return distance_functions


def combine_id_test_datasets(
    logging_directory: str,
    run_id: str,
    groups_per_similarity_level: dict[SimilarityLevel, dict[bool, int]],
    function_name: str,
    include_2d_wasserstein: bool,
):
    distance_function_dir = f"{logging_directory}/{run_id}/distance/{function_name}"
    csv_filename = f"{distance_function_dir}/pairwise_distances.csv"
    print(f"Combining datasets for {function_name} into {csv_filename}")
    first_file = not os.path.exists(csv_filename)
    for similarity_level in SimilarityLevel:
        if groups_per_similarity_level[similarity_level][include_2d_wasserstein] == 0:
            continue
        print(f"Processing {similarity_level.name}")
        for group_index in range(
            1,
            groups_per_similarity_level[similarity_level][include_2d_wasserstein] + 1,
        ):
            output_file = (
                f"{distance_function_dir}/{similarity_level.name.lower()}/"
                f"group_{group_index}.csv"
            )
            if not os.path.exists(output_file):
                if first_file:
                    print(f"\nOutput file {output_file} does not exist")
                    break
                continue
            this_df = pd.read_csv(output_file).assign(
                similarity_level=similarity_level.name, group_index=group_index
            )
            if this_df.shape[1] > 5:  # then we've recorded patient distances
                # reorder columns to allow for appending to csv
                this_df = this_df[
                    [
                        "replicate_1",
                        "replicate_2",
                        "distance",
                        "similarity_level",
                        "group_index",
                    ]
                    + sorted(
                        [
                            col
                            for col in this_df.columns
                            if col
                            not in [
                                "replicate_1",
                                "replicate_2",
                                "distance",
                                "similarity_level",
                                "group_index",
                            ]
                        ]
                    )
                ]
            this_df.to_csv(csv_filename, index=False, mode="a", header=first_file)
            first_file = False
            os.remove(output_file)
        if first_file:
            print(f"No files exist for {similarity_level.name}")
            continue
        print(" done")
        if os.path.isdir(f"{distance_function_dir}/{similarity_level.name.lower()}"):
            os.rmdir(f"{distance_function_dir}/{similarity_level.name.lower()}")


def calculate_paradigm_distances(
    logging_directory: str,
    run_id: str,
    function_name: str,
) -> None:
    distance_dir = f"{logging_directory}/{run_id}/distance/{function_name}/"
    if os.path.exists(f"{distance_dir}/paradigm_pairwise_distances.csv"):
        print(
            f"Output file {distance_dir}/paradigm_pairwise_distances.csv already exists"
        )
        return
    print(f"Calculating paradigm distances for {function_name}...", end="", flush=True)
    # TODO optimise for speed?
    paradigm_distances = (
        annotate_paradigm_names_two_replicates(
            pd.read_csv(
                f"{distance_dir}/pairwise_distances.csv",
                usecols=[
                    "replicate_1",
                    "replicate_2",
                    "distance",
                    "similarity_level",
                    "group_index",
                ],
            ),
            logging_directory,
            run_id,
        )
        .loc[
            lambda df: df.similarity_level != "REPLICATE",
            ["paradigm_1", "paradigm_2", "distance"],
        ]
        .groupby(["paradigm_1", "paradigm_2"])
        .agg({"distance": ["size", "mean", "std"]})
        .reset_index()
    )
    paradigm_distances.columns = [
        "_".join(col).strip() if col[1] else col[0]
        for col in paradigm_distances.columns.values
    ]
    paradigm_distances = paradigm_distances.rename(columns={"distance_size": "n_pairs"})
    paradigm_distances.to_csv(
        f"{distance_dir}/paradigm_pairwise_distances.csv", index=False
    )
    print("done")


def annotate_paradigm_names_two_replicates(
    combined_df: pd.DataFrame, logging_directory: str, run_id: str
) -> pd.DataFrame:
    replicate_indexer = IdentifiabilityDataset(
        logging_directory, run_id
    ).replicate_indexer
    all_index_paradigms = [
        replicate_indexer.paradigms.index(
            replicate_indexer.paradigm_from_index(index)[0]
        )
        for index in range(int(combined_df.replicate_2.max()) + 1)
    ]
    print("paradigms found")
    combined_df = (
        combined_df.assign(
            paradigm_1_index=lambda df: df.replicate_1.astype(int).apply(
                lambda x: all_index_paradigms[x]
            ),
            paradigm_2_index=lambda df: df.replicate_2.astype(int).apply(
                lambda x: all_index_paradigms[x]
            ),
            paradigm_1_index_reordered=lambda df: df.apply(
                lambda row: min(row.paradigm_1_index, row.paradigm_2_index), axis=1
            ),
            paradigm_2_index_reordered=lambda df: df.apply(
                lambda row: max(row.paradigm_1_index, row.paradigm_2_index), axis=1
            ),
            paradigm_1=lambda df: df.paradigm_1_index_reordered.apply(
                lambda x: replicate_indexer.paradigms[int(x)]
            ),
            paradigm_2=lambda df: df.paradigm_2_index_reordered.apply(
                lambda x: replicate_indexer.paradigms[int(x)]
            ),
            # reorder the replicate columns as well if the paradigm cols have been swapped
            replicate_1_reordered=lambda df: df.apply(
                lambda row: (
                    row.replicate_1
                    if row.paradigm_1_index_reordered == row.paradigm_1_index
                    else row.replicate_2
                ),
                axis=1,
            ),
            replicate_2_reordered=lambda df: df.apply(
                lambda row: (
                    row.replicate_2
                    if row.paradigm_2_index_reordered == row.paradigm_2_index
                    else row.replicate_1
                ),
                axis=1,
            ),
            replicate_1=lambda df: df.replicate_1_reordered,
            replicate_2=lambda df: df.replicate_2_reordered,
        )
        .sort_values(by=["paradigm_1_index_reordered", "paradigm_2_index_reordered"])
        .drop(
            columns=[
                "paradigm_1_index",
                "paradigm_2_index",
                "paradigm_1_index_reordered",
                "paradigm_2_index_reordered",
                "replicate_1_reordered",
                "replicate_2_reordered",
            ]
        )
    )
    return combined_df


def annotate_paradigm_names(
    distances_df: pd.DataFrame, logging_directory: str, run_id: str
) -> pd.DataFrame:
    replicate_indexer = IdentifiabilityDataset(
        logging_directory, run_id
    ).replicate_indexer
    return distances_df.assign(
        paradigm_index=lambda df: df.replicate_index.apply(
            lambda x: replicate_indexer.paradigms.index(
                replicate_indexer.paradigm_from_index(x)[0]
            )
        ),
        paradigm=lambda df: df.paradigm_index.apply(
            lambda x: replicate_indexer.paradigms[int(x)]
        ),
    ).drop(columns=["paradigm_index"])
