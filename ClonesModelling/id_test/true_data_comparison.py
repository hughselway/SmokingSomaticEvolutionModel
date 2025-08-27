import math
import os
import time
from multiprocessing import Pool
from typing import Callable
import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt

from .calculate_distances import (
    DISTANCE_FUNCTIONS,
    get_patient_weights,
    calculate_distance,
)
from .read_data import IdentifiabilityDataset, ReplicateData

from ..data.mixture_model_data import MixtureModelData, get_mixture_model
from ..data.mutations_data import get_total_mutations_data_per_patient, get_clones_data
from ..data.tree_data import get_patient_tree_balance, get_patient_branch_lengths
from ..parse_cmd_line_args import parse_id_test_arguments


def calculate_true_distances(
    logging_directory: str,
    run_id: str,
    include_2d_wasserstein: bool,
    number_of_processes: int,
    replicates_per_batch: int = 50,
) -> None:
    start_time = time.time()
    data = IdentifiabilityDataset(logging_directory, run_id)
    patient_weights = get_patient_weights()
    true_data = get_true_data()

    with Pool(number_of_processes) as pool:
        if check_distances_calculated(
            logging_directory, run_id, include_2d_wasserstein, data
        ):
            print("Distances already calculated.")
            return
        pool.starmap(
            calculate_distance_and_save,
            (
                (
                    batch_index,
                    replicates_per_batch,
                    data,
                    true_data,
                    patient_weights,
                    logging_directory,
                    run_id,
                    include_2d_wasserstein,
                )
                for batch_index in range(
                    math.ceil(len(data.replicate_indexer) / replicates_per_batch) + 1
                )
            ),
        )
    for top_n in [None, 10, 30, 100, 300]:
        plot_true_distances(
            data, logging_directory, run_id, include_2d_wasserstein, top_n
        )
    print(
        "Time taken:", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    )


def check_distances_calculated(
    logging_directory: str,
    run_id: str,
    include_2d_wasserstein: bool,
    data: IdentifiabilityDataset,
):
    distances_already_calculated = True
    for distance_function_name in DISTANCE_FUNCTIONS:
        if distance_function_name == "2D_wasserstein" and not include_2d_wasserstein:
            continue
        if not os.path.exists(
            f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
            "true_data_distances.csv"
        ):
            distances_already_calculated = False
            break
        if pd.read_csv(
            f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
            "true_data_distances.csv"
        ).shape[0] != len(data.replicate_indexer):
            distances_already_calculated = False
            break
    return distances_already_calculated


def calculate_distance_and_save(
    batch_index: int,
    replicates_per_batch: int,
    data: IdentifiabilityDataset,
    true_data: ReplicateData,
    patient_weights: dict[str, float],
    logging_directory: str,
    run_id: str,
    include_2d_wasserstein: bool,
) -> None:
    batch_distances: dict[str, list[tuple[int, float, dict[str, float]]]] = {
        df_name: [] for df_name in DISTANCE_FUNCTIONS
    }
    patient_subsets = get_patient_subsets()
    for replicate_index in range(
        batch_index * replicates_per_batch,
        min((batch_index + 1) * replicates_per_batch, len(data.replicate_indexer)),
    ):
        simulated_data = data.get_replicate(replicate_index)
        for distance_function_name in DISTANCE_FUNCTIONS:
            if (
                not include_2d_wasserstein
                and distance_function_name == "2D_wasserstein"
            ):
                continue
            distance, patient_distances = calculate_distance(
                simulated_data, true_data, patient_weights, distance_function_name
            )
            batch_distances[distance_function_name].append(
                (
                    replicate_index,
                    distance,
                    {
                        subset_name: sum(
                            dist * patient_weights[pt]
                            for pt, dist in patient_distances.items()
                            if is_included(pt)
                        )
                        for subset_name, is_included in patient_subsets
                    },
                )
            )
    for distance_function_name in DISTANCE_FUNCTIONS:
        if distance_function_name == "2D_wasserstein" and not include_2d_wasserstein:
            continue
        os.makedirs(
            f"{logging_directory}/{run_id}/distance/{distance_function_name}",
            exist_ok=True,
        )
        pd.DataFrame(
            {
                "replicate_index": [
                    replicate_index
                    for replicate_index, _, _ in batch_distances[distance_function_name]
                ],
                "distance": [
                    distance
                    for _, distance, _ in batch_distances[distance_function_name]
                ],
                **{
                    name: [
                        subset_distance[name]
                        for _, _, subset_distance in batch_distances[
                            distance_function_name
                        ]
                    ]
                    for name, _ in patient_subsets
                },
            }
        ).to_csv(
            f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
            f"true_data_distances.csv",
            mode="a",
            header=not os.path.exists(
                f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
                "true_data_distances.csv"
            ),
            index=False,
        )


def get_true_data(clones_data_only: bool = False) -> ReplicateData:
    clones_patients = get_clones_data()["patient"].unique()
    mutational_burden = {
        patient: mb
        for patient, mb in get_total_mutations_data_per_patient(True).items()
        if (not clones_data_only) or patient in clones_patients
    }
    branch_lengths: dict[str, np.ndarray | None] = {}
    tree_balances: dict[str, float | None] = {}
    mixture_models: dict[str, MixtureModelData] = {
        patient: get_mixture_model(patient) for patient in mutational_burden
    }
    for patient in mutational_burden:
        try:
            branch_lengths[patient] = get_patient_branch_lengths(patient)
            tree_balances[patient] = get_patient_tree_balance(patient)
        except FileNotFoundError:
            branch_lengths[patient] = None
            tree_balances[patient] = None
    return ReplicateData(
        mutational_burden, branch_lengths, tree_balances, mixture_models
    )


def plot_true_distances(
    data: IdentifiabilityDataset,
    logging_directory: str,
    run_id: str,
    include_2d_wasserstein: bool,
    top_n: int | None,
) -> None:
    ncol = 5
    df_count = len(DISTANCE_FUNCTIONS)
    if not include_2d_wasserstein:
        df_count -= 1
    nrow = df_count // ncol + (df_count % ncol > 0)
    combined_fig, combined_ax = plt.subplots(nrow, ncol, figsize=(ncol * 5, nrow * 5))
    for df_index, distance_function in enumerate(
        df
        for df in DISTANCE_FUNCTIONS
        if include_2d_wasserstein or df != "2D_wasserstein"
    ):
        distances_df = pd.read_csv(
            f"{logging_directory}/{run_id}/distance/{distance_function}/true_data_distances.csv"
        ).assign(
            paradigm=lambda x: x["replicate_index"].apply(
                lambda index: data.replicate_indexer.paradigm_from_index(index)[0]
            )
        )
        if top_n is not None:
            distances_df = (
                distances_df.groupby("paradigm")
                .apply(lambda x: x.nsmallest(top_n, "distance"))
                .reset_index(drop=True)
            )
        fig, ax = plt.subplots()

        distances_df.boxplot(
            "distance",
            by="paradigm",
            ax=ax,
            showfliers=False,
            rot=90,
            grid=False,
        )
        ax.set_ylabel("distance")
        ax.set_xlabel("paradigm")

        os.makedirs(
            f"{logging_directory}/{run_id}/distance/{distance_function}/plot",
            exist_ok=True,
        )
        fig.tight_layout()
        fig.savefig(
            f"{logging_directory}/{run_id}/distance/{distance_function}/plot/"
            f"true_data_distances_by_paradigm{f'_top_{top_n}' if top_n is not None else ''}.png"
        )
        plt.close(fig)
        distances_df.boxplot(
            "distance",
            by="paradigm",
            ax=combined_ax[df_index // ncol, df_index % ncol],
            showfliers=False,
            rot=90,
            grid=False,
        )
        combined_ax[df_index // ncol, df_index % ncol].set_title(distance_function)
    combined_fig.tight_layout()
    combined_fig.savefig(
        f"{logging_directory}/{run_id}/distance/true_data_distances_combined"
        f"{f'_top_{top_n}' if top_n is not None else ''}.png"
    )
    plt.close(combined_fig)


def get_patient_subsets() -> list[tuple[str, Callable]]:
    pt_statuses = (
        pd.read_csv("ClonesModelling/data/patient_data/smoking_records.csv")
        .set_index("patient")[["smoking_status"]]
        .to_dict()["smoking_status"]
    )
    return [
        ("nature_patients", lambda pt, pt_statuses_=pt_statuses: pt.startswith("PD")),
        (
            "nature_genetics_patients",
            lambda pt, pt_statuses_=pt_statuses: not pt.startswith("PD"),
        ),
        (
            "status_representatives",
            lambda pt, pt_statuses_=pt_statuses: pt
            in ["PD26988", "PD34204", "PD34209"],
        ),
        (
            "smokers_only",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] == "smoker",
        ),
        (
            "non_smokers_only",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] == "non-smoker",
        ),
        (
            "ex_smokers_only",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] == "ex-smoker",
        ),
        (
            "without_smokers",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] != "smoker",
        ),
        (
            "without_non_smokers",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] != "non-smoker",
        ),
        (
            "without_ex_smokers",
            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt] != "ex-smoker",
        ),
        ("total", lambda _, pt_statuses_=pt_statuses: True),
    ]


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    calculate_true_distances(
        parsed_args.logging_directory,
        parsed_args.run_id,
        parsed_args.compare_smoking_signature_mutations,
        parsed_args.number_of_processes,
    )
