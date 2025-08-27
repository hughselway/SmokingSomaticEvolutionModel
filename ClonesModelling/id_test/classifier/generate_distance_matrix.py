from dataclasses import dataclass
import json
import os
import numpy as np
import matplotlib.pyplot as plt

from .indexer import Indexer

from ..calculate_distances import DISTANCE_FUNCTIONS

from ...parameters.hypothetical_paradigm_class import is_protection_selection_subset


@dataclass
class DistanceMatrix:
    """
    Distance matrix, of shape (n_distance_functions, n_paradigms, n_paradigms,
    simulation_replicates_per_paradigm, simulation_replicates_per_paradigm).
    Symmetric in 1,2 and 3,4 axes, with diagonal 0s
    true_data_distance has shape (n_distance_functions, n_paradigms,
    simulation_replicates_per_paradigm).
    Can be run without true distances.
    """

    distance_matrix: np.ndarray
    true_data_distance: np.ndarray | None
    indexer: Indexer

    # initialise empty
    def __init__(
        self,
        indexer: Indexer,
        distance_matrix: np.ndarray | None = None,
        true_data_distance: np.ndarray | None = None,
    ):
        self.indexer = indexer
        if distance_matrix is None:
            self.distance_matrix = np.empty(indexer.get_distance_matrix_shape())
            if self.indexer.include_true_data:
                self.true_data_distance = np.empty(
                    indexer.get_true_data_distance_shape()
                )
            else:
                self.true_data_distance = None
        else:
            self.distance_matrix = distance_matrix
            self.true_data_distance = true_data_distance
            assert distance_matrix.shape == indexer.get_distance_matrix_shape(), (
                f"Shape is {distance_matrix.shape}; "
                f"expected {indexer.get_distance_matrix_shape()}"
            )
            assert (
                true_data_distance is None
                or true_data_distance.shape == indexer.get_true_data_distance_shape()
            ), (
                f"Shape is {true_data_distance.shape}; "
                f"expected {indexer.get_true_data_distance_shape()}"
            )

    def add_values(self, distance_function_name: str, values: np.ndarray) -> None:
        assert values.shape == (
            self.indexer.get_distance_matrix_shape()[1:]
        ), f"Shape is {values.shape}, expected {self.indexer.get_distance_matrix_shape()[1:]}"
        distance_function_index = self.indexer.distance_function_names.index(
            distance_function_name
        )
        self.distance_matrix[distance_function_index] = values

    def add_true_data_distances(
        self, distance_function_name: str, values: np.ndarray
    ) -> None:
        assert self.true_data_distance is not None
        assert values.shape == (
            self.indexer.get_true_data_distance_shape()[1:]
        ), f"Shape is {values.shape}, expected {self.indexer.get_true_data_distance_shape()[1:]}"
        distance_function_index = self.indexer.distance_function_names.index(
            distance_function_name
        )
        self.true_data_distance[distance_function_index] = values

    def write(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/distance_matrix.npy", self.distance_matrix)
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            np.save(f"{save_dir}/true_data_distance.npy", self.true_data_distance)
        with open(f"{save_dir}/indexer.json", "w", encoding="utf-8") as f:
            json.dump(self.indexer.json_dict(), f)

    def get_paradigm_agnostic_distances(
        self, distance_function_index: int
    ) -> np.ndarray:
        """
        Get the distances in the paradigm-agnostic format: a 2D array of shape
        (n_datapoints, n_datapoints). (if include_true_data, the last row and column
        are the true data distances and n_datapoints includes the true data)
        """
        paradigm_agnostic_distances = (
            (
                self.distance_matrix[distance_function_index]
                + self.distance_matrix[distance_function_index].transpose(1, 0, 3, 2)
            )
            .transpose(0, 2, 1, 3)
            .reshape(
                self.indexer.n_datapoints_excluding_true_data,
                self.indexer.n_datapoints_excluding_true_data,
            )
        )
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            true_data_distances = self.true_data_distance[distance_function_index]
            assert true_data_distances.shape == (
                self.indexer.n_paradigms,
                self.indexer.simulation_replicates_per_paradigm,
            ), f"Shape is {true_data_distances.shape}"
            paradigm_agnostic_distances = np.pad(paradigm_agnostic_distances, ((0, 1),))
            paradigm_agnostic_distances[:-1, -1] = true_data_distances.reshape(-1)
            paradigm_agnostic_distances[-1, :-1] = true_data_distances.reshape(-1)
        assert paradigm_agnostic_distances.shape == (
            self.indexer.n_datapoints,
            self.indexer.n_datapoints,
        ), f"Shape is {paradigm_agnostic_distances.shape}"
        assert isinstance(
            paradigm_agnostic_distances, np.ndarray
        ), f"Type is {type(paradigm_agnostic_distances)}"

        # test that the reshaping is correct
        paradigm_i, paradigm_j, datapoint_i, datapoint_j = 1, 2, 3, 4
        distance_from_paradigm_agnostic = paradigm_agnostic_distances[
            paradigm_i * self.indexer.simulation_replicates_per_paradigm + datapoint_i,
            paradigm_j * self.indexer.simulation_replicates_per_paradigm + datapoint_j,
        ]
        distance_from_matrix = self.distance_matrix[
            distance_function_index,
            paradigm_i,
            paradigm_j,
            datapoint_i,
            datapoint_j,
        ]
        assert np.isclose(distance_from_paradigm_agnostic, distance_from_matrix), (
            f"Incorrect reshaping: {distance_from_paradigm_agnostic} != "
            f"{distance_from_matrix}; {paradigm_i}, {paradigm_j}, "
            f"{datapoint_i}, {datapoint_j}"
        )
        assert np.allclose(
            paradigm_agnostic_distances, paradigm_agnostic_distances.T
        ), "Dissimilarity matrix is not symmetric"

        return paradigm_agnostic_distances

    def get_normalised_paradigm_agnostic_distances(
        self, distance_function_index: int
    ) -> np.ndarray:
        paradigm_agnostic_distances = self.get_paradigm_agnostic_distances(
            distance_function_index
        )
        if paradigm_agnostic_distances.mean() == 0:
            raise ValueError(
                "Mean of paradigm-agnostic distances is zero; distance_function = "
                + self.indexer.distance_function_names[distance_function_index]
            )
        normalised_distances = (
            paradigm_agnostic_distances / paradigm_agnostic_distances.mean()
        )
        assert isinstance(
            normalised_distances, np.ndarray
        ), f"Type is {type(normalised_distances)}"
        assert normalised_distances.shape == (
            self.indexer.n_datapoints,
            self.indexer.n_datapoints,
        ), f"Shape is {normalised_distances.shape}"
        assert np.allclose(
            normalised_distances, normalised_distances.T
        ), "Normalised dissimilarity matrix is not symmetric"
        return normalised_distances

    def subset_distance_functions(self, indexer: Indexer) -> "DistanceMatrix":
        assert indexer.is_df_subset(
            self.indexer
        ), f"read:\n{indexer}\nmade\n{self.indexer}"
        df_indices = [
            self.indexer.distance_function_names.index(df_name)
            for df_name in indexer.distance_function_names
        ]
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            return DistanceMatrix(
                indexer,
                self.distance_matrix[df_indices],
                self.true_data_distance[df_indices],
            )
        return DistanceMatrix(indexer, self.distance_matrix[df_indices])

    def subset_to_n_module_paradigms(self, n: int) -> "DistanceMatrix":
        assert n > 0 or n == -1
        paradigm_indices = (
            [
                i
                for i, paradigm in enumerate(self.indexer.hypothetical_paradigms)
                if len(paradigm.get_module_names(include_base=False)) <= n
            ]
            if n > 0
            else [
                i
                for i, paradigm in enumerate(self.indexer.hypothetical_paradigms)
                if is_protection_selection_subset(
                    paradigm.get_module_names(include_base=True)
                )
            ]
        )
        indexer = Indexer(
            "2D_wasserstein" in self.indexer.distance_function_names,
            max_modules=n,
            simulation_replicates_per_paradigm=self.indexer.simulation_replicates_per_paradigm,
            include_true_data=self.indexer.include_true_data,
            distance_function_names=self.indexer.distance_function_names,
        )
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            return DistanceMatrix(
                indexer,
                self.distance_matrix[:, paradigm_indices, :, :, :][
                    :, :, paradigm_indices, :, :
                ],
                self.true_data_distance[:, paradigm_indices, :],
            )
        return DistanceMatrix(
            indexer,
            self.distance_matrix[:, paradigm_indices, :, :, :][
                :, :, paradigm_indices, :, :
            ],
        )

    def subset_to_first_replicates(self, n_replicates: int) -> "DistanceMatrix":
        assert self.indexer.simulation_replicates_per_paradigm % n_replicates == 0
        n_simulations_per_paradigm = (
            self.indexer.simulation_replicates_per_paradigm // n_replicates
        )
        assert (
            self.distance_matrix.shape[3]
            == self.distance_matrix.shape[4]
            == n_simulations_per_paradigm * n_replicates
        )
        indexer = Indexer(
            "2D_wasserstein" in self.indexer.distance_function_names,
            self.indexer.max_modules,
            simulation_replicates_per_paradigm=n_simulations_per_paradigm,
            include_true_data=self.indexer.include_true_data,
            distance_function_names=self.indexer.distance_function_names,
        )
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            return DistanceMatrix(
                indexer,
                self.distance_matrix[:, :, :, ::n_replicates, ::n_replicates],
                self.true_data_distance[:, :, ::n_replicates],
            )
        return DistanceMatrix(
            indexer,
            self.distance_matrix[:, :, :, ::n_replicates, ::n_replicates],
        )

    def subset_to_simulation_subset(
        self, logging_directory: str, run_id: str, subset_index: int
    ) -> "DistanceMatrix":
        with np.load(
            f"{logging_directory}/{run_id}/classifiers/simulation_subset_masks/"
            f"subset_replicate_{subset_index}.npz"
        ) as data:
            unsorted_paradigm_simulation_indices = (
                np.array(  # n_paradigms x subset_n_simulations_per_paradigm
                    [
                        data[paradigm.get_modules_string(abbreviated=True)]
                        for paradigm in self.indexer.hypothetical_paradigms
                    ]
                )
            )
            # sort along 2nd axis
            paradigm_simulation_indices = np.empty_like(
                unsorted_paradigm_simulation_indices
            )
            for i in range(paradigm_simulation_indices.shape[0]):
                paradigm_simulation_indices[i] = np.sort(
                    unsorted_paradigm_simulation_indices[i]
                )
        subset_indexer = Indexer(
            "2D_wasserstein" in self.indexer.distance_function_names,
            self.indexer.max_modules,
            simulation_replicates_per_paradigm=paradigm_simulation_indices.shape[1],
            include_true_data=self.indexer.include_true_data,
            distance_function_names=(
                self.indexer.distance_function_names
                if len(self.indexer.distance_function_names) < len(DISTANCE_FUNCTIONS)
                else None
            ),
        )

        # self.distance_matrix has shape (n_distance_functions, n_paradigms,
        # n_paradigms, n_simulations_per_paradigm, n_simulations_per_paradigm)
        # paradigm_simulation_indices has shape (n_paradigms, subset_n_simulations_per_paradigm)
        subset_distance_matrix = np.empty(
            (
                self.distance_matrix.shape[0],
                self.distance_matrix.shape[1],
                self.distance_matrix.shape[1],
                paradigm_simulation_indices.shape[1],
                paradigm_simulation_indices.shape[1],
            )
        )
        for i, j in np.ndindex(self.distance_matrix.shape[1:3]):
            subset_distance_matrix[:, i, j] = self.distance_matrix[:, i, j, :, :][
                :, paradigm_simulation_indices[i]
            ][:, :, paradigm_simulation_indices[j]]
        if self.indexer.include_true_data:
            assert self.true_data_distance is not None
            subset_true_data_distance = np.empty(
                (
                    self.true_data_distance.shape[0],
                    self.true_data_distance.shape[1],
                    paradigm_simulation_indices.shape[1],
                )
            )
            for i in range(self.true_data_distance.shape[1]):
                subset_true_data_distance[:, i] = self.true_data_distance[:, i, :][
                    :, paradigm_simulation_indices[i]
                ]
            return DistanceMatrix(
                subset_indexer,
                subset_distance_matrix,
                subset_true_data_distance,
            )
        return DistanceMatrix(subset_indexer, subset_distance_matrix)


def read_from_file(save_dir: str, indexer: Indexer) -> DistanceMatrix:
    distance_matrix = np.load(f"{save_dir}/distance_matrix.npy")
    if os.path.exists(f"{save_dir}/true_data_distance.npy"):
        true_data_distance = np.load(f"{save_dir}/true_data_distance.npy")
    else:
        true_data_distance = None
    with open(f"{save_dir}/indexer.json", "r", encoding="utf-8") as f:
        indexer_params = json.load(f)
    recorded_indexer = Indexer(**indexer_params)
    if indexer == recorded_indexer:
        return DistanceMatrix(indexer, distance_matrix, true_data_distance)
    print("subsetting distance functions")
    return DistanceMatrix(
        recorded_indexer, distance_matrix, true_data_distance
    ).subset_distance_functions(indexer)


def read_idt_distances(
    logging_directory: str,
    run_id: str,
    indexer: Indexer,
    mds_directory: str,
    patient_subset: str | None,
) -> DistanceMatrix:
    if os.path.exists(f"{mds_directory}/distance_matrix.npy"):
        print(f"Reading distance matrix from {mds_directory}")
        return read_from_file(mds_directory, indexer)
    print(f"Reading IDT distances (MDS dir = {mds_directory})...", end="", flush=True)
    distance_matrix = DistanceMatrix(indexer)
    for distance_function_name in indexer.distance_function_names:
        print(f" {distance_function_name}", end="", flush=True)
        populate_idt_distance_values(
            distance_matrix,
            logging_directory,
            run_id,
            indexer,
            patient_subset,
            distance_function_name,
        )
        if indexer.include_true_data:
            populate_idt_true_data_distances(
                distance_matrix,
                logging_directory,
                run_id,
                indexer,
                patient_subset,
                distance_function_name,
            )

    print(" done")
    # # finally reorder the paradigms so they're in the sorted order
    # distance_matrix.sort_paradigms()
    distance_matrix.write(mds_directory)
    return distance_matrix


def populate_idt_distance_values(
    distance_matrix: DistanceMatrix,
    logging_directory: str,
    run_id: str,
    indexer: Indexer,
    patient_subset: str | None,
    distance_function_name: str,
):
    # if patient_subset is not None:
    patient_subset_index = (
        get_patient_subset_index(
            logging_directory, run_id, patient_subset, distance_function_name, False
        )
        if patient_subset is not None
        else None
    )

    pairwise_distances = np.genfromtxt(
        f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
        "pairwise_distances.csv",
        # "pairwise_distances_subsampled.csv",
        delimiter=",",
        skip_header=1,
        usecols=(range(3) if patient_subset is None else [0, 1, patient_subset_index]),
        # usecols=[0, 1, 13 if patient_subset_index is None else patient_subset_index],
    )
    n_replicates = (1 + np.sqrt(1 + 8 * pairwise_distances.shape[0])) / 2
    assert (
        n_replicates.is_integer()
    ), f"Number of replicates is {n_replicates} from {pairwise_distances.shape[0]}"
    n_replicates = int(n_replicates)
    swap_mask = pairwise_distances[:, 0] > pairwise_distances[:, 1]
    pairwise_distances[swap_mask, 0], pairwise_distances[swap_mask, 1] = (
        pairwise_distances[swap_mask, 1],
        pairwise_distances[swap_mask, 0],
    )
    pairwise_distances = pairwise_distances[
        np.lexsort((pairwise_distances[:, 1], pairwise_distances[:, 0]))
    ]
    this_distance_matrix = np.empty((n_replicates, n_replicates))
    this_distance_matrix[np.triu_indices(n_replicates, k=1)] = pairwise_distances[
        :, 2
    ]  # upper triangle
    this_distance_matrix[np.tril_indices(n_replicates)] = 0  # lower tri + diag
    simulation_replicates_per_paradigm = n_replicates / indexer.n_paradigms
    assert simulation_replicates_per_paradigm.is_integer(), (
        f"{n_replicates} / {indexer.n_paradigms} = "
        f"{simulation_replicates_per_paradigm}"
    )
    simulation_replicates_per_paradigm = int(simulation_replicates_per_paradigm)
    distance_matrix.add_values(
        distance_function_name,
        this_distance_matrix.reshape(
            (
                indexer.n_paradigms,
                simulation_replicates_per_paradigm,
                indexer.n_paradigms,
                simulation_replicates_per_paradigm,
            )
        ).transpose(0, 2, 1, 3),
    )


def get_patient_subset_index(
    logging_directory: str,
    run_id: str,
    patient_subset: str,
    distance_function_name: str,
    true_data: bool,
) -> int:
    with open(
        f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
        f"{'true_data_distances' if true_data else 'pairwise_distances_subsampled'}.csv",
        "r",
        encoding="utf-8",
    ) as f:
        colnames = f.readline().strip().split(",")
    try:
        patient_subset_index = colnames.index(patient_subset)
    except ValueError:
        patient_subset_index = colnames.index(patient_subset.replace("_distance", ""))
    return patient_subset_index


def populate_idt_true_data_distances(
    distance_matrix: DistanceMatrix,
    logging_directory: str,
    run_id: str,
    indexer: Indexer,
    patient_subset: str | None,
    distance_function_name: str,
):
    patient_subset_index = (
        get_patient_subset_index(
            logging_directory, run_id, patient_subset, distance_function_name, True
        )
        if patient_subset is not None
        else None
    )

    true_data_distances = np.genfromtxt(
        f"{logging_directory}/{run_id}/distance/{distance_function_name}/"
        "true_data_distances.csv",
        delimiter=",",
        skip_header=1,
        # usecols=(range(2) if patient_subset is None else [0, patient_subset_index]),
        usecols=[0, 1 if patient_subset_index is None else patient_subset_index],
    )
    n_replicates = true_data_distances.shape[0]

    true_data_distances = true_data_distances[true_data_distances[:, 0].argsort()]
    simulation_replicates_per_paradigm = n_replicates / indexer.n_paradigms
    assert simulation_replicates_per_paradigm.is_integer(), (
        f"{n_replicates} / {indexer.n_paradigms} = "
        f"{simulation_replicates_per_paradigm}"
    )
    simulation_replicates_per_paradigm = int(simulation_replicates_per_paradigm)
    distance_matrix.add_true_data_distances(
        distance_function_name,
        true_data_distances[:, 1].reshape(
            (indexer.n_paradigms, simulation_replicates_per_paradigm)
        ),
    )


def plot_paradigm_differences(distance_matrix: DistanceMatrix, save_dir: str) -> None:
    """
    Boxplot of intra- vs inter-paradigm distances for each distance function and
    variance multiplier.
    """
    fig, axes = plt.subplots(
        distance_matrix.indexer.n_distance_functions,
        1,
        figsize=(3, 3 * distance_matrix.indexer.n_distance_functions),
    )
    for df_index, function_name in enumerate(
        distance_matrix.indexer.distance_function_names
    ):
        # intra-paradigm distances
        axes[df_index].boxplot(
            distance_matrix.distance_matrix[df_index]
            .diagonal(axis1=0, axis2=1)[
                np.triu_indices(
                    distance_matrix.indexer.simulation_replicates_per_paradigm, k=1
                )
            ]
            .reshape(-1),
            positions=[0],
            patch_artist=True,
            boxprops={"facecolor": "red"},
            medianprops={"color": "black"},
        )
        # inter-paradigm distances
        i_upper, j_upper = np.triu_indices(distance_matrix.indexer.n_paradigms, k=1)
        axes[df_index].boxplot(
            distance_matrix.distance_matrix[df_index][i_upper, j_upper].reshape(-1),
            positions=[1],
            patch_artist=True,
            boxprops={"facecolor": "blue"},
            medianprops={"color": "black"},
        )
        axes[df_index].set_xticks([0, 1])
        axes[df_index].set_xticklabels(["intra-paradigm", "inter-paradigm"])
        axes[df_index].set_title(f"{function_name}")
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/paradigm_differences.pdf")
    plt.close(fig)


def write_distance_matrix(distance_matrix: np.ndarray, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    np.save(f"{save_dir}/distance_matrix.npy", distance_matrix)
