import math
import numpy as np
from Bio import Phylo
from scipy.optimize import linear_sum_assignment  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore

from .data.mixture_model_data import (
    MixtureModelData,
    fit_mixture_model,
    get_mixture_model,
)
from .data.mutations_data import get_patient_data


TOTAL_TRUE_DATA_MUTATIONS_PER_PT: dict[str, float] = {
    "total_mutations": 2439072 / 49,  # TODO store in json in data folder
    "smoking_signature_mutations": 947457 / 39,
}


def get_wasserstein_loss_abc(
    patient: str,
    simulation_mutations_data: np.ndarray,
    linear_scale_comparison: bool,
    compare_smoking_signature_mutations: bool,
    data_to_compare: np.ndarray | None = None,
) -> float:
    patient_data = (
        data_to_compare
        if data_to_compare is not None
        else get_patient_data(patient)["total_mutations"]
    )
    try:
        if not linear_scale_comparison:
            patient_data = np.log(patient_data + 1)
            simulation_mutations_data = np.log(simulation_mutations_data + 1)
        return float(
            wasserstein_distance(patient_data, simulation_mutations_data)
            if not compare_smoking_signature_mutations
            else two_dim_wasserstein_distance(
                simulation_mutations_data,
                patient_data,
            )
        )
    except ValueError as err:
        raise ValueError(patient, simulation_mutations_data) from err


def two_dim_wasserstein_distance(
    simulation_data: np.ndarray,
    patient_data: np.ndarray,
    ground_metric: str = "euclidean",
    simplified: bool = False,
) -> float:
    """
    Computes the 1-Wasserstein distance between (the empirical distributions of) two 2D
    point clouds. If simplified, it calculates the 1-Wasserstein of each of the two
    dimensions separately and returns the sum of the two distances. Otherwise, it
    uses equivalence to the linear sum assignment problem to calculate the 1-Wasserstein
    distance between the two point clouds, as described here:
    `https://stackoverflow.com/questions/57562613/python-earth-mover-distance-of-2d-arrays`
    Parameters
    simulation_data : array-like, shape (n_samples, 2)
        (mutation_count, smoking_signature_mutation_count) for each cell subsampled
        from the simulation
    true_data : array-like, shape (n_samples, 2)
        (mutation_count, smoking_signature_mutation_count) for each cell for which we
        have data from the patient
    """
    assert simulation_data.shape[1] == 2 and patient_data.shape[1] == 2, (
        "should have 2 dimensions in both datasets; "
        f"got {simulation_data.shape[1]} and {patient_data.shape[1]}"
    )

    if simplified:
        # calculate the 1-Wasserstein of each of the two dimensions separately
        distance = (
            wasserstein_distance(simulation_data[:, 0], patient_data[:, 0])
            / TOTAL_TRUE_DATA_MUTATIONS_PER_PT["total_mutations"]
            + wasserstein_distance(simulation_data[:, 1], patient_data[:, 1])
            / TOTAL_TRUE_DATA_MUTATIONS_PER_PT["smoking_signature_mutations"]
        )
        assert isinstance(distance, float)
        return distance

    distance_matrix = cdist(simulation_data, patient_data, metric=ground_metric)

    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    return float(distance_matrix[row_ind, col_ind].sum()) / len(row_ind)


def j_one(tree: Phylo.Newick.Tree) -> float:
    sub_terminal_counts = [
        [subclade.count_terminals() for subclade in clade]
        for clade in tree.get_nonterminals()
    ]
    terminal_counts = [clade.count_terminals() for clade in tree.get_nonterminals()]

    numerator = -sum(
        subclade_terminal_count
        * math.log(
            subclade_terminal_count / terminal_count, len(subclade_terminal_counts)
        )
        for terminal_count, subclade_terminal_counts in zip(
            terminal_counts, sub_terminal_counts
        )
        for subclade_terminal_count in subclade_terminal_counts
    )
    denominator = sum(terminal_counts)

    assert denominator > 0, "denominator should be positive"
    j_one_metric = numerator / denominator
    assert isinstance(j_one_metric, float)
    return j_one_metric


def get_mixture_model_distance(
    patient: str,
    simulation_mutations_data: np.ndarray,
    linear_scale_comparison: bool,
    data_to_compare: np.ndarray | MixtureModelData | None = None,
) -> tuple[float, float, float]:
    true_mixture_model = (
        get_mixture_model(patient)
        if data_to_compare is None
        else (
            fit_mixture_model(
                data_to_compare, log_transform=(not linear_scale_comparison)
            )
            if isinstance(data_to_compare, np.ndarray)
            else data_to_compare
        )
    )
    simulation_mixture_model = fit_mixture_model(
        simulation_mutations_data, log_transform=(not linear_scale_comparison)
    )
    return calculate_mixture_model_distance(
        true_mixture_model, simulation_mixture_model
    )


def calculate_mixture_model_distance(
    mixture_1: MixtureModelData, mixture_2: MixtureModelData
) -> tuple[float, float, float]:
    """
    Calculate a distance between two 2-component Gaussian Mixture Models.
    """
    return (
        abs(mixture_1.larger_mean_weight - mixture_2.larger_mean_weight),
        (mixture_1.larger_mean - mixture_2.larger_mean) ** 2,
        (mixture_1.smaller_mean - mixture_2.smaller_mean) ** 2,
    )
