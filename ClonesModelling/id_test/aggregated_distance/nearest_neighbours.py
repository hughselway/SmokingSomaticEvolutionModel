import numpy as np


def calculate_knn_fractions(
    aggregated_distance: np.ndarray, k_fractions: list[float]
) -> np.ndarray:
    n_paradigms, _, n_simulations_per_paradigm, _ = aggregated_distance.shape
    assert aggregated_distance.shape == (
        n_paradigms,
        n_paradigms,
        n_simulations_per_paradigm,
        n_simulations_per_paradigm,
    )
    result = np.zeros(
        (
            len(k_fractions),
            n_paradigms,
            n_simulations_per_paradigm,
            n_paradigms,
        )
    )

    for paradigm_index, simulation_index in np.ndindex(
        n_paradigms, n_simulations_per_paradigm
    ):
        flat_distances = aggregated_distance[
            paradigm_index, :, simulation_index, :
        ].flatten()
        sorted_indices = np.argsort(flat_distances)
        sorted_paradigm_indices = sorted_indices // n_simulations_per_paradigm

        for k_fraction_index, k_fraction in enumerate(k_fractions):
            k = int(k_fraction * n_paradigms * n_simulations_per_paradigm)
            k_nearest_neighbours_indices = sorted_paradigm_indices[1 : k + 1]
            for other_paradigm_index in range(n_paradigms):
                count = np.sum(k_nearest_neighbours_indices == other_paradigm_index)
                result[
                    k_fraction_index,
                    paradigm_index,
                    simulation_index,
                    other_paradigm_index,
                ] = (
                    count / k
                )

    return result
