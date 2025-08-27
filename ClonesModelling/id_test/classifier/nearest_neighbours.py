"""
Assess each distance function for ability to guess the correct paradigm from k nearest
neighbours
"""

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt

from .generate_distance_matrix import DistanceMatrix, read_idt_distances
from .indexer import Indexer
from .run_classifiers import get_classifier_directory

from ...parameters.hypothetical_paradigm_class import MODULE_ORDERING

from ...parse_cmd_line_args import parse_id_test_arguments


def nearest_neighbours(
    distance_matrix: DistanceMatrix,
    n_neighbours_list: list[int],
    subsample_fraction_list: list[float],
    replicates_per_simulation: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each datapoint, find the k nearest neighbours in the feature space, and
    record the paradigm of the majority of the neighbours. Return the accuracy and
    confusion matrix for each distance function.
    """
    accuracy = np.empty(
        (
            distance_matrix.indexer.n_distance_functions,
            len(subsample_fraction_list),
            len(n_neighbours_list),
        )
    )
    confusion_matrices = np.empty(
        (
            distance_matrix.indexer.n_distance_functions,
            len(subsample_fraction_list),
            len(n_neighbours_list),
            distance_matrix.indexer.n_paradigms,
            distance_matrix.indexer.n_paradigms,
        )
    )

    for df_index, df_name in enumerate(distance_matrix.indexer.distance_function_names):
        print(f"Distance function: {df_name}")

        paradigm_agnostic_distances = distance_matrix.get_normalised_paradigm_agnostic_distances(
            df_index
        )  # shape is (n_paradigms * simulation_replicates_per_paradigm, n_paradigms * simulation_replicates_per_paradigm)
        for subsample_fraction_index, subsample_fraction in enumerate(
            subsample_fraction_list
        ):
            # paradigm_agnostic_distances is a (n_paradigms*n_simulations, n_paradigms*n_simulations) matrix
            # we want to subsample this matrix to only include a fraction of the datapoints
            # subsample the rows and columns the same, and subsample within each paradigm
            subsample_size = int(
                subsample_fraction
                * distance_matrix.indexer.simulation_replicates_per_paradigm
            )
            subsample_indices = np.repeat(
                np.random.choice(
                    np.arange(
                        distance_matrix.indexer.simulation_replicates_per_paradigm
                    ),
                    size=subsample_size,
                    replace=False,
                ),
                distance_matrix.indexer.n_paradigms,
            )
            subsampled_paradigm_agnostic_distances = paradigm_agnostic_distances[
                np.ix_(subsample_indices, subsample_indices)
            ]
            sorted_indices = np.argsort(subsampled_paradigm_agnostic_distances, axis=1)
            sorted_paradigm_indices = (
                sorted_indices
                // distance_matrix.indexer.simulation_replicates_per_paradigm
            )
            for n_neighbours_index, n_neighbours in enumerate(n_neighbours_list):
                # find the n_nearest neighbours for each datapoint; not including the datapoint itself
                nearest_neighbours = sorted_paradigm_indices[
                    :,
                    replicates_per_simulation : n_neighbours
                    + replicates_per_simulation,
                ]
                predicted_paradigms = np.empty((len(subsample_indices),), dtype=int)
                # for datapoint_index in range(distance_matrix.indexer.n_datapoints):
                for datapoint_index in range(len(subsample_indices)):
                    predicted_paradigms[datapoint_index] = np.argmax(
                        np.bincount(nearest_neighbours[datapoint_index])
                    )
                # now populate accuracy and confusion matrix
                correct_paradigms = distance_matrix.indexer.paradigm_index_labels[
                    subsample_indices
                ]

                accuracy[df_index, n_neighbours_index] = np.mean(
                    predicted_paradigms == correct_paradigms
                )
                confusion_matrix = np.zeros(
                    (
                        distance_matrix.indexer.n_paradigms,
                        distance_matrix.indexer.n_paradigms,
                    )
                )
                # for i in range(distance_matrix.indexer.n_datapoints):
                for i in range(len(subsample_indices)):
                    confusion_matrix[
                        int(correct_paradigms[i]), int(predicted_paradigms[i])
                    ] += 1
                confusion_matrices[df_index, n_neighbours_index] = (
                    confusion_matrix / np.sum(confusion_matrix, axis=1)[:, np.newaxis]
                )
    return accuracy, confusion_matrices


# try for a few values of nearest_neighbours and plot the output
def fit_and_plot_nearest_neighbours(
    n_neighbours_list: list[int],
    subsample_fraction_list: list[float],
    logging_directory: str,
    run_id: str,
    include_2d_wasserstein: bool,
    max_modules: int,
    simulations_per_paradigm: int,
    replicates_per_simulation: int,
    distance_function_names: list[str] | None = None,
) -> None:
    """
    For each value of n_neighbours in n_neighbours_list, calculate the accuracy and confusion matrix
    for each distance function, and plot the results.
    """

    classifier_directory = get_classifier_directory(
        logging_directory, run_id, distance_function_names
    )
    distance_matrix = read_idt_distances(
        logging_directory,
        run_id,
        Indexer(
            include_2d_wasserstein,
            max_modules,
            simulations_per_paradigm * replicates_per_simulation,
            distance_function_names,
        ),
        classifier_directory,
    )
    if os.path.exists(f"{classifier_directory}/nearest_neighbours/output.npz"):
        accuracies, confusion_matrices = np.load(
            f"{classifier_directory}/nearest_neighbours/output.npz"
        ).values()
        print(
            "loaded data; accuracy shape:",
            accuracies.shape,
            "confusion shape:",
            confusion_matrices.shape,
        )
    else:
        accuracies, confusion_matrices = nearest_neighbours(
            distance_matrix,
            n_neighbours_list,
            subsample_fraction_list,
            replicates_per_simulation,
        )
    os.makedirs(f"{classifier_directory}/nearest_neighbours", exist_ok=True)
    np.savez(
        f"{classifier_directory}/nearest_neighbours/output.npz",
        accuracies=accuracies,
        confusion_matrices=confusion_matrices,
    )

    plot_accuracies(
        accuracies,
        n_neighbours_list,
        distance_matrix.indexer.distance_function_names,
        classifier_directory,
        paradigm_count=distance_matrix.indexer.n_paradigms,
    )
    plot_confusion_matrices(
        confusion_matrices,
        n_neighbours_list,
        distance_matrix.indexer.distance_function_names,
        [
            paradigm.get_modules_string(abbreviated=True)
            for paradigm in distance_matrix.indexer.hypothetical_paradigms
        ],
        classifier_directory,
    )
    plot_confusion_by_module(
        confusion_matrices,
        distance_matrix.indexer,
        classifier_directory,
        n_neighbours_list,
    )


def plot_accuracies(
    accuracies: np.ndarray,
    n_neighbours_list: list[int],
    distance_function_names: list[str],
    classifier_directory: str,
    paradigm_count: int,
) -> None:
    """
    Plot the accuracy of the nearest neighbours classifier for each distance function
    as a function of the number of neighbours.
    """
    # instead of boxplot, have a line for each distance function; x = n_neighbours, y = accuracy
    # separate plots for each distance function but with shared axes and minimal space between them
    mean_accuracies = np.mean(accuracies, axis=1)
    sorted_indices = np.argsort(mean_accuracies)[::-1]
    sorted_accuracies = accuracies[sorted_indices]
    sorted_distance_function_names = [
        distance_function_names[i] for i in sorted_indices
    ]

    ncols = 5
    nrows = len(distance_function_names) // ncols + (
        len(distance_function_names) % ncols > 0
    )
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    for i, df_name in enumerate(sorted_distance_function_names):
        ax = axes[i // ncols, i % ncols]
        ax.plot(n_neighbours_list, sorted_accuracies[i])
        ax.set_title(df_name)
        ax.set_ylim(0, 0.5)
        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(distance_function_names) - i <= ncols:
            # make x labels into integers
            ax.set_xticks(n_neighbours_list)
            ax.set_xticklabels(n_neighbours_list)
            ax.set_xlabel("n_neighbours")
        else:
            ax.set_xticks([])
        if i % ncols == 0:
            ax.set_ylabel("Accuracy")
        else:
            ax.set_yticks([])
        ax.axhline(1 / paradigm_count, color="red", linestyle="--")
    # remove empty axes
    for i in range(len(distance_function_names), nrows * ncols):
        fig.delaxes(axes[i // ncols, i % ncols])
    fig.suptitle("Accuracy of nearest neighbour voting")
    fig.tight_layout()
    os.makedirs(f"{classifier_directory}/plots/nearest_neighbours", exist_ok=True)
    fig.savefig(f"{classifier_directory}/plots/nearest_neighbours/accuracies.pdf")
    plt.close(fig)


def plot_confusion_matrices(
    confusion_matrices: np.ndarray,
    n_neighbours_list: list[int],
    distance_function_names: list[str],
    paradigm_names: list[str],
    classifier_directory: str,
) -> None:
    """
    Plot the confusion matrix of the nearest neighbours classifier for each distance function
    as a function of the number of neighbours.
    """
    os.makedirs(
        f"{classifier_directory}/plots/nearest_neighbours/confusion_matrices",
        exist_ok=True,
    )
    for df_index, df_name in enumerate(distance_function_names):
        fig, axes = plt.subplots(
            1, len(n_neighbours_list), figsize=(5 * len(n_neighbours_list), 5)
        )
        for i, n_neighbours in enumerate(n_neighbours_list):
            ax = axes[i]
            ax.imshow(confusion_matrices[df_index, i])
            ax.set_title(f"n_neighbours = {n_neighbours}")
            # axis labels now
            ax.set_xticks(np.arange(len(paradigm_names)))
            ax.set_xticklabels(paradigm_names, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(paradigm_names)))
            ax.set_yticklabels(paradigm_names)
            ax.set_xlabel("Predicted paradigm")
            ax.set_ylabel("True paradigm")
        fig.tight_layout()
        fig.savefig(
            f"{classifier_directory}/plots/nearest_neighbours/confusion_matrices/{df_name}"
            ".pdf"
        )
        plt.close(fig)


def plot_confusion_by_module(
    confusion_matrices: np.ndarray,
    indexer: Indexer,
    classifier_directory: str,
    n_neighbours_list: list[int],
) -> None:
    """
    Plot the confusion matrix of the nearest neighbours classifier for each distance function
    as a function of the number of neighbours, but with rows and columns grouped by module.
    """
    os.makedirs(
        f"{classifier_directory}/plots/nearest_neighbours/confusion_matrices_by_module",
        exist_ok=True,
    )
    # each paradigm is like a_bc_d, with a, bc and d being modules
    # confusion_matrix[i, j] is the probability of predicting j when the true value is i
    # we want to group the rows and columns by module
    # so we want module_confusion_matrix[i,j] to be the probability of module j being
    # included in the predicted paradigm when module i is the true paradigm
    assert indexer.n_distance_functions == confusion_matrices.shape[0]
    modules = sorted(
        set().union(
            *[
                paradigm.get_module_names(abbreviated=True)
                for paradigm in indexer.hypothetical_paradigms
            ]
        ),
        key=lambda module: MODULE_ORDERING.index(module),
    )
    module_confusion_matrices = np.zeros(
        (
            indexer.n_distance_functions,
            confusion_matrices.shape[1],
            # confusion_matrices.shape[2],
            len(modules),
            len(modules),
        )
    )

    for (
        df_index,
        n_neighbour_index,
        (true_paradigm_index, true_paradigm),
        (predicted_paradigm_index, predicted_paradigm),
    ) in itertools.product(
        range(indexer.n_distance_functions),
        range(confusion_matrices.shape[1]),
        enumerate(indexer.hypothetical_paradigms),
        enumerate(indexer.hypothetical_paradigms),
    ):
        for module_i, module_j in itertools.product(
            true_paradigm.get_module_names(abbreviated=True),
            predicted_paradigm.get_module_names(abbreviated=True),
        ):
            module_confusion_matrices[
                df_index,
                n_neighbour_index,
                modules.index(module_i),
                modules.index(module_j),
            ] += confusion_matrices[
                df_index,
                n_neighbour_index,
                true_paradigm_index,
                predicted_paradigm_index,
            ]
    # normalise by the sum of the rows and columns
    normalised_module_confusion_matrices = module_confusion_matrices.copy()
    for df_index in range(indexer.n_distance_functions):
        for n_neighbour_index in range(confusion_matrices.shape[1]):
            for module_i_index in range(len(modules)):
                normalised_module_confusion_matrices[
                    df_index, n_neighbour_index, module_i_index
                ] /= np.sum(
                    normalised_module_confusion_matrices[
                        df_index, n_neighbour_index, module_i_index
                    ]
                )
            for module_j_index in range(len(modules)):
                normalised_module_confusion_matrices[
                    df_index, n_neighbour_index, :, module_j_index
                ] /= np.sum(
                    normalised_module_confusion_matrices[
                        df_index, n_neighbour_index, :, module_j_index
                    ]
                )
    # now plot the module_confusion_matrices
    for normalised, relevant_confusion_matrices in [
        (False, module_confusion_matrices),
        (True, normalised_module_confusion_matrices),
    ]:
        os.makedirs(
            f"{classifier_directory}/plots/nearest_neighbours/"
            "confusion_matrices_by_module"
            + ("/normalised" if normalised else "/unnormalised"),
            exist_ok=True,
        )
        for df_index, df_name in enumerate(indexer.distance_function_names):
            fig, axes = plt.subplots(
                1,
                relevant_confusion_matrices.shape[1],
                figsize=(5 * relevant_confusion_matrices.shape[1], 5),
            )
            for i in range(relevant_confusion_matrices.shape[1]):
                ax = axes[i]
                ax.imshow(relevant_confusion_matrices[df_index, i])
                ax.set_title(f"n_neighbours = {n_neighbours_list[i]}")
                # axis labels now
                ax.set_xticks(np.arange(len(modules)))
                ax.set_xticklabels(modules, rotation=45, ha="right")
                ax.set_yticks(np.arange(len(modules)))
                ax.set_yticklabels(modules)
                ax.set_xlabel("Predicted module")
                ax.set_ylabel("True module")
            fig.tight_layout()
            fig.savefig(
                f"{classifier_directory}/plots/nearest_neighbours/"
                f"confusion_matrices_by_module/"
                f"{'normalised' if normalised else 'unnormalised'}/{df_name}.png"
            )
            plt.close(fig)


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    fit_and_plot_nearest_neighbours(
        n_neighbours_list=[1, 3, 5, 10, 20],
        subsample_fraction_list=[0.25, 0.5, 0.75, 1],
        logging_directory=parsed_args.logging_directory,
        run_id=parsed_args.run_id,
        include_2d_wasserstein=parsed_args.compare_smoking_signature_mutations,
        max_modules=parsed_args.max_modules,
        simulations_per_paradigm=parsed_args.simulations_per_paradigm,
        replicates_per_simulation=parsed_args.replicate_count,
        distance_function_names=None,
    )
