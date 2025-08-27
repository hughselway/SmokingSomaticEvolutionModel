import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns  # type: ignore
from sklearn.manifold import MDS  # type: ignore

from .calculate import calculate_aggregated_distances
from .nearest_neighbours import calculate_knn_fractions

from ..calculate_distances import DISTANCE_FUNCTIONS
from ..classifier.indexer import Indexer
from ..classifier.run_classifiers import get_classifier_directory
from ..classifier.run_df_subsets_classifiers import get_distance_function_names_subsets

from ...parse_cmd_line_args import parse_id_test_arguments


def plot_aggregated_distances(
    logging_directory: str,
    run_id: str,
    max_modules: int,
    simulations_per_paradigm: int,
    replicate_count: int,
    distance_function_names: list[str] | None,
    protection_selection_paradigms: bool,
    simulation_subset_index: int | None,
    include_true_data: bool,
    vif_threshold: float | None,
    normalisation_method: str,
    k_fractions: list[float],
    verbose: bool = False,
):
    (
        aggregated_distance,
        true_data_aggregated_distance,
        normalisation_factors,
        included_df_indices,
    ) = calculate_aggregated_distances(
        logging_directory,
        run_id,
        max_modules,
        simulations_per_paradigm * replicate_count,
        distance_function_names,
        protection_selection_paradigms,
        simulation_subset_index,
        include_true_data,
        vif_threshold=vif_threshold,
        normalisation_method=normalisation_method,
        verbose=verbose,
    )
    knn_fractions = calculate_knn_fractions(aggregated_distance, k_fractions)

    indexer = Indexer(
        False,
        -1 if protection_selection_paradigms else max_modules,
        simulations_per_paradigm * replicate_count,
        include_true_data,
        distance_function_names=distance_function_names,
    )
    print(
        "included distance functions:",
        np.array(indexer.distance_function_names)[included_df_indices],
    )
    _, classifier_directory = get_classifier_directory(
        logging_directory,
        run_id,
        distance_function_names=distance_function_names,
        patient_subset=None,
        restrict_to_first_replicate=False,
        restrict_to_n_module_paradigms=-1 if protection_selection_paradigms else None,
        simulation_subset_index=simulation_subset_index,
        include_true_data=include_true_data,
    )
    plot_directory = (
        f"{classifier_directory}/plots/aggregated/{normalisation_method}_normalised/"
        f"vif_threshold_{vif_threshold}"
    )

    # std_boxplot(aggregated_distance, indexer, plot_directory)
    # paradigm_heatmap(aggregated_distance, indexer, plot_directory)
    # confusion_heatmap(knn_fractions, indexer, k_fractions, plot_directory)

    # if protection_selection_paradigms:
    #     for plot_true_data in [True, False] if include_true_data else [False]:
    #         for log_scale in [True, False]:
    #             print(
    #                 f"plotting MDS{'_true_data' if plot_true_data else ''}"
    #                 f"{'_log_scale' if log_scale else ''}..."
    #             )
    #             plot_aggregated_mds(
    #                 aggregated_distance,
    #                 true_data_aggregated_distance if plot_true_data else None,
    #                 indexer,
    #                 f"{plot_directory}/mds{'_true_data' if plot_true_data else ''}",
    #                 log_scale=log_scale,
    #             )


def paradigm_heatmap(
    aggregated_distance: np.ndarray, indexer: Indexer, plot_directory: str
):
    fig, ax = plt.subplots(figsize=(5, 2.7) if indexer.n_paradigms < 10 else (6, 5))
    sns.heatmap(
        aggregated_distance.mean(axis=2).mean(axis=2),
        xticklabels=indexer.get_paradigm_names(abbreviated=False),
        yticklabels=indexer.get_paradigm_names(abbreviated=False),
        cbar_kws={"label": "Mean Aggregated Distance"},
        ax=ax,
        cmap="viridis_r",
        vmin=0,
    )
    ax.set_xlabel("Paradigm")
    ax.set_ylabel("Other Paradigm")
    plt.xticks(rotation=30, ha="right")

    fig.tight_layout()
    save_dir = f"{plot_directory}/paradigm_heatmaps"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/unclustered.pdf")
    fig.savefig(f"{save_dir}/unclustered.png")
    plt.close(fig)

    clustermap = sns.clustermap(
        aggregated_distance.mean(axis=2).mean(axis=2),
        xticklabels=indexer.get_paradigm_names(abbreviated=False),
        yticklabels=indexer.get_paradigm_names(abbreviated=False),
        figsize=(5.2, 4) if indexer.n_paradigms < 10 else (6, 6),
        cbar_kws={"label": "Agg dist"},
        cmap="viridis_r",
        vmin=0,
        dendrogram_ratio=0.24,
        cbar_pos=(0.01, 0.85, 0.02, 0.1),
    )
    clustermap.ax_heatmap.set_xlabel("Paradigm")
    clustermap.ax_heatmap.set_ylabel("Other Paradigm")
    plt.setp(clustermap.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right")
    clustermap.savefig(f"{save_dir}/clustered.pdf")
    clustermap.savefig(f"{save_dir}/clustered.png")
    plt.close()


def confusion_heatmap(
    knn_fractions: np.ndarray,
    indexer: Indexer,
    k_fractions: list[float],
    plot_directory: str,
):
    mean_knn_probabilities = knn_fractions.mean(axis=2)

    for k_fraction_index, k_fr in enumerate(k_fractions):
        k = int(indexer.n_datapoints_excluding_true_data * k_fr)
        fig, ax = plt.subplots(figsize=(5, 4) if indexer.n_paradigms < 10 else (6, 5))
        sns.heatmap(
            mean_knn_probabilities[k_fraction_index],
            xticklabels=indexer.get_paradigm_names(
                abbreviated=indexer.n_paradigms > 10
            ),
            yticklabels=indexer.get_paradigm_names(
                abbreviated=indexer.n_paradigms > 10
            ),
            cmap="viridis",
            ax=ax,
            vmin=0,
            vmax=1,
            cbar_kws={"label": f"Mean proportion of {k} neighbours"},
        )
        # ax.set_title(f"Mean proportion of {k} ({k_fr:.0%}) neighbours")
        ax.set_xlabel("KNN predicted paradigm")
        ax.set_ylabel("True paradigm")

        fig.tight_layout()
        save_dir = f"{plot_directory}/confusion_heatmaps/unclustered"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/top_{k_fr:.0%}.pdf")
        plt.close(fig)

        clustermap = sns.clustermap(
            mean_knn_probabilities[k_fraction_index],
            xticklabels=indexer.get_paradigm_names(
                abbreviated=indexer.n_paradigms > 10
            ),
            yticklabels=indexer.get_paradigm_names(
                abbreviated=indexer.n_paradigms > 10
            ),
            figsize=(5, 4.5) if indexer.n_paradigms < 10 else (6, 6),
            cmap="viridis",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Nb. prop"},
            dendrogram_ratio=0.34,
        )
        clustermap.ax_heatmap.set_xlabel("Neighbour paradigm")
        clustermap.ax_heatmap.set_ylabel("True paradigm")
        plt.setp(
            clustermap.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right"
        )
        # plt.suptitle(f"Mean proportion of {k} ({k_fr:.0%}) neighbours")
        save_dir = f"{plot_directory}/confusion_heatmaps/clustered"
        os.makedirs(save_dir, exist_ok=True)
        clustermap.savefig(f"{save_dir}/top_{k_fr:.0%}.pdf")
        plt.close()


def plot_aggregated_mds(
    aggregated_distance: np.ndarray,
    true_data_aggregated_distance: np.ndarray | None,
    indexer: Indexer,
    plot_directory: str,
    log_scale: bool,
):
    assert np.allclose(aggregated_distance, aggregated_distance.transpose(1, 0, 3, 2))
    aggregated_distance_paradigm_agnostic = np.empty(
        (indexer.n_datapoints, indexer.n_datapoints)
    )
    if true_data_aggregated_distance is not None:
        aggregated_distance_paradigm_agnostic[1:, 1:] = aggregated_distance.transpose(
            0, 2, 1, 3
        ).reshape(
            indexer.n_datapoints_excluding_true_data,
            indexer.n_datapoints_excluding_true_data,
        )
        aggregated_distance_paradigm_agnostic[0, 1:] = (
            true_data_aggregated_distance.flatten()
        )
        aggregated_distance_paradigm_agnostic[1:, 0] = (
            true_data_aggregated_distance.flatten()
        )
        aggregated_distance_paradigm_agnostic[0, 0] = 0
    else:
        aggregated_distance_paradigm_agnostic = aggregated_distance.transpose(
            0, 2, 1, 3
        ).reshape(
            indexer.n_datapoints_excluding_true_data,
            indexer.n_datapoints_excluding_true_data,
        )
    assert np.allclose(
        aggregated_distance_paradigm_agnostic,
        aggregated_distance_paradigm_agnostic.transpose(),
    )
    if log_scale:
        aggregated_distance_paradigm_agnostic = np.log1p(
            aggregated_distance_paradigm_agnostic
        )
    print("fitting MDS...", end="", flush=True)
    mds = MDS(n_components=2, dissimilarity="precomputed", verbose=False, n_jobs=-1)
    mds_coordinates = mds.fit_transform(aggregated_distance_paradigm_agnostic)
    print("done")
    print("MDS stress:", mds.stress_)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.scatterplot(
        x=mds_coordinates[:, 0],
        y=mds_coordinates[:, 1],
        hue=(
            [x + 1 for x in ([-1] + list(indexer.paradigm_index_labels))]
            if true_data_aggregated_distance is not None
            else indexer.paradigm_index_labels
        ),
        alpha=0.5,
        ax=ax,
        palette="tab10",
        legend="full",
        size=5,
    )
    if true_data_aggregated_distance is not None:
        ax.scatter(mds_coordinates[0, 0], mds_coordinates[0, 1], color="red", s=30)
    ax.legend(
        handles=(
            (
                [
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="red",
                        markersize=10,
                    ),
                    # Line2D(
                    #     [0],
                    #     [0],
                    #     marker="s",
                    #     color="w",
                    #     markerfacecolor="purple",
                    #     markersize=10,
                    # ),
                ]
                if true_data_aggregated_distance is not None
                else []
            )
            + (ax.legend_.legend_handles if ax.legend_ is not None else [])
        ),
        labels=(
            (["True Data"] + indexer.get_paradigm_names())
            if true_data_aggregated_distance is not None
            else indexer.get_paradigm_names()
        ),
        # labels=["True Data", "Nearest 1%"] + indexer.get_paradigm_names(),
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Paradigm",
    )

    ax.set_xlabel("MDS Feature 1")
    ax.set_ylabel("MDS Feature 2")
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    os.makedirs(plot_directory, exist_ok=True)
    fig.savefig(f"{plot_directory}/mds{'_log_scale' if log_scale else ''}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    simulation_subset_index_ = None  # range(20)
    protection_selection_paradigms_ = False
    include_true_data_ = True
    vif_threshold_: int | None = 5
    # distance_function_names = None  # get_distance_function_names_subsets(3,True)
    distance_function_names_ = [
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
                # ## two DFs removed because they didn't map properly in the idt_2025-02-05_14-53-34 run
                # "mean_subtracted",
                # "total_branch_length",
                ## extra DFs removed for true data comparison analysis
                "random_control",
                # 0th percentile idt_2025-04-07_10-16-59
                # - none
                # # 10th percentile idt_2025-04-07_10-16-59
                # "mean_subtracted",
                # "mean_subtracted_2D_simplified",
                # "abs_j_one",
                # "l2_j_one",
                # # 20th percentile (as well as above) idt_2025-04-07_10-16-59
                # "z_values",
                # "branch_length_wasserstein",
                # # 25th percentile (as well as above) idt_2025-04-07_10-16-59
                # "wasserstein",
                # "2D_wasserstein_simplified",
                # "smoking_sig_only",
                # "mm_larger_means_sq_diff",
                # "total_branch_length",
                #
                # # 5th percentile idt_2025-02-05_14-53-34
                # "mean_subtracted_2D_simplified",
                # "l2_j_one",
                # # 10th percentile (as well as 5th) idt_2025-02-05_14-53-34
                # "abs_j_one",
                # # 25th percentile (as well as above) idt_2025-02-05_14-53-34
                # "wasserstein",
                # "2D_wasserstein_simplified",
                # "smoking_sig_only",
                # "mm_larger_means_sq_diff",
                # "branch_length_wasserstein",
                # # 5th percentile idt_2024-10-03_19-04-47
                # "abs_j_one",
                # "l2_j_one",
                # # 10th percentile (as well as 5th) idt_2024-10-03_19-04-47
                # "mean_subtracted",
                # "mean_subtracted_2D_simplified",
                # # 25th percentile (as well as above) idt_2024-10-03_19-04-47
                # "wasserstein",
                # "2D_wasserstein_simplified",
                # "smoking_sig_only",
                # "z_values",
                # "mm_larger_means_sq_diff",
            ]
        )
    ]
    for normalisation_method_ in [
        "robust_range_1",
        # "robust_range_2",
        # "robust_range_5",
        # "robust_range_10",
        # "overall_mean",
        # "range",
    ]:
        for vif_threshold_ in [None, 5, 20]:
            for protection_selection_paradigms_option in [False]:  # , True]:
                print(
                    "protection_selection_paradigms:",
                    protection_selection_paradigms_option,
                    "vif_threshold:",
                    vif_threshold_,
                    "normalisation_method:",
                    normalisation_method_,
                )
                plot_aggregated_distances(
                    parsed_args.logging_directory,
                    parsed_args.run_id,
                    parsed_args.max_modules,
                    parsed_args.simulations_per_paradigm,
                    parsed_args.replicate_count,
                    distance_function_names_,
                    protection_selection_paradigms_option,
                    simulation_subset_index_,
                    include_true_data_,
                    vif_threshold_,
                    normalisation_method_,
                    [0.01, 0.05, 0.1, 0.2, 0.5, 1],
                    verbose=True,
                )
            # for simulation_subset_index_option in range(20):
            #     print("simulation_subset_index:", simulation_subset_index_option)
            #     plot_aggregated_distances(
            #         parsed_args.logging_directory,
            #         parsed_args.run_id,
            #         parsed_args.max_modules,
            #         parsed_args.simulations_per_paradigm,
            #         parsed_args.replicate_count,
            #         [
            #             x
            #             for x in DISTANCE_FUNCTIONS
            #             if "control" not in x
            #             and x
            #             not in [
            #                 "mm_dominant_means_sq_diff",
            #                 "mm_larger_means_sq_diff",
            #                 "mm_smaller_means_sq_diff",
            #                 "mm_weighted_means_by_dominance",
            #             ]
            #         ],
            #         protection_selection_paradigms,
            #         simulation_subset_index_option,
            #         include_true_data,
            #         [0.01, 0.05, 0.1, 0.2, 0.5, 1],
            #         verbose=True,
            #     )
            # for distance_function_names_subset in get_distance_function_names_subsets(3, True):
            #     print(distance_function_names_subset)
            #     plot_aggregated_distances(
            #         parsed_args.logging_directory,
            #         parsed_args.run_id,
            #         parsed_args.max_modules,
            #         parsed_args.simulations_per_paradigm,
            #         parsed_args.replicate_count,
            #         distance_function_names_subset,
            #         protection_selection_paradigms,
            #         simulation_subset_index,
            #         include_true_data,
            #         [0.01, 0.05, 0.1, 0.2, 0.5, 1],
            #     )
