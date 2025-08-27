from enum import Enum
import os
import numpy as np  # type: ignore
import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import scipy.cluster.hierarchy as sch  # type: ignore

from .distance_post_proc import (
    process_distance_comparison_outputs,
    annotate_paradigm_names_two_replicates,
)
from .read_data import IdentifiabilityDataset

from ..parameters.parameter_class import abbreviate_name
from ..parse_cmd_line_args import parse_id_test_comparison_arguments
from ..parameters.hypothetical_paradigm_class import MODULE_ORDERING


class ModuleComparisonLevel(Enum):
    NEITHER = 0
    ONE = 1
    BOTH = 2


def get_module_comparison(module, paradigm1, paradigm2):
    in_first = module in paradigm1
    in_second = module in paradigm2
    if in_first and in_second:
        return ModuleComparisonLevel.BOTH
    if in_first or in_second:
        return ModuleComparisonLevel.ONE
    return ModuleComparisonLevel.NEITHER


def plot_id_test(
    logging_directory: str,
    run_id: str,
    n_pairs_to_sample: int | None,
    compare_smoking_signature_mutations: bool,
    abbreviate_module_names: bool = False,
) -> None:
    distance_functions = process_distance_comparison_outputs(
        logging_directory,
        run_id,
        n_pairs_to_sample,
        compare_smoking_signature_mutations,
    )
    for function_name in distance_functions:
        distance_dir = f"{logging_directory}/{run_id}/distance/{function_name}"
        pairwise_distances = annotate_paradigm_names_two_replicates(
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
        os.makedirs(f"{distance_dir}/plot", exist_ok=True)
        make_similarity_level_boxplot(
            pairwise_distances, f"{distance_dir}/plot", n_pairs_to_sample, function_name
        )
        make_module_boxplot(
            pairwise_distances, f"{distance_dir}/plot", abbreviate_module_names
        )
        make_paradigm_boxplot(
            pairwise_distances, f"{distance_dir}/plot", abbreviate_module_names
        )
        make_paradigm_violinplot(
            pairwise_distances, f"{distance_dir}/plot", abbreviate_module_names
        )
        for normalisation in [None, "additive", "multiplicative"]:
            plot_paradigm_heatmap(distance_dir, normalisation, abbreviate_module_names)
        plot_dendrogram(
            pairwise_distances,
            logging_directory,
            run_id,
            function_name,
            abbreviate_module_names,
        )
        for include_axis_labels in [True, False]:
            plot_paradigm_dendrogram_heatmap(
                distance_dir, abbreviate_module_names, include_axis_labels
            )


def make_similarity_level_boxplot(
    pairwise_distances: pd.DataFrame,
    output_dir: str,
    n_pairs_to_sample: int | None,
    function_name: str,
) -> None:
    for stratify_by_group in [True, False]:
        fig, axis = plt.subplots()
        sns.boxplot(
            x="similarity_level",
            y="distance",
            hue="group_index" if stratify_by_group else None,
            data=pairwise_distances,
            showfliers=stratify_by_group,
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(function_name)
        if axis.legend_ is not None:
            axis.legend_.remove()
        axis.set_ylim(0, axis.get_ylim()[1])
        if n_pairs_to_sample is not None:
            axis.annotate(
                f"{n_pairs_to_sample} pairs per level",
                xy=(0.1, 0.9),
                xycoords="axes fraction",
                ha="left",
                va="top",
            )
        fig.savefig(
            f"{output_dir}/pairwise_distance"
            f"{'_group_stratified' if stratify_by_group else ''}.pdf"
        )
        plt.close(fig)


def make_module_boxplot(
    pairwise_distances: pd.DataFrame, output_dir: str, abbreviate_module_names: bool
) -> None:
    fig, axis = plt.subplots()
    modules = [x for x in MODULE_ORDERING if x == "base" or len(x) < 3]
    sns.boxplot(
        x="module",
        y="distance",
        data=pairwise_distances.assign(module=[modules] * len(pairwise_distances))
        .explode("module")
        .assign(
            module_comparison=lambda df: df.apply(
                lambda row: get_module_comparison(
                    row["module"], row["paradigm_1"], row["paradigm_2"]
                ).name,
                axis=1,
            )
        ),
        ax=axis,
        hue="module_comparison",
    )
    if abbreviate_module_names:
        axis.set_xticklabels(
            reverse_module_name_abbreviation(
                [x.get_text() for x in axis.get_xticklabels()]
            ),
            rotation=40,
            ha="right",
            va="top",
        )
    axis.set_xlabel("")
    axis.set_ylabel("distance function value")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/module_boxplot.pdf")
    plt.close(fig)


def make_paradigm_boxplot(
    pairwise_distances: pd.DataFrame, output_dir: str, abbreviate_module_names: bool
) -> None:
    fig, axis = plt.subplots()
    sns.boxplot(
        x="paradigm_1",
        y="distance",
        data=pd.concat(
            (
                pairwise_distances,
                (
                    pairwise_distances.loc[
                        lambda df: df.similarity_level != "REPLICATE", :
                    ]
                    .rename(columns={"paradigm_2": "paradigm_2_tmp"})
                    .rename(columns={"paradigm_1": "paradigm_2"})
                    .rename(columns={"paradigm_2_tmp": "paradigm_1"})
                ),
            )
        ),
        ax=axis,
        hue="similarity_level",
    )
    if abbreviate_module_names:
        axis.set_xticklabels(
            reverse_module_name_abbreviation(
                [x.get_text() for x in axis.get_xticklabels()]
            ),
            rotation=40,
            ha="right",
            va="top",
        )
    axis.set_xlabel("")
    axis.set_ylabel("distance function value")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/paradigm_boxplot.pdf")
    plt.close(fig)


def make_paradigm_violinplot(
    pairwise_distances: pd.DataFrame, output_dir: str, abbreviate_module_names: bool
) -> None:
    fig, axis = plt.subplots()
    sns.violinplot(
        x="paradigm_1",
        y="distance",
        data=pairwise_distances.query(
            "similarity_level != 'REPLICATE' & similarity_level != 1"
        ),
        ax=axis,
        hue="similarity_level",
        inner="quart",
        cut=0,
    )
    if abbreviate_module_names:
        axis.set_xticklabels(
            reverse_module_name_abbreviation(
                [x.get_text() for x in axis.get_xticklabels()]
            ),
            rotation=40,
            ha="right",
            va="top",
        )
    axis.set_xlabel("")
    axis.set_ylabel("distance function value")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/paradigm_violinplot.pdf")
    plt.close(fig)


def plot_paradigm_heatmap(
    distance_dir: str, normalisation: str | None, abbreviate_module_names: bool
) -> None:
    paradigm_pairwise_distances = pd.read_csv(
        f"{distance_dir}/paradigm_pairwise_distances.csv"
    ).pivot(index="paradigm_1", columns="paradigm_2", values="distance_mean")
    paradigm_pairwise_distances = paradigm_pairwise_distances.fillna(
        paradigm_pairwise_distances.T
    )
    paradigm_names = (
        reverse_module_name_abbreviation(list(paradigm_pairwise_distances.index))
        if abbreviate_module_names
        else list(paradigm_pairwise_distances.index)
    )
    assert (
        paradigm_pairwise_distances.index == paradigm_pairwise_distances.columns
    ).all()
    paradigm_pairwise_distances = paradigm_pairwise_distances.to_numpy()
    if normalisation == "additive":
        row_means = paradigm_pairwise_distances.mean(axis=1)
        column_means = paradigm_pairwise_distances.mean(axis=0)
        paradigm_pairwise_distances = (
            paradigm_pairwise_distances
            - row_means[:, None]
            - column_means[None, :]
            + row_means.mean()
        )
    elif normalisation == "multiplicative":
        root_row_means = paradigm_pairwise_distances.prod(axis=1) ** (
            1 / (2 * len(paradigm_names))
        )
        root_column_means = paradigm_pairwise_distances.prod(axis=0) ** (
            1 / (2 * len(paradigm_names))
        )
        paradigm_pairwise_distances = (
            paradigm_pairwise_distances
            / root_row_means[:, None]
            / root_column_means[None, :]
        )
    fig, axis = plt.subplots()
    sns.heatmap(paradigm_pairwise_distances, ax=axis, cmap="viridis_r")
    axis.set_xticklabels(paradigm_names, rotation=40, ha="right", va="top")
    axis.set_yticklabels(paradigm_names, rotation=0)
    axis.set_xlabel("")
    axis.set_ylabel("")
    normalisation_string = f" ({normalisation}_normalised)" if normalisation else ""
    axis.set_title("Paradigm Pairwise Distance" + normalisation_string)
    cbar = axis.collections[0].colorbar
    cbar.set_label("distance function value" + normalisation_string)
    fig.tight_layout()
    fig.savefig(
        f"{distance_dir}/plot/paradigm_heatmap"
        f"{normalisation_string.replace(' ', '_')}.pdf"
    )
    plt.close(fig)


def plot_dendrogram(
    pairwise_distances: pd.DataFrame,
    logging_directory: str,
    run_id: str,
    distance_function: str,
    abbreviate_module_names: bool,
) -> None:
    paradigms = IdentifiabilityDataset(
        logging_directory, run_id
    ).replicate_indexer.paradigms_by_index
    condensed_distance_matrix = pairwise_distances.sort_values(
        by=["replicate_1", "replicate_2"]
    ).distance.values
    hierarchy = sch.linkage(condensed_distance_matrix, method="average")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 4))
    output = sch.dendrogram(
        hierarchy, labels=paradigms, orientation="top", ax=axes[0], color_threshold=0
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    width = 5
    module_colours = get_abbreviated_module_colours_dict()
    axes[1].imshow(
        np.array(
            [
                (
                    [module_colours[paradigm] for _ in range(2 * width)]
                    if "-" not in paradigm
                    else (
                        [module_colours[paradigm.split("-")[0]] for _ in range(width)]
                        + [module_colours[paradigm.split("-")[1]] for _ in range(width)]
                    )
                )
                for paradigm in output["ivl"]
            ]
        ).transpose((1, 0, 2)),
    )
    axes[1].set_yticks([])
    axes[1].set_xticks([])

    # add legend for colours
    legend = [
        plt.Line2D([0, 1], [0, 0], color=colour, linewidth=6)
        for _, colour in module_colours.items()
    ]
    legend_labels = (
        reverse_module_name_abbreviation(list(module_colours.keys()))
        if abbreviate_module_names
        else list(module_colours.keys())
    )
    axes[0].legend(legend, legend_labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(
        f"{logging_directory}/{run_id}/distance/{distance_function}/plot/dendrogram.pdf"
    )
    plt.close(fig)


def plot_paradigm_dendrogram_heatmap(
    distance_dir: str, abbreviate_module_names: bool, include_axis_labels: bool
) -> None:
    paradigm_pairwise_distances = pd.read_csv(
        f"{distance_dir}/paradigm_pairwise_distances.csv"
    ).sort_values(by=["paradigm_1", "paradigm_2"])
    paradigm_names = (
        reverse_module_name_abbreviation(
            list(paradigm_pairwise_distances.paradigm_1.unique())
        )
        if abbreviate_module_names
        else list(paradigm_pairwise_distances.paradigm_1.unique())
    )
    module_colours = get_abbreviated_module_colours_dict()

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
    hierarchy, output = plot_paradigm_dendrogram(
        axes[0],
        paradigm_pairwise_distances,
        paradigm_names,
        module_colours,
        abbreviate_module_names,
        include_axis_labels,
    )
    plot_modules(axes[1], output, module_colours)
    plot_paradigm_heatmap_axis(
        axes[2],
        paradigm_pairwise_distances,
        paradigm_names,
        include_axis_labels,
        hierarchy_order=sch.leaves_list(hierarchy)[::-1],
    )

    fig.tight_layout()
    fig.savefig(
        f"{distance_dir}/plot/paradigm_dendrogram"
        f"{'_no_labels' if not include_axis_labels else ''}.pdf"
    )
    plt.close(fig)


def plot_paradigm_dendrogram(
    ax: plt.Axes,
    paradigm_pairwise_distances: pd.DataFrame,
    paradigm_names: list[str],
    module_colours: dict[str, tuple[float, float, float]],
    abbreviate_module_names: bool,
    include_axis_labels: bool,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    hierarchy = sch.linkage(
        paradigm_pairwise_distances.loc[
            lambda df: df.paradigm_1 != df.paradigm_2
        ].distance_mean.values,
        method="average",
    )
    output = sch.dendrogram(
        hierarchy,
        labels=paradigm_names,
        orientation="left",
        ax=ax,
        color_threshold=0,
    )
    ax.set_xticks([])
    if not include_axis_labels:
        ax.set_yticks([])

    legend = [
        plt.Line2D([0, 1], [0, 0], color=colour, linewidth=6)
        for _, colour in module_colours.items()
    ]
    legend_labels = (
        reverse_module_name_abbreviation(list(module_colours.keys()))
        if abbreviate_module_names
        else list(module_colours.keys())
    )
    ax.legend(legend, legend_labels, loc="upper left")
    return hierarchy, output


def plot_modules(
    ax: plt.Axes,
    output: dict[str, list[str]],
    module_colours: dict[str, tuple[float, float, float]],
    max_bar_width: int = 12,
) -> None:
    ax.imshow(
        np.array(
            [
                sum(
                    [
                        [
                            module_colours[module]
                            for _ in range(
                                int(max_bar_width / len(paradigm.split("-")))
                            )
                        ]
                        for module in paradigm.split("-")
                    ],
                    [],
                )
                for paradigm in output["ivl"][::-1]
            ],
        )
    )
    ax.set_yticks([])
    ax.set_xticks([])


def plot_paradigm_heatmap_axis(
    ax: plt.Axes,
    paradigm_pairwise_distances: pd.DataFrame,
    paradigm_names: list[str],
    include_axis_labels: bool,
    hierarchy_order: np.ndarray,
) -> None:
    paradigm_pairwise_distances = paradigm_pairwise_distances.pivot(
        index="paradigm_1", columns="paradigm_2", values="distance_mean"
    ).iloc[hierarchy_order, hierarchy_order]
    paradigm_pairwise_distances = paradigm_pairwise_distances.fillna(
        paradigm_pairwise_distances.T
    )
    paradigm_pairwise_distances = paradigm_pairwise_distances.to_numpy()
    row_means = paradigm_pairwise_distances.mean(axis=1)
    column_means = paradigm_pairwise_distances.mean(axis=0)
    paradigm_pairwise_distances = (
        paradigm_pairwise_distances
        - row_means[:, None]
        - column_means[None, :]
        + row_means.mean()
    )
    tick_labels = (
        [paradigm_names[i] for i in hierarchy_order] if include_axis_labels else False
    )
    sns.heatmap(
        paradigm_pairwise_distances,
        ax=ax,
        cmap="viridis_r",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    for label in ax.get_xticklabels():
        label.set_rotation(40)


def reverse_module_name_abbreviation(module_names: list[str]) -> list[str]:
    reverse_abbreviation_dict = {
        (abbreviate_name(full_name) if full_name != "base" else full_name): full_name
        for full_name in MODULE_ORDERING
        if len(full_name) > 3
    }
    return [
        (
            reverse_abbreviation_dict[module]
            if "-" not in module
            else "-".join(
                reverse_abbreviation_dict[module_part]
                for module_part in module.split("-")
            )
        )
        for module in module_names
    ]


def get_abbreviated_module_colours_dict() -> dict[str, tuple[float, float, float]]:
    return {
        (abbreviate_name(module_name) if module_name != "base" else module_name): colour
        for module_name, colour in get_module_colours_dict().items()
    }


if __name__ == "__main__":
    parsed_args = parse_id_test_comparison_arguments()
    plot_id_test(
        parsed_args.logging_directory,
        parsed_args.run_id,
        parsed_args.n_pairs_to_sample,
        parsed_args.compare_smoking_signature_mutations,
    )
