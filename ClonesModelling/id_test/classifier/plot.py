import os
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

from .classifier import ClassifierIndexer, CLASSIFIER_NAMES

from ...parameters.parameter_class import abbreviate_name
from ...parameters.hypothesis_module_class import get_module_names
from ...parameters.hypothetical_paradigm_class import MODULE_ORDERING


def plot_if_files_exist(classifier_directory: str) -> bool:
    if (
        os.path.exists(f"{classifier_directory}/cross_val_scores.csv")
        and os.path.exists(f"{classifier_directory}/dist_fn_importances.csv")
        and os.path.exists(f"{classifier_directory}/stresses.csv")
        and os.path.exists(f"{classifier_directory}/confusion_matrices.csv")
    ):
        # return True  # to speed up when running df subsets - comment out otherwise
        print(f"Plotting results from {classifier_directory}")
        cross_val_scores = pd.read_csv(f"{classifier_directory}/cross_val_scores.csv")
        dist_fn_importances = pd.read_csv(
            f"{classifier_directory}/dist_fn_importances.csv"
        )
        stresses = pd.read_csv(f"{classifier_directory}/stresses.csv")
        confusion_matrices = pd.read_csv(
            f"{classifier_directory}/confusion_matrices.csv"
        )

        classifier_indexer = ClassifierIndexer(
            distance_function_names=dist_fn_importances[
                "distance_function_name"
            ].unique(),
            n_features_per_dist_fn_options=sorted(
                dist_fn_importances["n_features_per_dist_fn"].unique()
            ),
        )
        plot_cross_val_scores(
            cross_val_scores,
            classifier_directory,
            n_paradigms=len(confusion_matrices["true_paradigm"].unique()),
        )
        plot_dist_fn_importances(
            dist_fn_importances, classifier_indexer, classifier_directory
        )
        plot_stresses(stresses, classifier_directory)
        plot_confusion_matrices(confusion_matrices, classifier_directory)

        if os.path.exists(f"{classifier_directory}/true_data_predictions.csv"):
            true_data_predictions = pd.read_csv(
                f"{classifier_directory}/true_data_predictions.csv"
            )
            if len(true_data_predictions.predicted_paradigm.unique()) <= 10:
                plot_true_data_predictions(true_data_predictions, classifier_directory)
            plot_module_level_true_data_predictions(
                true_data_predictions, classifier_directory
            )
        return True
    return False


def plot_cross_val_scores(
    cross_val_scores: pd.DataFrame,
    classifier_directory: str,
    n_paradigms: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(3.75, 4.2))
    sns.boxplot(
        data=cross_val_scores,
        x="classifier_name",
        y="cross_val_score",
        hue="n_features_per_dist_fn",
        ax=ax,
    )
    if n_paradigms is not None:
        ax.axhline(y=1 / n_paradigms, color="black", linestyle="--", linewidth=0.5)
        ax.text(
            1,
            1 / n_paradigms + 0.001,
            f"Random Baseline",
            horizontalalignment="center",
            verticalalignment="bottom",
        )
    ax.legend(ncol=2, title="MDS Features per distance function")
    ax.set_ylim(0, 0.645)
    ax.set_ylabel("5-fold cross-validation accuracy")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(
        [label.get_text().replace("_", " ").title() for label in ax.get_xticklabels()],
        rotation=30,
        horizontalalignment="right",
    )
    ax.set_xlabel("Classification Method")
    fig.tight_layout()
    save_dir = f"{classifier_directory}/plots"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/cross_val_scores.pdf")
    ax.get_legend().remove()
    fig.savefig(f"{save_dir}/cross_val_scores_no_legend.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dist_fn_importances(
    dist_fn_importances: pd.DataFrame,
    classifier_indexer: ClassifierIndexer,
    classifier_directory: str,
) -> None:
    for classifier_name in CLASSIFIER_NAMES:
        for n_features_per_dist_fn in classifier_indexer.n_features_per_dist_fn_options:
            fig, ax = plt.subplots(
                figsize=(6, 1 + 0.2 * classifier_indexer.n_distance_functions),
            )
            sns.boxplot(
                data=dist_fn_importances.loc[
                    (dist_fn_importances["classifier_name"] == classifier_name)
                    & (
                        dist_fn_importances["n_features_per_dist_fn"]
                        == n_features_per_dist_fn
                    )
                ],
                x="dist_fn_importance",
                y="distance_function_name",
                ax=ax,
            )
            ax.set_xlim(0, None)
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Distance Function")
            save_dir = (
                f"{classifier_directory}/plots/dist_fn_importances/"
                f"{n_features_per_dist_fn}_features_per_df"
            )
            os.makedirs(save_dir, exist_ok=True)
            fig.tight_layout()
            fig.savefig(f"{save_dir}/{classifier_name}.pdf")
            plt.close(fig)


def plot_stresses(stresses: pd.DataFrame, classifier_directory: str) -> None:
    """
    stresses has columns ["n_features_per_dist_fn",
        "mds_replicate_index","distance_function_name", "stress"]
    """
    df_count = stresses.distance_function_name.nunique()
    fig, ax = plt.subplots(figsize=(8, 1 + 0.2 * df_count))
    ax.set_xscale("log")
    sns.stripplot(
        data=stresses,
        ax=ax,
        x="stress",
        y="distance_function_name",
        order=(
            stresses.groupby("distance_function_name")["stress"]
            .mean()
            .sort_values()
            .index
        ),
        hue="n_features_per_dist_fn",
        orient="h",
        dodge=True,
        size=3,
    )
    ax.set_xlabel("MDS stress")
    ax.set_ylabel("Distance function")
    ax.legend(title="Features per distance function", loc="upper right", ncol=2)
    ax.set_title("Multi-dimensional scaling stress")
    ax.grid(True)
    fig.tight_layout()
    save_dir = f"{classifier_directory}/plots"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/stresses.pdf")
    plt.close(fig)


def plot_confusion_matrices(
    confusion_matrices: pd.DataFrame, classifier_directory: str
) -> None:
    plot_module_accuracies(confusion_matrices, classifier_directory)
    full_module_names = {
        abbreviate_name(module_name): module_name for module_name in get_module_names()
    }
    full_module_names["base"] = "base"
    for classifier_name in confusion_matrices["classifier_name"].unique():
        for n_features_per_dist_fn in confusion_matrices[
            "n_features_per_dist_fn"
        ].unique():
            pivot_df = (
                confusion_matrices.assign(
                    true_paradigm_full=lambda x: x["true_paradigm"].map(
                        lambda abbr_name: "-".join(
                            full_module_names[abbr_module]
                            for abbr_module in abbr_name.split("-")
                        )
                    ),
                    predicted_paradigm_full=lambda x: x["predicted_paradigm"].map(
                        lambda abbr_name: "-".join(
                            full_module_names[abbr_module]
                            for abbr_module in abbr_name.split("-")
                        )
                    ),
                )
                .drop(columns=["true_paradigm", "predicted_paradigm"])
                .rename(
                    columns={
                        "true_paradigm_full": "true_paradigm",
                        "predicted_paradigm_full": "predicted_paradigm",
                    }
                )[
                    (confusion_matrices["classifier_name"] == classifier_name)
                    & (
                        confusion_matrices["n_features_per_dist_fn"]
                        == n_features_per_dist_fn
                    )
                ]
                .groupby(["true_paradigm", "predicted_paradigm"])["confusion"]
                .mean()
                .reset_index()
                .pivot(
                    index="true_paradigm",
                    columns="predicted_paradigm",
                    values="confusion",
                )
            )

            paradigms_in_order = sorted(
                pivot_df.columns,
                key=lambda x: (
                    len(x.split("-")),
                    *[MODULE_ORDERING.index(module) for module in x.split("-")],
                ),
            )
            fig, ax = plt.subplots(
                figsize=(4, 3) if len(paradigms_in_order) < 10 else (6, 4.5)
            )
            sns.heatmap(
                pivot_df.loc[paradigms_in_order, paradigms_in_order],
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Confusion probability"},
                cmap="viridis",
                xticklabels=1,
                yticklabels=1,
            )
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, horizontalalignment="right"
            )
            ax.set_xlabel("Predicted Paradigm")
            ax.set_ylabel("True Paradigm")

            fig.tight_layout()
            save_dir = (
                f"{classifier_directory}/plots/confusion_matrices/"
                f"unclustered/{classifier_name}"
            )
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn.pdf")
            plt.close(fig)

            clustermap = sns.clustermap(
                pivot_df,
                method="average",
                metric="euclidean",
                cmap="viridis",
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Confusion"},
                figsize=(3.5, 3.5) if len(paradigms_in_order) < 10 else (6, 5),
                dendrogram_ratio=(0.2, 0.24),
                xticklabels=1,
                yticklabels=1,
            )
            clustermap.ax_heatmap.set_xticklabels(
                clustermap.ax_heatmap.get_xticklabels(),
                rotation=45,
                horizontalalignment="right",
            )
            clustermap.ax_heatmap.set_xlabel("Predicted Paradigm")
            clustermap.ax_heatmap.set_ylabel("True Paradigm")
            save_dir = (
                f"{classifier_directory}/plots/confusion_matrices/"
                f"clustered/{classifier_name}"
            )
            os.makedirs(save_dir, exist_ok=True)
            clustermap.savefig(
                f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn.pdf"
            )
            plt.close(clustermap.figure)


def plot_true_data_predictions(
    true_data_predictions: pd.DataFrame, classifier_directory: str
) -> None:
    paradigm_palette = [
        (paradigm, sns.color_palette("tab10")[ix])
        for ix, paradigm in enumerate(
            # ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"]
            sorted(
                true_data_predictions["predicted_paradigm"].unique(),
                key=lambda x: (
                    len(x.split("-")),
                    *[MODULE_ORDERING.index(module) for module in x.split("-")],
                ),
            )
        )
    ]
    for n_features_per_dist_fn in [
        None,
        *true_data_predictions["n_features_per_dist_fn"].unique(),
    ]:
        df = (
            true_data_predictions[
                true_data_predictions["n_features_per_dist_fn"]
                == n_features_per_dist_fn
            ]
            if n_features_per_dist_fn is not None
            else true_data_predictions
        )
        frequencies = (
            df.groupby(["classifier_name", "predicted_paradigm"])
            .aggregate(prediction_count=("predicted_paradigm", "count"))
            .reset_index()
            .assign(
                confidence=lambda x: x["prediction_count"]
                / x.groupby("classifier_name")["prediction_count"].transform("sum")
            )
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(
            data=frequencies,
            x="classifier_name",
            y="confidence",
            hue="predicted_paradigm",
            # hue_order=sorted(
            #     df["predicted_paradigm"].unique(),
            #     key=lambda x: (
            #         len(x.split("-")),
            #         *[MODULE_ORDERING.index(module) for module in x.split("-")],
            #     ),
            # ),
            # palette="tab10",
            hue_order=[paradigm for paradigm, _ in paradigm_palette],
            palette={paradigm: colour for paradigm, colour in paradigm_palette},
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Prediction frequency")
        ax.set_xlabel("Classification Method")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            [
                label.get_text().replace("_", " ").title()
                for label in ax.get_xticklabels()
            ],
            rotation=15,
            horizontalalignment="right",
        )
        ax.legend(title="Predicted Paradigm", ncol=3)

        fig.tight_layout()
        save_dir = f"{classifier_directory}/plots/true_data_predictions"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn.pdf"
            if n_features_per_dist_fn is not None
            else f"{save_dir}/all_features_per_dist_fn.pdf"
        )
        ax.get_legend().remove()
        fig.savefig(
            f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn_no_legend.pdf"
            if n_features_per_dist_fn is not None
            else f"{save_dir}/all_features_per_dist_fn_no_legend.pdf"
        )
        plt.close(fig)


def plot_module_level_true_data_predictions(
    true_data_predictions: pd.DataFrame, classifier_directory: str
) -> None:
    module_colours = [
        (module, sns.color_palette("tab10")[ix])
        for ix, module in enumerate(
            sorted(
                [mod for mod in MODULE_ORDERING if mod == "base" or len(mod) < 3],
                key=lambda x: (x == "qp", MODULE_ORDERING.index(x)),
            )
        )
    ]
    for n_features_per_dist_fn in [
        None,
        *true_data_predictions["n_features_per_dist_fn"].unique(),
    ]:
        df = (
            true_data_predictions[
                true_data_predictions["n_features_per_dist_fn"]
                == n_features_per_dist_fn
            ]
            if n_features_per_dist_fn is not None
            else true_data_predictions
        )
        # modules = paradigm.split("-"); need to split and then count
        frequencies = (
            df.assign(module=lambda x: x["predicted_paradigm"].str.split("-"))
            .explode("module")
            .groupby(["classifier_name", "module"])
            .aggregate(prediction_count=("module", "count"))
            .reset_index()
            .assign(
                confidence=lambda x: x["prediction_count"]
                / x.groupby("classifier_name")["prediction_count"].transform("sum")
            )
        )
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.barplot(
            data=frequencies,
            x="classifier_name",
            y="confidence",
            hue="module",
            hue_order=[module for module, _ in module_colours],
            palette=dict(module_colours),
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Prediction frequency")
        ax.set_xlabel("Classification Method")
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            [
                label.get_text().replace("_", " ").title()
                for label in ax.get_xticklabels()
            ],
            rotation=15,
            horizontalalignment="right",
        )
        ax.legend(title="Predicted Module", ncol=3)

        fig.tight_layout()
        save_dir = f"{classifier_directory}/plots/true_data_predictions/module_level"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn.pdf"
            if n_features_per_dist_fn is not None
            else f"{save_dir}/all_features_per_dist_fn.pdf"
        )
        ax.get_legend().remove()
        fig.savefig(
            f"{save_dir}/{n_features_per_dist_fn}_features_per_dist_fn_no_legend.pdf"
            if n_features_per_dist_fn is not None
            else f"{save_dir}/all_features_per_dist_fn_no_legend.pdf"
        )
        plt.close(fig)
