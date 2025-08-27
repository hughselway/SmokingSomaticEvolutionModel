# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: clonesmodelling-US-Ycvdi-py3.10
#     language: python
#     name: python3
# ---

# %%
import os

os.chdir("/Users/hughselway/Documents/ClonesModelling")

# %%
import numpy as np
from ClonesModelling.id_test.classifier.classifier import read_classifier_output


def get_classifier_output(
    idt_id: str,
    subset_replicate: int | None,
    classifier: str,
    n_features_per_dist_fn: int,
    replicate_index: int,
    all_paradigms: bool = False,
):
    return read_classifier_output(
        f"logs/{idt_id}/classifiers/all_pts/including_true_data/all_replicates/"
        f"{'all' if all_paradigms else 'protection_selection'}_paradigms/df_subsets/"
        "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv/"
        # "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms2s_msmsd_mwmbd_mwmbp_sso_w_zv/"
        + (
            f"simulation_subsets/subset_replicate_{subset_replicate}/"
            if subset_replicate is not None
            else "all_simulations/"
        )
        + f"classifier_outputs/{classifier}/{n_features_per_dist_fn}"
        f"_features_per_dist_fn/replicate_{replicate_index}/output.npz"
    )


def get_subset_size(idt_id: str, subset_replicate: int):
    return np.load(
        f"logs/{idt_id}/classifiers/simulation_subset_masks/subset_replicate_{subset_replicate}.npz"
    )["base"].size


# idt_id = "idt_2024-10-03_19-04-47"
# idt_id = "idt_2025-02-05_14-53-34"
idt_id = "idt_2025-04-07_10-16-59"

# %%
import pandas as pd

# iterator = (
#     (subset_replicate_index, n_features_per_dist_fn, replicate_index, classifier)
#     for subset_replicate_index in range(20)
#     for n_features_per_dist_fn in [2, 3, 5, 10, 20]
#     for replicate_index in range(3)
#     for classifier in [
#         "random_forest",
#         "logistic_regression",
#         "support_vector_machine",
#     ]
# )

all_paradigms = False
classifier_outputs_df = pd.DataFrame(
    {
        "subset_replicate": subset_replicate_index,
        "n_features_per_dist_fn": n_features_per_dist_fn,
        "replicate_index": replicate_index,
        "classifier": classifier,
        "subset_size": get_subset_size(idt_id, subset_replicate_index),
        "cross_val_score": get_classifier_output(
            idt_id,
            subset_replicate_index,
            classifier,
            n_features_per_dist_fn,
            replicate_index,
            all_paradigms=all_paradigms,
        ).cross_val_score,
    }
    for subset_replicate_index in range(20)
    for n_features_per_dist_fn in [2, 3, 5, 10, 20]
    for replicate_index in range(3)
    for classifier in [
        "random_forest",
        "logistic_regression",
        "support_vector_machine",
    ]
).assign(
    cross_val_score=lambda df: df.cross_val_score.astype(float),
)
classifier_outputs_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

random_baseline = 1 / 24 if all_paradigms else 1 / 9

fig, axes = plt.subplots(3, 1, figsize=(6, 5.5), sharex=True, sharey=True)
for i, (ax, classifier) in enumerate(
    zip(axes, ["logistic_regression", "support_vector_machine", "random_forest"])
):
    sns.boxplot(
        data=classifier_outputs_df.query(f"classifier == '{classifier}'"),
        x="subset_size",
        y="cross_val_score",
        hue="n_features_per_dist_fn",
        # palette="viridis",
        ax=ax,
        legend=i == 0,
    )
    if i == 1:
        ax.set_ylabel("Cross-validation accuracy")
    else:
        ax.set_ylabel("")
    ax.set_title(classifier.replace("_", " ").title(), fontsize=11)
    ax.axhline(random_baseline, linestyle="--", alpha=0.5, color="black")
    if i != 0:
        ax.text(
            0.9 * ax.get_xlim()[1],
            random_baseline,
            "Random Baseline",
            verticalalignment="bottom",
            horizontalalignment="right",
        )
# axes[0].set_ylim(0, 1)
axes[0].set_ylim(0, None)
axes[2].set_xlabel("Simulation subset size")
axes[0].legend(
    title="Features per Distance Function",
    loc="lower right",
    ncol=5,
    columnspacing=0.45,
    # title="Features per Distance Function", loc="upper left", ncol=5, columnspacing=0.5
)
save_dir = f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}"
fig.tight_layout()
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/cross_val_score_vs_subset_size.pdf")
display(fig)
plt.close(fig)

# %%
# lineplot, x="subset_size", y="cross_val_score", hue="classifier"
fig, ax = plt.subplots(figsize=(6, 3))
sns.lineplot(
    data=classifier_outputs_df,
    x="subset_size",
    y="cross_val_score",
    style="classifier",
    # all lines black
    color="black",
    markers=True,
    ax=ax,
)
ax.axhline(random_baseline, linestyle="--", alpha=0.5, color="black")
ax.text(
    0.9 * ax.get_xlim()[1],
    random_baseline * 0.95,
    "Random Baseline",
    verticalalignment="top",
    horizontalalignment="right",
)
ax.set_ylabel("Cross-validation accuracy")
ax.set_xlabel("Simulation subset size")
ax.legend(
    title="Classification Method",
    handles=ax.get_legend_handles_labels()[0],
    labels=[
        x.replace("_", " ").title()
        for x in ax.get_legend_handles_labels()[1]
    ],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
save_dir = f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}"
fig.tight_layout()
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/cross_val_score_vs_subset_size_lineplot.pdf")
display(fig)
plt.close(fig)

# %%
classifier_outputs_df

# %%
from matplotlib import ticker as mtick

# lineplot, averaged over n_features_per_dist_fn, x=subset_size, y=cross_val_score, hue=classifier
fig, ax = plt.subplots(1, 1, figsize=(4.5, 2.2))
sns.lineplot(
    data=classifier_outputs_df,
    x="subset_size",
    y="cross_val_score",
    hue="classifier",
    hue_order=["logistic_regression", "support_vector_machine", "random_forest"],
    style="classifier",
    markers=True,
    dashes=False,
    ax=ax,
    n_boot=1000,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    labels=[label.replace("_", " ").title() for label in labels],
    title="Classifier",
    loc="upper left",
    bbox_to_anchor=(1, 1),
    title_fontsize=10,
)
ax.set_xscale("log")
ax.set_xlabel("Simulation subset size")
ax.set_ylabel("Cross-validation accuracy")
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
subset_sizes = sorted(classifier_outputs_df.subset_size.unique())
ax.set_xticks(subset_sizes)
ax.set_xticklabels(subset_sizes)
ax.axhline(random_baseline, linestyle="--", alpha=0.5, color="black")
ax.text(
    0.9 * ax.get_xlim()[1], random_baseline, "Random Baseline", va="bottom", ha="right"
)

# %%
# (subset_replicate_index, n_features_per_dist_fn, replicate_index, classifier) = next(
#     (subset_replicate_index, n_features_per_dist_fn, replicate_index, classifier)
#     for subset_replicate_index in range(20)
#     for n_features_per_dist_fn in [2, 3, 5, 10, 20]
#     for replicate_index in range(3)
#     for classifier in [
#         "random_forest",
#         "logistic_regression",
#         "support_vector_machine",
#     ]
# )
# get_classifier_output(
#     idt_id,
#     subset_replicate_index,
#     classifier,
#     n_features_per_dist_fn,
#     replicate_index,
# ).confusion_matrix
confusion_matrices = np.array(
    [
        [
            [
                [
                    [
                        get_classifier_output(
                            idt_id,
                            subset_size_index * 3 + subset_replicate_index,
                            classifier,
                            n_features_per_dist_fn,
                            mds_replicate_index,
                        ).confusion_matrix
                        for mds_replicate_index in range(3)
                    ]
                    for n_features_per_dist_fn in [2, 3, 5, 10, 20]
                ]
                for classifier in [
                    "random_forest",
                    "logistic_regression",
                    "support_vector_machine",
                ]
            ]
            for subset_replicate_index in range(3)
        ]
        for subset_size_index, _ in enumerate([5, 10, 20, 50, 100, 150, 200])
    ]
)
# subset_size, subset_replicate, classifier, n_features_per_dist_fn, mds_replicate, true, predicted
confusion_matrices.shape

# %%
# mean over first 4 axes
mean_confusion_matrices = confusion_matrices.mean(axis=(1, 2, 3, 4))
mean_confusion_matrices.shape

# %%
# heatmap and clustermap of mean confusion matrices
# paradigm_names = ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"]
full_paradigm_names = {
    "base": "base",
    "q": "quiescent",
    "p": "protected",
    "ir": "immune_response",
    "sd": "smoking_driver",
    "q-ir": "quiescent-immune_response",
    "q-sd": "quiescent-smoking_driver",
    "p-ir": "protected-immune_response",
    "p-sd": "protected-smoking_driver",
}
paradigm_colours = {x: f"C{i}" for i, x in enumerate(full_paradigm_names.keys())}
module_activations = pd.DataFrame(
    {  # row for each x in full_paradigm_names.keys(), column for each module "quiescent","protected","immune_response","smoking_driver" with value being the True or False for whether its included in the paradigm
        hypothesis_name: [
            {
                True: "darkgrey",
                False: "white",
            }[hypothesis_name in paradigm_name.split("-")]
            for paradigm_name in full_paradigm_names.values()
        ]
        for hypothesis_name in [
            "quiescent",
            "protected",
            "immune_response",
            "smoking_driver",
        ]
    },
)
module_activations["combination"] = [paradigm_colours[x] for x in full_paradigm_names.keys()]

for subset_size_index, subset_size in enumerate([5, 10, 20, 50, 100, 150, 200]):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(
        mean_confusion_matrices[subset_size_index],
        cmap="viridis",
        ax=ax,
        xticklabels=full_paradigm_names.values(),
        yticklabels=full_paradigm_names.values(),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Prediction Frequency"},
    )
    ax.set_ylabel("True Paradigm")
    ax.set_xlabel("Predicted Paradigm")
    plt.xticks(rotation=30, ha="right")
    print(f"Mean Confusion Matrix\nSubset Size: {subset_size}")
    fig.tight_layout()
    # display(fig)
    save_dir = f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}/mean_confusion_matrices/unclustered"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/subset_size_{subset_size}.pdf")
    plt.close(fig)

    clustermap = sns.clustermap(
        pd.DataFrame(mean_confusion_matrices[subset_size_index]),
        cmap="viridis",
        figsize=(2.8, 2.4),
        # figsize=(5.5, 4.6),
        xticklabels=list(full_paradigm_names.keys()),
        yticklabels=list(full_paradigm_names.keys()),
        # xticklabels=list(paradigm_names.values()),
        # yticklabels=list(paradigm_names.values()),
        # xticklabels=paradigm_names.values(),
        # yticklabels=paradigm_names.values(),
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Frequency"},
        dendrogram_ratio=(0.23, 0.26),
        row_colors=[paradigm_colours[x] for x in full_paradigm_names.keys()],
        col_colors=[paradigm_colours[x] for x in full_paradigm_names.keys()],
        # row_colors=module_activations,
        # col_colors=module_activations,
    )
    clustermap.ax_heatmap.set_ylabel("True Paradigm")
    clustermap.ax_heatmap.set_xlabel("Predicted Paradigm")
    clustermap.ax_heatmap.set_xticklabels(
        clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha="right"
    )
    print(f"Mean Confusion Matrix\nSubset Size: {subset_size}")
    display(clustermap.figure)
    save_dir = f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}/mean_confusion_matrices/clustered"
    os.makedirs(save_dir, exist_ok=True)
    clustermap.savefig(f"{save_dir}/subset_size_{subset_size}.pdf")
    plt.close(clustermap.figure)

# %%
all_sims_confusion_matrix = {
    all_paradigms: np.array(
        [
            [
                [
                    get_classifier_output(
                        idt_id,
                        None,
                        classifier,
                        n_features_per_dist_fn,
                        mds_replicate_index,
                        all_paradigms,
                    ).confusion_matrix
                    for mds_replicate_index in range(3)
                ]
                for n_features_per_dist_fn in [2, 3, 5, 10, 20]
            ]
            for classifier in [
                "random_forest",
                "logistic_regression",
                "support_vector_machine",
            ]
        ]
    )
    for all_paradigms in [True, False]
}
all_sims_confusion_matrix[False].shape

# %%
all_sims_mean_confusion_matrix = {
    all_paradigms: confusion_matrix.mean(axis=(0, 1, 2))
    for all_paradigms, confusion_matrix in all_sims_confusion_matrix.items()
}
all_sims_mean_confusion_matrix[False].shape

# %%
paradigm_names = {
    True: [
        "base",
        "q",
        "p",
        "ir",
        "sd",
        "q-qp",
        "q-p",
        "q-ir",
        "q-sd",
        "p-ir",
        "p-sd",
        "ir-sd",
        "q-qp-p",
        "q-qp-ir",
        "q-qp-sd",
        "q-p-ir",
        "q-p-sd",
        "q-ir-sd",
        "p-ir-sd",
        "q-qp-p-ir",
        "q-qp-p-sd",
        "q-qp-ir-sd",
        "q-p-ir-sd",
        "q-qp-p-ir-sd",
    ],
    False: ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"],
}

# %%
module_activations = {  # dataframe with columns paradigm_name, quiescent, quiescent_protected, protected, immune_response, smoking_driver
    # eg q-p-ir has q as True, p as True, ir as True, sd as False
    all_paradigms: pd.DataFrame(
        {
            "paradigm_name": paradigm_names[all_paradigms],
            "quiescent": [
                "gray" if "q" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[all_paradigms]
            ],
            "quiescent_protected": [
                "gray" if "qp" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[all_paradigms]
            ],
            "protected": [
                "gray" if "p" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[all_paradigms]
            ],
            "immune_response": [
                "gray" if "ir" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[all_paradigms]
            ],
            "smoking_driver": [
                "gray" if "sd" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[all_paradigms]
            ],
        }
    ).loc[
        :,
        lambda x: (
            x.columns != "quiescent_protected" if not all_paradigms else slice(None)
        ),
    ]
    for all_paradigms in [False, True]
}
module_activations[True]

# %%
pd.DataFrame(
    all_sims_mean_confusion_matrix[all_paradigms],
    index=paradigm_names[all_paradigms],
    columns=paradigm_names[all_paradigms],
)

# %%
for all_paradigms in [True, False]:
    fig, ax = plt.subplots(figsize=(5, 5) if all_paradigms else (5.5, 4))
    sns.heatmap(
        all_sims_mean_confusion_matrix[all_paradigms],
        cmap="viridis",
        ax=ax,
        xticklabels=paradigm_names[all_paradigms],
        yticklabels=paradigm_names[all_paradigms],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Prediction Frequency"},
    )
    plt.xticks(rotation=30, ha="right")
    print(f"Mean Confusion Matrix\nAll Simulations")
    fig.tight_layout()
    display(fig)
    save_dir = (
        f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}/mean_confusion_matrices/unclustered/"
        + ("all_paradigms" if all_paradigms else "protection_selection")
    )
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/all_sims.pdf")
    plt.close(fig)

    clustermap = sns.clustermap(
        pd.DataFrame(
            all_sims_mean_confusion_matrix[all_paradigms],
            index=paradigm_names[all_paradigms],
            columns=paradigm_names[all_paradigms],
        ),
        cmap="viridis",
        figsize=(8, 6) if all_paradigms else (6, 5),
        xticklabels=paradigm_names[all_paradigms],
        yticklabels=paradigm_names[all_paradigms],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Frequency"},
        dendrogram_ratio=(0.13, 0.13),
        # row_colors=[paradigm_colours[x] for x in paradigm_names.keys()],
        # col_colors=[paradigm_colours[x] for x in paradigm_names.keys()],
        row_colors=module_activations[all_paradigms].set_index("paradigm_name"),
        col_colors=module_activations[all_paradigms].set_index("paradigm_name"),
        colors_ratio=(0.04, 0.03) if all_paradigms else (0.06, 0.05),
        cbar_pos=(0.02, 0.85, 0.06, 0.15),
    )
    clustermap.ax_heatmap.set_xticklabels(
        clustermap.ax_heatmap.get_xticklabels(), rotation=30, ha="right"
    )
    clustermap.ax_row_colors.set_xticklabels(
        clustermap.ax_row_colors.get_xticklabels(), rotation=30, ha="right"
    )
    clustermap.ax_heatmap.set_ylabel("True Paradigm")
    clustermap.ax_heatmap.set_xlabel("Predicted Paradigm")

    # clustermap.ax_row_colors.set_xlabel("Module", labelpad=30, x=0.35)
    # clustermap.ax_col_colors.set_ylabel("Module", labelpad=-265, y=0.65)

    print(f"Mean Confusion Matrix\nAll Simulations")
    display(clustermap.figure)
    save_dir = (
        f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}/mean_confusion_matrices/clustered/"
        + ("all_paradigms" if all_paradigms else "protection_selection")
    )
    os.makedirs(save_dir, exist_ok=True)
    clustermap.savefig(f"{save_dir}/all_sims.pdf")
    plt.close(clustermap.figure)

# %%
confusion_matrices[0, 0, 0, 0, 0].sum(axis=1)

# %%
# now we want sensitivity and specificity values
# recall shape of confusion_matrices is :
# subset_size, subset_replicate, classifier, n_features_per_dist_fn, mds_replicate,
# true, predicted
# we want to calculate sensitivity and specificity for each confusion matrix, for each true class
# and the specificity for each predicted class
# separate values for each subset size, subset replicate, classifier, n_features_per_dist_fn, mds_replicate
# so we can plot with error bars
sensitivity = np.zeros(confusion_matrices.shape[:-1])
specificity = np.zeros(confusion_matrices.shape[:-1])
(
    subset_size_count,
    subset_replicate_count,
    classifier_count,
    n_features_per_dist_fn_count,
    mds_replicate_count,
    n_paradigms,
) = confusion_matrices.shape[:-1]
assert confusion_matrices.shape[-1] == n_paradigms
for subset_size_index in range(subset_size_count):
    for subset_replicate_index in range(subset_replicate_count):
        for classifier_index in range(classifier_count):
            for n_features_per_dist_fn_index in range(n_features_per_dist_fn_count):
                for mds_replicate_index in range(mds_replicate_count):
                    confusion_matrix = confusion_matrices[
                        subset_size_index,
                        subset_replicate_index,
                        classifier_index,
                        n_features_per_dist_fn_index,
                        mds_replicate_index,
                    ]
                    for paradigm_index in range(n_paradigms):
                        sensitivity[
                            subset_size_index,
                            subset_replicate_index,
                            classifier_index,
                            n_features_per_dist_fn_index,
                            mds_replicate_index,
                            paradigm_index,
                        ] = confusion_matrix[paradigm_index, paradigm_index]

                        true_negatives = (
                            # sum of predictions where the paradigm is neither true nor predicted
                            confusion_matrix.sum()
                            - confusion_matrix[paradigm_index].sum()
                            - confusion_matrix[:, paradigm_index].sum()
                            + confusion_matrix[paradigm_index, paradigm_index]
                        )
                        false_positives = (
                            confusion_matrix[:, paradigm_index].sum()
                            - confusion_matrix[paradigm_index, paradigm_index]
                        )
                        specificity[
                            subset_size_index,
                            subset_replicate_index,
                            classifier_index,
                            n_features_per_dist_fn_index,
                            mds_replicate_index,
                            paradigm_index,
                        ] = true_negatives / (true_negatives + false_positives)

# %%
sensitivity.shape, specificity.shape

# %%
sensitivity[:, :, [0, 1, 2], :, 0].mean((0, 2, 3, 4)).shape

# %%
paradigm_names

# %%
svm_index = (
    classifier_outputs_df.classifier.unique().tolist().index("support_vector_machine")
)
for omit_svm in [True]:
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.7), sharex=True)
    for i, (ax, metric, metric_name) in enumerate(
        zip(axes, [sensitivity, specificity], ["Sensitivity", "Specificity"])
    ):
        ax.set_xscale("log")
        sns.lineplot(
            data=pd.DataFrame(
                {
                    "subset_size_index": subset_size_index,
                    "subset_size": [5, 10, 20, 50, 100, 150, 200][subset_size_index],
                    # "subset_size": str([5, 10, 20, 50, 100, 150, 200][subset_size_index]),
                    "classifier_index": classifier_index,
                    "n_features_per_dist_fn_index": n_features_per_dist_fn_index,
                    "replicate_index": replicate_index,
                    "paradigm_index": paradigm_index,
                    "paradigm_name": list(full_paradigm_names.values())[paradigm_index]
                    .replace("_", " ")
                    .replace("-", ", ")
                    .title(),
                    metric_name: metric[
                        subset_size_index,
                        subset_replicate_index,
                        classifier_index,
                        n_features_per_dist_fn_index,
                        replicate_index,
                        paradigm_index,
                    ],
                }
                for (
                    subset_size_index,
                    subset_replicate_index,
                    classifier_index,
                    n_features_per_dist_fn_index,
                    replicate_index,
                    paradigm_index,
                ) in np.ndindex(metric.shape)
                if (not omit_svm or classifier_index != svm_index)
            ),
            x="subset_size",
            y=metric_name,
            hue="paradigm_name",
            palette="tab10",
            ax=ax,
            legend=i == 1,
            # markers=["o"],
        )
        # manually scatter
        for paradigm_index in range(n_paradigms):
            ax.scatter(
                [5, 10, 20, 50, 100, 150, 200],
                metric[
                    :,
                    :,
                    [
                        x
                        for x in range(classifier_count)
                        if x != svm_index or not omit_svm
                    ],
                    :,
                    :,
                    paradigm_index,
                ].mean(axis=(0, 2, 3, 4)),
                color=f"C{paradigm_index}",
                marker="o",
                s=10,
            )
        ax.set_xticks([5, 10, 20, 50, 100, 150, 200])
        ax.set_xticklabels([5, 10, 20, 50, 100, None, 200])
        if i == 1:
            ax.set_ylim(None, 1)
            ax.legend(title="Paradigm", bbox_to_anchor=(1, 1.03), loc="upper left")
        else:
            ax.set_ylim(0, 1)
        # set y axis labels to be percentages
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax.set_xlabel("Simulation Subset Size")
        # hline for expected value if random: 1/n_paradigms for sensitivity, 1 - 1/n_paradigms for specificity
        ax.axhline(
            (
                (1 / n_paradigms)
                if metric_name == "Sensitivity"
                else (1 - 1 / n_paradigms)
            ),
            color="k",
            linestyle="--",
        )
        ax.text(
            200,
            (
                (1 / n_paradigms) - 0.01
                if metric_name == "Sensitivity"
                else (1 - 1 / n_paradigms) - 0.002
            ),
            "Random Baseline",
            # transform=ax.transAxes,
            ha="right",
            va="top",
        )
    axes[1].set_yticks([x / 100 for x in range(84, 101, 2)])
    axes[1].set_yticklabels([f"{x/100:.0%}" for x in range(84, 101, 2)])
    fig.tight_layout()
    save_dir = f"notebooks/plots/24-11-12_idt_sim_subsets_plots/{idt_id}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/sensitivity_specificity_vs_subset_size{'_omit_svm' if omit_svm else ''}.pdf"
    )
    print("omitted" if omit_svm else "not omitted", "svm")
    display(fig)
    plt.close(fig)
