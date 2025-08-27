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
import pandas as pd

df_subset_strings = {
    ## idt_2024-10-03_19-04-47
    # 5: "2ws_blw_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
    # 10: "2ws_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
    # 25: "blw_mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_tbl",
    # None: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
    ## idt_2025-02-05_14-53-34
    # 5: "2ws_ajo_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
    # 10: "2ws_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
    # 25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_zv",
    # None: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms2s_msmsd_mwmbd_mwmbp_sso_w_zv",
    ## idt_2025-04-07_10-16-59
    0: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
    10: "2ws_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
    20: "2ws_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w",
    25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp",
    # None: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv",
}


def get_true_data_predictions(
    idt_id: str,
    subset_replicate: int | None,
    df_percentile_threshold: int | None,
    protection_selection: bool,
    omit_svm: bool = False,
) -> pd.DataFrame:
    df_subset_string = df_subset_strings[df_percentile_threshold or 0]

    true_data_predictions = pd.read_csv(
        f"logs/{idt_id}/classifiers/all_pts/including_true_data/all_replicates/"
        f"{'protection_selection' if protection_selection else 'all'}_paradigms/"
        f"df_subsets/{df_subset_string}/"
        + (
            f"simulation_subsets/subset_replicate_{subset_replicate}/"
            if subset_replicate is not None
            else "all_simulations/"
        )
        + "true_data_predictions.csv"
    )
    if omit_svm:
        true_data_predictions = true_data_predictions.loc[
            (lambda x: x["classifier_name"] != "support_vector_machine")
        ].reset_index(drop=True)
    return true_data_predictions


def get_confusion_matrix(
    idt_id: str,
    subset_replicate: int | None,
    df_percentile_threshold: int,
    protection_selection: bool,
    omit_svm: bool,
) -> pd.DataFrame:
    df_subset_string = df_subset_strings[df_percentile_threshold]

    confusion_matrix = pd.read_csv(
        f"logs/{idt_id}/classifiers/all_pts/including_true_data/all_replicates/"
        f"{'protection_selection' if protection_selection else 'all'}_paradigms/"
        f"df_subsets/{df_subset_string}/"
        + (
            f"simulation_subsets/subset_replicate_{subset_replicate}/"
            if subset_replicate is not None
            else "all_simulations/"
        )
        + "confusion_matrices.csv"
    )
    if omit_svm:
        return confusion_matrix.loc[
            (lambda x: x["classifier_name"] != "support_vector_machine")
        ].reset_index(drop=True)
    return confusion_matrix


# logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/protection_selection_paradigms/df_subsets/blw_mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_tbl/simulation_subsets/subset_replicate_0/confusion_matrices.csv


def get_subset_size(idt_id: str, subset_replicate: int | None) -> int:
    if subset_replicate is None:
        return 250 if idt_id != "idt_2025-04-07_10-16-59" else 300
    return np.load(
        f"logs/{idt_id}/classifiers/simulation_subset_masks/subset_replicate_{subset_replicate}.npz"
    )["base"].size


# idt_id = "idt_2024-10-03_19-04-47"
# idt_id = "idt_2025-02-05_14-53-34"
idt_id = "idt_2025-04-07_10-16-59"

# %%
# df_percentile_thresholds = [5, 10, 25]
# df_percentile_thresholds = [None, 5, 10, 25]
df_percentile_thresholds = [0, 10, 20, 25]
dfs_removed = {0: 0, 10: 3, 20: 6, 25: 9}
true_data_predictions = pd.concat(
    [
        get_true_data_predictions(
            idt_id,
            subset_replicate,
            df_percentile_threshold,
            protection_selection,
            omit_svm=True,
        ).assign(
            subset_replicate=subset_replicate if subset_replicate is not None else 0,
            df_percentile_threshold=df_percentile_threshold,
            dfs_removed=dfs_removed[df_percentile_threshold],
            subset_size=get_subset_size(idt_id, subset_replicate),
            protection_selection=protection_selection,
        )
        for subset_replicate in [None, *range(20)]
        for df_percentile_threshold in df_percentile_thresholds
        for protection_selection in [False, True]
    ]
)
true_data_predictions

# %%
import seaborn as sns
from ClonesModelling.parameters.hypothetical_paradigm_class import MODULE_ORDERING

module_colours = [
    (module, sns.color_palette("tab10")[ix])
    for ix, module in enumerate(
        sorted(
            [mod for mod in MODULE_ORDERING if mod == "base" or len(mod) < 3],
            key=lambda x: (x == "qp", MODULE_ORDERING.index(x)),
        )
    )
]
module_colours

# %%
full_module_names = {
    "base": "base",
    "q": "quiescent",
    "p": "protected",
    "ir": "immune_response",
    "sd": "smoking_driver",
    "qp": "quiescent_protected",
}

# %%
from itertools import product


def get_confidence_values(
    filtered_true_data_predictions: pd.DataFrame,
    x_var: str,
    spread_vars: list[str] | None = None,
    hue_var: str = "predicted_paradigm",
) -> pd.DataFrame:
    """
    Get frequency with which true data is predicted to be each paradigm, stratified by
    `x_var`, `hue_var`, and `spread_vars`.
    """
    if spread_vars is None:
        spread_vars = []
    group_vars = [x_var, hue_var] + spread_vars
    unique_values = [filtered_true_data_predictions[col].unique() for col in group_vars]
    return (
        filtered_true_data_predictions.groupby(group_vars)
        .aggregate(prediction_count=(hue_var, "count"))
        .reindex(
            pd.MultiIndex.from_tuples(list(product(*unique_values)), names=group_vars),
            fill_value=0,
        )
        .reset_index()
        .assign(
            confidence=lambda x: x["prediction_count"]
            / x.groupby([x_var] + spread_vars)["prediction_count"].transform("sum")
        )
    )


# %%
import matplotlib.pyplot as plt

for spread in [
    [],
    # ["classifier_name"],
    # ["classifier_name", "n_features_per_dist_fn"],
    ["subset_replicate"],
    # ["subset_replicate", "classifier_name"],
    # ["subset_replicate", "n_features_per_dist_fn"],
]:
    fig, axes = plt.subplots(
        len(df_percentile_thresholds),
        len(true_data_predictions["classifier_name"].unique()),
        figsize=(7, 2.5 * len(df_percentile_thresholds)),
        sharex=True,
        sharey=True,
    )
    for row, df_percentile_threshold in enumerate(df_percentile_thresholds):
        for col, classifier_name in enumerate(
            true_data_predictions["classifier_name"].unique()
        ):
            module_confidence = get_confidence_values(
                true_data_predictions.query(
                    "protection_selection == False and "
                    f"df_percentile_threshold == {df_percentile_threshold} "
                    f"and classifier_name == '{classifier_name}'"
                    if df_percentile_threshold is not None
                    else "protection_selection == False and "
                    "df_percentile_threshold.isnull() "
                    f"and classifier_name == '{classifier_name}'"
                )
                .assign(module=lambda x: x["predicted_paradigm"].str.split("-"))
                .explode("module"),
                "subset_size",
                spread_vars=spread,
                hue_var="module",
            )
            sns.lineplot(
                # data=true_data_predictions.query(
                #     "protection_selection == False and "
                #     f"df_percentile_threshold == {df_percentile_threshold} "
                #     f"and classifier_name == '{classifier_name}'"
                #     if df_percentile_threshold is not None
                #     else "protection_selection == False and "
                #     "df_percentile_threshold.isnull() "
                #     f"and classifier_name == '{classifier_name}'"
                # )
                # .assign(module=lambda x: x["predicted_paradigm"].str.split("-"))
                # .explode("module")
                # .groupby(["subset_size", "module"] + spread)
                # .aggregate(prediction_count=("module", "count"))
                # .reset_index()
                # .assign(
                #     confidence=lambda x: x["prediction_count"]
                #     / x.groupby(["subset_size"] + spread)["prediction_count"].transform(
                #         "sum"
                #     )
                # ),
                data=module_confidence,
                x="subset_size",
                y="confidence",
                hue="module",
                hue_order=[mod for mod, _ in module_colours],
                ax=axes[row, col],
                palette=dict(module_colours),
            )
            sns.scatterplot(
                # data=true_data_predictions.query(
                #     "protection_selection == False and "
                #     f"df_percentile_threshold == {df_percentile_threshold} and classifier_name == '{classifier_name}'"
                # )
                # .assign(module=lambda x: x["predicted_paradigm"].str.split("-"))
                # .explode("module")
                # .groupby(["subset_size", "module"])
                # .aggregate(prediction_count=("module", "count"))
                # .reset_index()
                # .assign(
                #     confidence=lambda x: x["prediction_count"]
                #     / x.groupby(["subset_size"])["prediction_count"].transform("sum")
                # ),
                data=(
                    module_confidence.groupby(["module", "subset_size"])
                    .agg(confidence=("confidence", "mean"))
                    .reset_index()
                ),
                x="subset_size",
                y="confidence",
                hue="module",
                hue_order=[mod for mod, _ in module_colours],
                ax=axes[row, col],
                palette=dict(module_colours),
                legend=False,
                s=10,
            )
            axes[row, col].set_title(
                f"DF threshold {df_percentile_threshold}%\n"
                f"{classifier_name.replace('_', ' ').title()}",
                fontsize=10,
            )
            axes[row, col].set_xlabel("Subset size")
            if (
                row == 1
                and col == len(true_data_predictions["classifier_name"].unique()) - 1
            ):
                original_handles, original_labels = axes[
                    row, col
                ].get_legend_handles_labels()
                axes[row, col].legend(
                    original_handles,
                    [
                        full_module_names[label].replace("_", "\n_")
                        for label in original_labels
                    ],
                    bbox_to_anchor=(1.05, 1),
                    loc="upper left",
                    title="Module",
                )
                # axes[row, col].legend(
                #     bbox_to_anchor=(1.05, 1), loc="upper left", title="Module", # labels as full module names
                # )
            else:
                axes[row, col].get_legend().remove()
        axes[row, 0].set_ylabel("Predction frequency")
        axes[row, 0].set_xscale("log")
    fig.tight_layout()
    print("spread:", spread)
    display(fig)
    save_dir = "notebooks/plots/24-11-27_true_data_sim_subsets/by_module"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/spread_{'-'.join(spread) if spread else 'none'}.pdf")
    plt.close(fig)

# %%
list(true_data_predictions.columns)

# %%
paradigm_palette = [
    (paradigm, sns.color_palette("tab10")[ix])
    for ix, paradigm in enumerate(
        # ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"]
        sorted(
            true_data_predictions.query("protection_selection == True")[
                "predicted_paradigm"
            ].unique(),
            key=lambda x: (
                len(x.split("-")),
                *[MODULE_ORDERING.index(module) for module in x.split("-")],
            ),
        )
    )
]

for spread in [
    # [],
    # ["classifier_name"],
    # ["subset_replicate"],
    [
        "subset_replicate",
        "classifier_name",
        "n_features_per_dist_fn",
        "mds_replicate_index",
    ]
]:
    fig, axes = plt.subplots(
        len(df_percentile_thresholds),
        1,
        figsize=(7, 2.5 * len(df_percentile_thresholds)),
        sharex=True,
        sharey=True,
    )
    for row, df_percentile_threshold in enumerate(df_percentile_thresholds):
        paradigm_confidence = get_confidence_values(
            true_data_predictions.query(
                "protection_selection == True and "
                f"df_percentile_threshold == {df_percentile_threshold}"
                if df_percentile_threshold is not None
                else "protection_selection == True and df_percentile_threshold.isnull()"
            ),
            "subset_size",
            spread_vars=spread,
        )
        sns.lineplot(
            # data=true_data_predictions.query(
            #     "protection_selection == True and "
            #     f"df_percentile_threshold == {df_percentile_threshold}"
            #     if df_percentile_threshold is not None
            #     else "protection_selection == True and df_percentile_threshold.isnull()"
            # )
            # .groupby(["subset_size", "predicted_paradigm"] + spread)
            # .aggregate(prediction_count=("predicted_paradigm", "count"))
            # .reset_index()
            # .assign(
            #     confidence=lambda x: x["prediction_count"]
            #     / x.groupby(["subset_size"] + spread)["prediction_count"].transform(
            #         "sum"
            #     )
            # ),
            data=paradigm_confidence,
            x="subset_size",
            y="confidence",
            hue="predicted_paradigm",
            hue_order=[paradigm for paradigm, _ in paradigm_palette],
            ax=axes[row],
            palette=dict(paradigm_palette),
        )
        sns.scatterplot(
            # data=true_data_predictions.query(
            #     "protection_selection == True and "
            #     f"df_percentile_threshold == {df_percentile_threshold}"
            #     if df_percentile_threshold is not None
            #     else "protection_selection == True and df_percentile_threshold.isnull()"
            # )
            # .groupby(["subset_size", "predicted_paradigm"])
            # .aggregate(prediction_count=("predicted_paradigm", "count"))
            # .reset_index()
            # .assign(
            #     confidence=lambda x: x["prediction_count"]
            #     / x.groupby("subset_size")["prediction_count"].transform("sum")
            # ),
            data=(
                paradigm_confidence.groupby(["predicted_paradigm", "subset_size"])
                .agg(confidence=("confidence", "mean"))
                .reset_index()
            ),
            x="subset_size",
            y="confidence",
            hue="predicted_paradigm",
            hue_order=[paradigm for paradigm, _ in paradigm_palette],
            ax=axes[row],
            palette=dict(paradigm_palette),
            s=15,
            marker="o",
            # edgecolor="black",
            linewidth=0.5,
            legend=False,
        )
        axes[row].set_ylabel("Prediction frequency")
        axes[row].set_title(f"DF percentile threshold: {df_percentile_threshold}")
        if row == 0:
            axes[row].legend(loc="upper left", bbox_to_anchor=(1, 1))
            # axes[i].legend(loc="upper left", ncol=3)
        else:
            axes[row].get_legend().remove()
    axes[-1].set_xlabel("Subset size")
    axes[-1].set_xscale("log")
    axes[-1].set_xticks(sorted(true_data_predictions["subset_size"].unique()))
    axes[-1].set_xticklabels(
        [f"{size:,}" for size in sorted(true_data_predictions["subset_size"].unique())]
    )
    # axes[-1].set_ylim(0, None)
    fig.tight_layout()
    # reduce hspace
    fig.subplots_adjust(hspace=0.35)
    print("include_spread:", spread)
    display(fig)
    plt.close(fig)

# %%
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D

for spread in [
    # [],
    [
        "subset_replicate",
        "classifier_name",
        "n_features_per_dist_fn",
        "mds_replicate_index",
    ],
]:
    fig, axes = plt.subplots(
        # len([x for x in df_percentile_thresholds if x is not None]),
        len(df_percentile_thresholds),
        len(true_data_predictions["classifier_name"].unique()),
        figsize=(5.5, 1.5 * len(df_percentile_thresholds)),
        sharex=True,
        sharey=True,
    )
    for row, df_percentile_threshold in enumerate(
        # [x for x in df_percentile_thresholds if x is not None]
        df_percentile_thresholds
    ):
        for col, classifier_name in enumerate(
            true_data_predictions["classifier_name"].unique()
        ):
            paradigm_confidence = get_confidence_values(
                true_data_predictions.query(
                    "protection_selection == True and "
                    f"df_percentile_threshold == {df_percentile_threshold} and classifier_name == '{classifier_name}'"
                    if df_percentile_threshold is not None
                    else "protection_selection == True and "
                    f"df_percentile_threshold.isnull() and classifier_name == '{classifier_name}'"
                ),
                "subset_size",
                spread_vars=spread,
            )
            # grouped_paradigm_frequencies = (
            #     true_data_predictions.query(
            #         "protection_selection == True and "
            #         f"df_percentile_threshold == {df_percentile_threshold} and classifier_name == '{classifier_name}'"
            #         if df_percentile_threshold is not None
            #         else "protection_selection == True and "
            #         f"df_percentile_threshold.isnull() and classifier_name == '{classifier_name}'"
            #     )
            #     .groupby(["subset_size", "predicted_paradigm"] + spread)
            #     .aggregate(prediction_count=("predicted_paradigm", "count"))
            #     .reset_index()
            #     .assign(
            #         confidence=lambda x: x["prediction_count"]
            #         / x.groupby(["subset_size"] + spread)["prediction_count"].transform(
            #             "sum"
            #         )
            #     )
            # )
            sns.lineplot(
                data=paradigm_confidence,
                x="subset_size",
                y="confidence",
                hue="predicted_paradigm",
                hue_order=[paradigm for paradigm, _ in paradigm_palette],
                ax=axes[row, col],
                palette=dict(paradigm_palette),
                n_boot=1000,
            )
            sns.scatterplot(
                data=paradigm_confidence.groupby(["subset_size", "predicted_paradigm"])
                .agg(confidence=("confidence", "mean"))
                .reset_index(),
                x="subset_size",
                y="confidence",
                hue="predicted_paradigm",
                hue_order=[paradigm for paradigm, _ in paradigm_palette],
                ax=axes[row, col],
                palette=dict(paradigm_palette),
                s=15,
                marker="o",
                # edgecolor="black",
                linewidth=0.5,
                legend=False,
            )
            axes[row, col].set_title(
                f"{classifier_name.replace('_', ' ').title()}\n" * (row == 0)
                + f"{dfs_removed[df_percentile_threshold]} DFs removed",
                # + f"DF threshold {df_percentile_threshold}%",
                # f"{''.join(x[0] for x in classifier_name.title().split('_'))}"
                size=10,
            )
            axes[row, col].axhline(
                1 / len(paradigm_palette), color="black", linestyle=":", alpha=0.75
            )
            axes[row, col].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
            if (
                row == 1
                and col == len(true_data_predictions["classifier_name"].unique()) - 1
            ):
                original_handles, original_labels = axes[
                    row, col
                ].get_legend_handles_labels()
                # original_handles.append(
                #     plt.Line2D([0], [0], color="black", linestyle="--")
                # )
                # original_labels.append("Random\nbaseline")
                axes[row, col].legend(
                    original_handles
                    + [Line2D([0], [0], color="black", linestyle=":", alpha=0.75)],
                    [
                        "-\n".join(full_module_names[mod] for mod in label.split("-"))
                        for label in original_labels
                    ]
                    + ["Random\nbaseline"],
                    bbox_to_anchor=(1, 0.5),
                    loc="center left",
                    title="Paradigm",
                    # labelspacing=0.1,
                )
                axes[row, 0].set_ylabel("Prediction frequency")
                # axes[row, col].legend(loc="center left", bbox_to_anchor=(1, 0.5))
            else:
                axes[row, col].get_legend().remove()
                axes[row, col].set_ylabel("")
    axes[-1, 1].set_xlabel("Subset size")
    for ax in (axes[-1, 0], axes[-1, 1]):
        ax.set_xlabel("")
    axes[-1, -1].set_xscale("log")
    # axes[-1,-1].set_xticks(sorted(true_data_predictions["subset_size"].unique()))
    # axes[-1,-1].set_xticklabels(
    #     [f"{size:,}" for size in sorted(true_data_predictions["subset_size"].unique())]
    # )
    # axes[-1].set_ylim(0, None)
    fig.tight_layout()
    # reduce hspace
    fig.subplots_adjust(hspace=0.34)
    print("include_spread:", spread)
    display(fig)
    save_dir = "notebooks/plots/24-11-27_true_data_sim_subsets/by_classifier"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/{'sim_subsets_paradigm_pred' if spread else 'no_spread'}.pdf"
    )
    plt.close(fig)

# %%
true_data_predictions.columns

# %%
# two columns, 3 rows; all at subset_size=250
# 1st column: all paradigms (protection_selection=False), module-level barplot
# 2nd column: protection_selection=True, paradigm-level barplot
# row by df_percentile_threshold
fig, axes = plt.subplots(
    len(df_percentile_thresholds),
    2,
    figsize=(5, 1.6 * len(df_percentile_thresholds)),
    sharex=True,
    sharey=True,
)
for row, df_percentile_threshold in enumerate(df_percentile_thresholds):
    for col, protection_selection in enumerate([False, True]):
        grouped_paradigm_frequencies = true_data_predictions.query(
            f"subset_size == {true_data_predictions['subset_size'].max()} and "
            f"protection_selection == {protection_selection} and "
            f"df_percentile_threshold == {df_percentile_threshold}"
            if df_percentile_threshold is not None
            else f"subset_size == {true_data_predictions['subset_size'].max()} and "
            f"protection_selection == {protection_selection} and "
            "df_percentile_threshold.isnull()"
        )
        sns.barplot(
            data=(
                grouped_paradigm_frequencies.assign(
                    module=lambda x: x["predicted_paradigm"].str.split("-")
                )
                .explode("module")
                .groupby(["classifier_name", "module"])
                .aggregate(prediction_count=("module", "count"))
                .reset_index()
                .assign(
                    confidence=lambda x: x["prediction_count"]
                    / x.groupby("classifier_name")["prediction_count"].transform("sum"),
                    module=lambda x: x["module"].map(full_module_names),
                )
                if not protection_selection
                else grouped_paradigm_frequencies.groupby(
                    ["classifier_name", "predicted_paradigm"]
                )
                .aggregate(prediction_count=("predicted_paradigm", "count"))
                .reset_index()
                .assign(
                    confidence=lambda x: x["prediction_count"]
                    / x.groupby("classifier_name")["prediction_count"].transform("sum")
                )
            ),
            x="classifier_name",
            y="confidence",
            order=["logistic_regression", "random_forest"],
            hue="predicted_paradigm" if protection_selection else "module",
            ax=axes[row, col],
            palette=(
                dict(paradigm_palette)
                if protection_selection
                else {full_module_names[mod]: col for mod, col in module_colours}
            ),
            hue_order=(
                [paradigm for paradigm, _ in paradigm_palette]
                if protection_selection
                else [full_module_names[mod] for mod, _ in module_colours]
            ),
        )
        axes[row, col].set_title(
            f"{'Protection selection' if protection_selection else 'All paradigms'}\n"
            * (row == 0)
            + f"DF threshold {df_percentile_threshold}%",
            fontsize=10,
        )
        if row == 1:
            axes[row, col].set_ylabel("Prediction frequency")
        else:
            axes[row, col].set_ylabel("")
        axes[row, col].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
        if row == 0:
            # legend above
            axes[row, col].legend(
                loc="lower center",
                bbox_to_anchor=(0.45 if col == 0 else 0.5, 1.4),
                ncol=2 if col == 0 else 3,
                title="Paradigm" if protection_selection else "Module",
                # less space between columns
                columnspacing=0.5,
            )
        else:
            axes[row, col].get_legend().remove()
for col, protection_selection in enumerate([False, True]):
    axes[-1, col].set_xlabel("Classifier")
    axes[-1, col].set_xticklabels(
        [
            x.get_text().replace("_", " ").title()
            for x in axes[-1, col].get_xticklabels()
        ],
        rotation=15,
        ha="right",
    )
fig.tight_layout()
fig.subplots_adjust(hspace=0.35, wspace=0.1, left=0.15, top=0.8, right=1)
display(fig)
save_dir = "notebooks/plots/24-11-27_true_data_sim_subsets/all_simulations"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/module_paradigm_level.pdf")
plt.close(fig)

# %%
# same plot, but only the protection_selection=True
fig, axes = plt.subplots(
    len(df_percentile_thresholds),
    1,
    figsize=(3.2, 1.8 * len(df_percentile_thresholds)),
    sharex=True,
    sharey=True,
)
for row, df_percentile_threshold in enumerate(df_percentile_thresholds):
    grouped_paradigm_frequencies = true_data_predictions.query(
        f"subset_size == {true_data_predictions['subset_size'].max()} and "
        f"protection_selection == True and "
        f"df_percentile_threshold == {df_percentile_threshold}"
        if df_percentile_threshold is not None
        else f"subset_size == {true_data_predictions['subset_size'].max()} and "
        f"protection_selection == True and "
        "df_percentile_threshold.isnull()"
    )
    sns.barplot(
        data=grouped_paradigm_frequencies.groupby(
            ["classifier_name", "predicted_paradigm"]
        )
        .aggregate(prediction_count=("predicted_paradigm", "count"))
        .reset_index()
        .assign(
            confidence=lambda x: x["prediction_count"]
            / x.groupby("classifier_name")["prediction_count"].transform("sum")
        ),
        x="classifier_name",
        y="confidence",
        order=["logistic_regression", "random_forest"],
        hue="predicted_paradigm",
        ax=axes[row],
        palette=dict(paradigm_palette),
        hue_order=[paradigm for paradigm, _ in paradigm_palette],
    )
    axes[row].set_title(f"DF threshold {df_percentile_threshold}%", fontsize=10)
    if row == 1:
        axes[row].set_ylabel("Prediction frequency")
    else:
        axes[row].set_ylabel("")
    if row == len(df_percentile_thresholds) - 1:
        axes[row].set_xlabel("Classification method")
        axes[row].set_xticklabels(
            [
                x.get_text().replace("_", " ").title()
                for x in axes[row].get_xticklabels()
            ],
            rotation=30,
            ha="right",
        )
    axes[row].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
    if row == 0:
        # legend above
        axes[row].legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.35),
            ncol=3,
            title="Paradigm",
            # less space between columns
            columnspacing=0.5,
        )
    else:
        axes[row].get_legend().remove()
fig.tight_layout()
fig.savefig(f"{save_dir}/paradigm_level.pdf")

# %%
true_data_predictions.columns

# %%
# let's do that again, but this time...x=df_percentile_threshold, y=prediction frequency, colour=paradigm
# different classifiers provide the spread
fig, ax = plt.subplots(1, 2, figsize=(8, 3.8), width_ratios=(1, 0.2), sharey=True)


grouped_paradigm_frequencies = get_confidence_values(
    true_data_predictions.query(
        f"subset_size == {true_data_predictions['subset_size'].max()} and protection_selection == True"
    ),
    x_var="dfs_removed",
    spread_vars=["classifier_name", "mds_replicate_index", "n_features_per_dist_fn"],
)

sns.lineplot(
    grouped_paradigm_frequencies,
    x="dfs_removed",
    y="confidence",
    hue="predicted_paradigm",
    palette=dict(paradigm_palette),
    hue_order=[paradigm for paradigm, _ in paradigm_palette],
    ax=ax[0],
)
sns.scatterplot(
    grouped_paradigm_frequencies.groupby(["predicted_paradigm", "dfs_removed"])
    .agg(confidence=("confidence", "mean"))
    .reset_index(),
    x="dfs_removed",
    y="confidence",
    hue="predicted_paradigm",
    palette=dict(paradigm_palette),
    hue_order=[paradigm for paradigm, _ in paradigm_palette],
    ax=ax[0],
    legend=False,
)
ax[0].axhline(1 / len(paradigm_palette), color="black", linestyle=":", alpha=0.75)
ax[0].text(
    6,
    1 / len(paradigm_palette),
    "Random baseline",
    ha="center",
    va="bottom",
    fontsize=10,
    color="black",
)

ax[0].set_ylim(0, 1)
ax[0].yaxis.set_major_formatter(mtick.PercentFormatter(1, 0))
ax[0].set_xlabel("Removed distance functions")
ax[0].set_ylabel("Prediction frequency")
ax[0].set_xticks(sorted(true_data_predictions["dfs_removed"].unique()))

# second plot, stacked barplot
grouped_paradigm_frequencies = (
    true_data_predictions.query(
        f"subset_size == {true_data_predictions['subset_size'].max()} and protection_selection == True"
    )
    .groupby(["dfs_removed", "predicted_paradigm"])
    .aggregate(prediction_count=("predicted_paradigm", "count"))
    .reset_index()
    .assign(
        confidence=lambda x: x["prediction_count"]
        / x.groupby("dfs_removed")["prediction_count"].transform("sum")
    )
)
grouped_paradigm_frequencies_pivot = grouped_paradigm_frequencies.pivot(
    index="dfs_removed", columns="predicted_paradigm", values="confidence"
)
grouped_paradigm_frequencies_pivot.plot(
    kind="bar",
    stacked=True,
    ax=ax[1],
    color=dict(paradigm_palette),
    width=0.8,
)
ax[1].set_xlabel("Removed distance functions")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
original_handles, original_labels = ax[0].get_legend_handles_labels()
ax[0].get_legend().remove()
ax[1].legend(
    original_handles + [Line2D([0], [0], color="black", linestyle=":", alpha=0.75)],
    [
        "-\n".join(full_module_names[mod] for mod in label.split("-"))
        for label in original_labels
    ]
    + ["Random\nbaseline"],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
    title="Paradigm",
)

fig.tight_layout()

save_dir = "notebooks/plots/24-11-27_true_data_sim_subsets/by_extremity"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/max_subset_size.pdf")
display(fig)
plt.close(fig)

# %%
grouped_paradigm_frequencies_pivot

# %% [markdown]
# This is the plot. Now for the supplementary, want to separate it by classifier, by n_features maybe... seeing what subset size does to it I think might still be the previous version of the plot (where subset size is x).
#
# Additional bit for the main fig: want specificity (y) by extremity threshold (x) with hue=paradigm.
#
# Then can show this for MDS classifiers and for aggregated separately, to show that agg is worse generally & degrades more but also shows q-ir.

# %%
paradigm_palette

# %%
# (
#     get_confusion_matrix(
#         idt_id,
#         subset_replicate=None,
#         df_percentile_threshold=25,
#         protection_selection=True,
#         omit_svm=False,
#     )
#     .assign(paradigm=[[paradigm for paradigm, _ in paradigm_palette]])
#     .groupby(
#         [
#             "classifier_name",
#             "mds_replicate_index",
#             "n_features_per_dist_fn",
#             "true_paradigm",
#         ]
#     )
#     .apply(
#         lambda group: pd.Series(
#             {
#                 "specificity": (
#                     group.loc[
#                         group["true_paradigm"] != group["predicted_paradigm"],
#                         "confusion",
#                     ].sum()  # True negatives
#                     / group.loc[
#                         group["true_paradigm"] != group["predicted_paradigm"],
#                         "confusion",
#                     ].sum()  # True negatives
#                     + group.loc[
#                         group["true_paradigm"] == group["predicted_paradigm"],
#                         "confusion",
#                     ].sum()  # False positives
#                 )
#             }
#         )
#     )
#     .reset_index()
#     .rename(columns={"true_paradigm": "paradigm"})
# )

# %%

# %%
mean_confusion_matrices = {
    protection_selection: (
        pd.concat(
            [
                get_confusion_matrix(
                    idt_id,
                    subset_replicate,
                    df_percentile_threshold,
                    protection_selection,
                    omit_svm=True,
                ).assign(
                    subset_replicate=(
                        subset_replicate if subset_replicate is not None else -1
                    ),
                    df_percentile_threshold=str(df_percentile_threshold),
                    subset_size=get_subset_size(idt_id, subset_replicate),
                    protection_selection=protection_selection,
                )
                for subset_replicate in [None]  # , *range(20)]
                for df_percentile_threshold in df_percentile_thresholds
            ]
        )
        .groupby(["df_percentile_threshold", "true_paradigm", "predicted_paradigm"])
        .aggregate(confusion=("confusion", "mean"))
        .reset_index()
    )
    for protection_selection in [False, True]
}
display(mean_confusion_matrices[True])
display(mean_confusion_matrices[False])

# %%
mean_confusion_matrices[True]["df_percentile_threshold"].unique()

# %%
paradigm_names = {
    protection_selection: sorted(
        mean_confusion_matrices[protection_selection]["predicted_paradigm"].unique(),
        key=lambda x: (
            len(x.split("-")),
            *[MODULE_ORDERING.index(module) for module in x.split("-")],
        ),
    )
    for protection_selection in [False, True]
}
module_activations = {  # dataframe with columns paradigm_name, quiescent, quiescent_protected, protected, immune_response, smoking_driver
    # eg q-p-ir has q as True, p as True, ir as True, sd as False
    protection_selection: pd.DataFrame(
        {
            "paradigm_name": paradigm_names[protection_selection],
            "quiescent": [
                "gray" if "q" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[protection_selection]
            ],
            "quiescent_protected": [
                "gray" if "qp" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[protection_selection]
            ],
            "protected": [
                "gray" if "p" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[protection_selection]
            ],
            "immune_response": [
                "gray" if "ir" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[protection_selection]
            ],
            "smoking_driver": [
                "gray" if "sd" in paradigm.split("-") else "white"
                for paradigm in paradigm_names[protection_selection]
            ],
        }
    ).loc[
        :,
        lambda x: (
            x.columns != "quiescent_protected" if protection_selection else slice(None)
        ),
    ]
    for protection_selection in [False, True]
}
module_activations[True]

# %%
# paradigm_names = {
#     "base": "base",
#     "q": "quiescent",
#     "p": "protected",
#     "ir": "immune_response",
#     "sd": "smoking_driver",
#     "q-ir": "quiescent-immune_response",
#     "q-sd": "quiescent-smoking_driver",
#     "p-ir": "protected-immune_response",
#     "p-sd": "protected-smoking_driver",
# }
paradigm_colours = {x: f"C{i}" for i, x in enumerate(paradigm_names.keys())}

for protection_selection in [False, True]:
    for df_percentile_threshold in df_percentile_thresholds:
        clustermap = sns.clustermap(
            pd.DataFrame(
                mean_confusion_matrices[protection_selection][
                    (
                        mean_confusion_matrices[protection_selection][
                            "df_percentile_threshold"
                        ]
                        == str(df_percentile_threshold)
                    )
                    # & (
                    #     mean_confusion_matrices[protection_selection][
                    #         "subset_replicate"
                    #     ]
                    #     == -1
                    # )
                ][["true_paradigm", "predicted_paradigm", "confusion"]]
                .pivot(
                    index="true_paradigm",
                    columns="predicted_paradigm",
                    values="confusion",
                )
                .loc[
                    paradigm_names[protection_selection],
                    paradigm_names[protection_selection],
                ]
            ),
            cmap="viridis",
            # figsize=(3, 2.6) if protection_selection else (6, 6),
            figsize=(4, 4) if protection_selection else (6, 6),
            xticklabels=paradigm_names[protection_selection],
            yticklabels=paradigm_names[protection_selection],
            # xticklabels=list(paradigm_names.values()),
            # yticklabels=list(paradigm_names.values()),
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Frequency"},
            # dendrogram_ratio=(0.26, 0.36),
            dendrogram_ratio=(0.12, 0.15),
            # row_colors=[paradigm_colours[x] for x in paradigm_names.keys()],
            # col_colors=[paradigm_colours[x] for x in paradigm_names.keys()],
            # row_colors=pd.Series(paradigm_colours),
            # col_colors=pd.Series(paradigm_colours),
            row_colors=module_activations[protection_selection].set_index(
                "paradigm_name"
            ),
            col_colors=module_activations[protection_selection].set_index(
                "paradigm_name"
            ),
            colors_ratio=(0.06, 0.05),
        )
        clustermap.ax_heatmap.set_ylabel("True Paradigm")
        clustermap.ax_heatmap.set_xlabel("Predicted Paradigm")
        clustermap.ax_heatmap.set_xticklabels(
            clustermap.ax_heatmap.get_xticklabels(), rotation=90
        )
        clustermap.ax_row_colors.set_xticklabels(
            clustermap.ax_row_colors.get_xticklabels(), rotation=45, ha="right"
        )
        print(
            f"Mean Confusion Matrix\nDF Percentile Threshold: {df_percentile_threshold}"
        )
        display(clustermap.figure)
        save_dir = f"notebooks/plots/24-11-27_true_data_sim_subsets/mean_confusion_matrices/{'protection_selection' if protection_selection else 'all_paradigms'}"
        os.makedirs(save_dir, exist_ok=True)
        clustermap.savefig(f"{save_dir}/df_threshold_{df_percentile_threshold}.pdf")
        plt.close(clustermap.figure)

# %%
mean_confusion_matrices[True].loc[
    lambda x: x["true_paradigm"] == x["predicted_paradigm"]
].groupby(["df_percentile_threshold"])["confusion"].mean().reset_index().assign(
    df_percentile_threshold=lambda x: x["df_percentile_threshold"].astype(int)
)

# %%
# for each percentile threshold, calculate the accuracy of the classifier - plot with threshold on x axis
confusion_matrices_by_classifier = {
    protection_selection: (
        pd.concat(
            [
                get_confusion_matrix(
                    idt_id,
                    subset_replicate,
                    df_percentile_threshold,
                    protection_selection,
                    omit_svm=True,
                ).assign(
                    subset_replicate=(
                        subset_replicate if subset_replicate is not None else -1
                    ),
                    df_percentile_threshold=str(df_percentile_threshold),
                    subset_size=get_subset_size(idt_id, subset_replicate),
                    protection_selection=protection_selection,
                )
                for subset_replicate in [None]  # , *range(20)]
                for df_percentile_threshold in df_percentile_thresholds
            ]
        )
        .groupby(
            [
                "df_percentile_threshold",
                "classifier_name",
                "true_paradigm",
                "predicted_paradigm",
            ]
        )
        .aggregate(confusion=("confusion", "mean"))
        .reset_index()
    )
    for protection_selection in [False, True]
}
fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), sharex=True, sharey=True)
for i, protection_selection in enumerate([False, True]):
    accuracy_df = (
        confusion_matrices_by_classifier[protection_selection]
        .loc[lambda x: x["true_paradigm"] == x["predicted_paradigm"]]
        .groupby(["df_percentile_threshold", "classifier_name"])["confusion"]
        .mean()
        .reset_index()
        .assign(
            df_percentile_threshold=lambda x: x["df_percentile_threshold"].astype(int)
        )
    )
    sns.lineplot(
        data=accuracy_df,
        x="df_percentile_threshold",
        y="confusion",
        hue="classifier_name",
        # palette=["C0", "C1", "C2"],
        ax=axes[i],
        legend=i == 0,
    )
    sns.scatterplot(
        data=accuracy_df,
        x="df_percentile_threshold",
        y="confusion",
        hue="classifier_name",
        # palette=["C0", "C1", "C2"],
        ax=axes[i],
        s=10,
        legend=False,
    )
    axes[i].set_ylabel("Cross-validation accuracy")
    axes[i].set_xlabel("DF Percentile Threshold")
    axes[i].set_ylim(0, 1)
    axes[i].set_title(
        f"{'All paradigms' if not protection_selection else 'Protection selection'}",
    )
    axes[i].set_xticks(sorted(accuracy_df["df_percentile_threshold"].unique()))

# %%
pd.concat(
    [
        get_confusion_matrix(
            idt_id,
            subset_replicate,
            df_percentile_threshold,
            protection_selection,
            omit_svm=True,
        ).assign(
            subset_replicate=(subset_replicate if subset_replicate is not None else -1),
            df_percentile_threshold=str(df_percentile_threshold),
            subset_size=get_subset_size(idt_id, subset_replicate),
            protection_selection=protection_selection,
        )
        for subset_replicate in [None]  # , *range(20)]
        for df_percentile_threshold in df_percentile_thresholds
    ]
)
