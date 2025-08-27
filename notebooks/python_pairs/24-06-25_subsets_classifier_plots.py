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
df_names = [
    x
    for x in [
        "wasserstein",
        "2D_wasserstein_simplified",
        "smoking_sig_only",
        "z_values",
        "mean_subtracted",
        "mean_subtracted_2D_simplified",
        "mm_larger_weight_sq_diff",
        "mm_larger_weight_abs_diff",
        "mm_dominant_means_sq_diff",
        "mm_larger_means_sq_diff",
        "mm_smaller_means_sq_diff",
        "mm_weighted_means_by_dominance",
        "mm_weighted_means_by_position",
        "total_branch_length",
        # "total_branch_length_squared",
        "branch_length_wasserstein",
        "abs_j_one",
        "l2_j_one",
        "random_control",
        # "sum_control",
    ]
    # if x not in [] # all dfs
    #
]

colour_df_groups = {
    "Reds": [
        "wasserstein",
        "smoking_sig_only",
        "mean_subtracted",
        "z_values",
    ],
    "Greens": [
        # "mm_larger_weight_sq_diff",
        "mm_larger_weight_abs_diff",
        "mm_dominant_means_sq_diff",
        "mm_larger_means_sq_diff",
        "mm_smaller_means_sq_diff",
        "mm_weighted_means_by_dominance",
        "mm_weighted_means_by_position",
    ],
    "Blues": ["branch_length_wasserstein", "abs_j_one"],
    "Greys": ["random_control"],
}
readable_df_names = {
    "wasserstein": "Wasserstein",
    "smoking_sig_only": "Smoking signature (Wasserstein)",
    "mean_subtracted": "Mean-normalised (Wasserstein)",
    "z_values": "Z-normalised (Wasserstein)",
    "mm_larger_weight_abs_diff": "Larger weight",
    "mm_dominant_means_sq_diff": "Dominant mean",
    "mm_larger_means_sq_diff": "More-mutated mean",
    "mm_smaller_means_sq_diff": "Less-mutated mean",
    "mm_weighted_means_by_dominance": "Weighted means by dominance",
    "mm_weighted_means_by_position": "Weighted means by position",
    "branch_length_wasserstein": "Branch lengths (Wasserstein)",
    "abs_j_one": "Tree balance",
    "random_control": "Random",
}
titles = ["Mutational Burden", "Mixture Model", "Phylogenies", "Negative Control"]


def abbreviate_df_name(name):
    return "".join([word[0] for word in name.split("_")])


full_names = {
    abbreviate_df_name(name): name
    for name in df_names
    # for group in colour_df_groups.values()
    # for name in group
}
full_names

# %%
import pandas as pd

subsets_dir = "logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets"
# subsets_dir = "logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets"
# subsets_dir = "logs/idt_2024-04-05_11-17-08/classifiers/first_replicate/protection_selection_paradigms/df_subsets"
# subsets_dir = "logs/idt_2024-04-05_11-17-08/classifiers/first_replicate/1_module_paradigms/df_subsets"
# subsets_dir = "logs/idt_2024-04-05_11-17-08/classifiers/df_subsets"
unfiltered_subsets = sorted(
    list(os.listdir(subsets_dir)), key=lambda x: len(x.split("_"))
)
subsets = [
    subset
    for subset in unfiltered_subsets
    if all([df in full_names for df in subset.split("_")])
]
cross_val_scores = {
    subset: pd.read_csv(f"{subsets_dir}/{subset}/all_simulations/cross_val_scores.csv")
    for subset in subsets
    if os.path.exists(f"{subsets_dir}/{subset}/all_simulations/cross_val_scores.csv")
}

subsets = [subset for subset in subsets if subset in cross_val_scores]
dist_fn_importances = {
    subset: pd.read_csv(
        f"{subsets_dir}/{subset}/all_simulations/dist_fn_importances.csv"
    )
    for subset in subsets
    if os.path.exists(f"{subsets_dir}/{subset}/all_simulations/dist_fn_importances.csv")
}
stresses = {
    subset: pd.read_csv(f"{subsets_dir}/{subset}/all_simulations/stresses.csv")
    for subset in subsets
    if os.path.exists(f"{subsets_dir}/{subset}/all_simulations/stresses.csv")
}
# for subset in subsets:
#     # print out the shape of each dataframe
#     print(
#         f"{subset}:\tcvs {cross_val_scores[subset].shape}\tdfi {dist_fn_importances[subset].shape}\tst {stresses[subset].shape}"
#     )
print(f"cross_val_scores: {list(cross_val_scores[subsets[0]].columns)}")
print(f"dist_fn_importances: {list(dist_fn_importances[subsets[0]].columns)}")
print(f"stresses: {list(stresses[subsets[0]].columns)}")

# %%
# histogram of length of subsets
import matplotlib.pyplot as plt

plt.hist(
    [len(subset.split("_")) for subset in subsets],
    bins=range(1, max([len(subset.split("_")) for subset in subsets]) + 1),
)
plt.xlabel("Number of dfs in subset")
plt.ylabel("Number of subsets")
plt.show()
plt.close()

# %%
# for each key in cross_val_scores, dist_fn_importances and stresses
# if len(key.split(_)) > 4, replace it with '-' followed by the ones it's missing from the full_names dict
include_exclude_cross_val_scores = {}
include_exclude_dist_fn_importances = {}
include_exclude_stresses = {}
for old_dict, new_dict in zip(
    [cross_val_scores, dist_fn_importances, stresses],
    [
        include_exclude_cross_val_scores,
        include_exclude_dist_fn_importances,
        include_exclude_stresses,
    ],
):

    for key in old_dict:
        if len(key.split("_")) > 5:
            new_key = "-" + "_".join(
                name for name in full_names if name not in key.split("_")
            )
            new_dict[new_key] = old_dict[key]
        else:
            new_dict[key] = old_dict[key]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# plot the cross_val_scores as boxplot; y = cross_val_score;x=subset;hue=classifier. Only plot where n_features_per_dist_fn=10
for subset_category in ["exclude", "include"]:
    fig, ax = plt.subplots(figsize=(5, 5))
    cross_val_scores_df = pd.concat(
        [
            include_exclude_cross_val_scores[subset].assign(subset=subset)
            for subset in include_exclude_cross_val_scores
        ]
    )
    top_n = 30
    top_n_subsets = (
        cross_val_scores_df.query("n_features_per_dist_fn == 3")
        .query(
            ("not " if subset_category == "exclude" else "")
            + "subset.str.startswith('-')"
        )
        .sort_values("cross_val_score", ascending=subset_category == "include")
        .head(top_n)
        .subset.unique()
    )
    sns.boxplot(
        data=(
            cross_val_scores_df[
                # cross_val_scores_df["subset"].str.contains("mwmbp")
                # |
                cross_val_scores_df["subset"].isin(top_n_subsets)
            ]
            .query("n_features_per_dist_fn == 3")
            .query("classifier_name == 'random_forest'")
            .sort_values("cross_val_score", ascending=False)
        ),
        y="subset",
        x="cross_val_score",
        ax=ax,
    )
    ax.set_xlim(0, 1)
# grid

# %%
# separate axis for each df_name; x=n, y=frequency of df_name in the top n subsets (by cross_val_score)
import numpy as np
import matplotlib as mpl

subsets_in_order = (
    cross_val_scores_df.query("n_features_per_dist_fn == 3")
    .sort_values("cross_val_score", ascending=False)
    .subset
)

all_groups_fig, all_groups_ax = plt.subplots(figsize=(10, 5))
fig_by_group, ax_by_group = plt.subplots(
    1, len(colour_df_groups), figsize=(5 * len(colour_df_groups), 5)
)
for (colour, df_group), group_ax in zip(colour_df_groups.items(), ax_by_group):
    for i, df_name in enumerate(df_group):
        abbrev_name = abbreviate_df_name(df_name)
        pattern = f"(_{abbrev_name}$|_{abbrev_name}_|^{abbrev_name}_)"
        contains_df_name = subsets_in_order.str.contains(pattern, regex=True)
        cumulative_sum = contains_df_name.cumsum()
        frequency_in_top_n = cumulative_sum / (np.arange(len(subsets_in_order)) + 1)
        for ax in [all_groups_ax, group_ax]:
            ax.plot(
                np.arange(len(subsets_in_order)),
                frequency_in_top_n,
                label=readable_df_names[df_name],
                color=mpl.colormaps[colour](0.25 + 3 * i / (4 * len(df_group))),
            )
    group_ax.set_title(titles.pop(0), fontsize=20)
    group_ax.set_xlabel("n")
    group_ax.set_ylabel("frequency in top n")
    # group_ax.axhline(0.5 , 0, len(subsets_in_order), color="red", linestyle="--")
    group_ax.set_ylim(0, 1)
    group_ax.grid()
    group_ax.legend()

# remove y axis from ax_by_group except for the first one
for ax in ax_by_group[1:]:
    ax.set_yticklabels([])
    ax.set_ylabel("")
fig_by_group.tight_layout()

all_groups_ax.set_title("Frequency of distance function in top n subsets")
all_groups_ax.set_xlabel("n")
all_groups_ax.set_ylabel("frequency in top n")
# all_groups_ax.axhline(0.5, 0, len(subsets_in_order), color="red", linestyle="--")
all_groups_ax.grid()
all_groups_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
all_groups_fig.tight_layout()

# %%
# boxplot of only the single-df classifiers
fig, ax = plt.subplots(figsize=(5, 3))

sns.boxplot(
    data=cross_val_scores_df.query("n_features_per_dist_fn == 3")
    .query("not subset.str.startswith('-')")
    .query("classifier_name == 'random_forest'")
    .assign(n_df=lambda x: x["subset"].apply(lambda x: x.count("_") + 1))
    .query("n_df == 1"),
    x="subset",
    y="cross_val_score",
    ax=ax,
)
# replace x axis labels with full names
ax.set_xticklabels(
    [full_names[x.get_text()] for x in ax.get_xticklabels()], rotation=30, ha="right"
)
ax.set_xlabel("")
ax.set_ylabel("Cross Val Score")
ax.set_ylim(0, 1)

# %%
# fig, axes = plt.subplots(nrows=len(df_names), ncols=1, figsize=(8, 5 * len(df_names)))


# def get_other_df(subset, addition_compare):
#     if subset.count("_") == 0:
#         return None
#     if subset.startswith(f"{addition_compare}_"):
#         return subset.split("_")[1]
#     if subset.endswith(f"_{addition_compare}"):
#         return subset.split("_")[-2]
#     return None


# for i, addition_compare in enumerate([abbreviate_df_name(name) for name in df_names]):
#     ax = axes[i]
#     boxplot_data = (
#         cross_val_scores_df.assign(
#             n_dfs=lambda x: x["subset"].apply(lambda x: x.count("_") + 1),
#         )
#         # .query("classifier_name == 'random_forest'")
#         .query(
#             "n_dfs == 1 or (n_dfs == 2 and (subset.str.startswith("
#             f"'{addition_compare}_') or subset.str.endswith('_{addition_compare}')))"
#         )
#         .assign(
#             other_df=lambda x: x["subset"].apply(
#                 lambda x: get_other_df(x, addition_compare)
#             ),
#             # {other_df}_{addition_compare} if other_df isn't None, otherwise just subset
#             ordered_subset=lambda x: x.apply(
#                 lambda x: (
#                     f"{x['other_df']}_{addition_compare}"
#                     if x["other_df"]
#                     else x["subset"]
#                 ),
#                 axis=1,
#             ),
#             isnt_addition_compare=lambda x: x["subset"].apply(
#                 lambda x: int(x != addition_compare)
#             ),
#         )
#         .sort_values(["isnt_addition_compare", "ordered_subset"])
#     )
#     sns.boxplot(
#         data=boxplot_data,
#         x="ordered_subset",
#         y="cross_val_score",
#         hue="classifier_name",
#         ax=ax,
#     )
#     ax.set_title(f"{full_names[addition_compare]}")
#     ax.set_ylim(0, 1)

# fig.tight_layout()

# %%
# p_values of change in cross_val_score by adding or removing each individual df, within each classifier
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

p_values: dict[bool, pd.DataFrame] = {}

for include in [True, False]:
    max_subset_size = max([len(subset.split("_")) for subset in cross_val_scores])
    cross_val_scores_df = pd.concat(
        [
            cross_val_scores[subset].assign(subset=subset)
            for subset in cross_val_scores
            if (len(subset.split("_")) < 4 and include)
            or (len(subset.split("_")) > max_subset_size - 4 and not include)
        ]
    )  # all included, not like '-ajo_zv' etc

    # add column for each df with True or False
    for df_name in df_names:
        if not any(
            [
                abbreviate_df_name(df_name) in subset
                for subset in cross_val_scores_df["subset"].unique()
            ]
        ):
            continue
        if "subtracted" in df_name:
            continue
        abbreviated_df_name = abbreviate_df_name(df_name)
        cross_val_scores_df[df_name] = cross_val_scores_df["subset"].apply(
            lambda x: (
                x.startswith(f"{abbreviated_df_name}_")
                or x.endswith(f"_{abbreviated_df_name}")
                or f"_{abbreviated_df_name}_" in x
                or x == abbreviated_df_name
            )
        )
    p_values_dict = {
        "df_name": [],
        "classifier_name": [],
        "p_value": [],
        "n": [],
        "mean_fold_change": [],
    }
    for df_name in df_names:
        if not any(
            [
                abbreviate_df_name(df_name) in subset
                for subset in cross_val_scores_df["subset"].unique()
            ]
        ):
            print(f"skipping {df_name},a")
            continue
        if "subtracted" in df_name:
            print(f"skipping {df_name}")
            continue
        if df_name == "mm_smaller_means_sq_diff":
            continue
        for classifier_name in cross_val_scores_df["classifier_name"].unique():
            df_name_mask = cross_val_scores_df[df_name]
            classifier_mask = cross_val_scores_df["classifier_name"] == classifier_name
            df_name_classifier_mask = df_name_mask & classifier_mask
            not_df_name_classifier_mask = ~df_name_mask & classifier_mask
            p_value = ttest_ind(
                cross_val_scores_df.loc[df_name_classifier_mask, "cross_val_score"],
                cross_val_scores_df.loc[not_df_name_classifier_mask, "cross_val_score"],
            ).pvalue
            mean_fold_change = (
                cross_val_scores_df.loc[
                    df_name_classifier_mask, "cross_val_score"
                ].mean()
                / cross_val_scores_df.loc[
                    not_df_name_classifier_mask, "cross_val_score"
                ].mean()
            )
            n = sum(df_name_classifier_mask)
            p_values_dict["df_name"].append(df_name)
            p_values_dict["classifier_name"].append(classifier_name)
            p_values_dict["p_value"].append(p_value)
            p_values_dict["n"].append(n)
            p_values_dict["mean_fold_change"].append(mean_fold_change)

    p_values[include] = pd.DataFrame(p_values_dict).assign(
        # first cap p_value at min float
        p_value=lambda x: x["p_value"].apply(lambda x: max(x, 1e-31)),
        # p_value=lambda x: x["p_value"].apply(lambda x: max(x, np.finfo(float).eps)),
        adjusted_p_value=lambda x: x.groupby("classifier_name")["p_value"].transform(
            lambda x: pd.Series(multipletests(x, method="holm")[1], index=x.index)
        ),
        log2_fold_change=lambda x: x["mean_fold_change"].apply(lambda x: np.log2(x)),
        neg_log10_p_value=lambda x: -np.log10(x["p_value"]),
    )
p_values[True]

# %%
from adjustText import adjust_text
import matplotlib as mpl

# volcano plot: x = log2(fold_change), y = -log10(p_value), color = classifier_name
# classifier_name = "random_forest"
for include in [True, False]:
    for classifier_name in p_values[include]["classifier_name"].unique():
        fig, ax = plt.subplots(figsize=(8, 3.5))

        for colour_map_name, group in colour_df_groups.items():
            subset_data = p_values[include].query(
                f"classifier_name == '{classifier_name}' and df_name in {group}"
            )
            cmap = mpl.colormaps[colour_map_name]
            sns.scatterplot(
                data=subset_data,
                x="log2_fold_change",
                y="neg_log10_p_value",
                hue="df_name",
                # style="classifier_name",
                palette=[
                    cmap(0.25 + 3 * i / (4 * len(group))) for i in range(len(group))
                ],
                ax=ax,
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        bigger_x = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
        ax.set_xlim(-bigger_x, bigger_x)

        p_threshold = 0.05
        log_fold_change_threshold = 0.05

        ax.axhline(-np.log10(p_threshold), color="red", linestyle="--")
        # ax.axvline(log_fold_change_threshold, color="red", linestyle="--")
        # ax.axvline(-log_fold_change_threshold, color="red", linestyle="--")
        # at 0 instead
        ax.axvline(0, color="red", linestyle="--")

        # ax.set_title("Effect of adding distance metric to classifier on x-val score")

        # point_labels = []
        # for i, row in p_values[include].iterrows():
        #     if (
        #         (row["p_value"] < p_threshold)
        #         # and (abs(row["log2_fold_change"]) > log_fold_change_threshold)
        #         and (row["classifier_name"] == classifier_name)
        #         and any(
        #             [row["df_name"] in group for group in colour_df_groups.values()]
        #         )
        #     ):
        #         point_labels.append(
        #             ax.text(
        #                 row["log2_fold_change"],
        #                 -np.log10(row["p_value"]),
        #                 row["df_name"],
        #                 fontsize=8,
        #             )
        #         )
        # adjust_text(
        #     point_labels,
        #     ax=ax,
        #     arrowprops={"arrowstyle": "->", "lw": 0.5},
        #     force_explode=(0.2, 1.0),
        # )
        print()
        fig.tight_layout()
        savedir = "/Users/hughselway/Documents/ClonesModelling/notebooks/plots/24-06-25_subsets_classifier_plots"
        os.makedirs(savedir, exist_ok=True)
        fig.savefig(f"{savedir}/volcano_plot_adding.pdf")
        print(f"include: {include}, classifier_name: {classifier_name}")
        display(fig)
        plt.close(fig)

# %%
pd.get_dummies(
    cross_val_scores_df.loc[:, ["n_features_per_dist_fn", "classifier_name", df_name]],
    columns=["n_features_per_dist_fn", "classifier_name"],
)

# %%
# repeat p-value analysis but with linear model and incorporating all classifiers
import numpy as np
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

p_values: dict[bool, pd.DataFrame] = {}

max_subset_size = max([len(subset.split("_")) for subset in cross_val_scores])

classifiers = list(cross_val_scores_df["classifier_name"].unique())

for include in [True, False]:
    cross_val_scores_df = pd.concat(
        [
            cross_val_scores[subset].assign(subset=subset)
            for subset in cross_val_scores
            if (len(subset.split("_")) < 4 and include)
            or (len(subset.split("_")) > max_subset_size - 4 and not include)
        ]
    )  # all included, not like '-ajo_zv' etc

    # add column for each df with True or False
    for df_name in df_names:
        if not any(
            [
                abbreviate_df_name(df_name) in subset
                for subset in cross_val_scores_df["subset"].unique()
            ]
        ):
            continue
        # if "subtracted" in df_name:
        #     continue
        abbreviated_df_name = abbreviate_df_name(df_name)
        cross_val_scores_df[df_name] = cross_val_scores_df["subset"].apply(
            lambda x: (
                x.startswith(f"{abbreviated_df_name}_")
                or x.endswith(f"_{abbreviated_df_name}")
                or f"_{abbreviated_df_name}_" in x
                or x == abbreviated_df_name
            )
        )

    # linear model df_name ~ classifier + n_features_per_dist_fn
    p_values_dict = {
        "df_name": [],
        "p_value": [],
        "mean_fold_change": [],
    }
    for df_name in df_names:
        if not any(
            [
                abbreviate_df_name(df_name) in subset
                for subset in cross_val_scores_df["subset"].unique()
            ]
        ):
            print(f"skipping {df_name},a")
            continue
        # if "subtracted" in df_name:
        #     print(f"skipping {df_name}")
        #     continue
        # if df_name == "mm_smaller_means_sq_diff":
        #     continue
        df_name_mask = cross_val_scores_df[df_name]
        not_df_name_mask = ~df_name_mask
        X = pd.get_dummies(
            cross_val_scores_df.loc[
                :, ["n_features_per_dist_fn", "classifier_name", df_name]
            ],
            columns=["n_features_per_dist_fn", "classifier_name"],
        ).assign(
            **{
                df_name: cross_val_scores_df[df_name].astype(int),
            },
        )

        X = sm.add_constant(X)
        y = cross_val_scores_df.loc[:, "cross_val_score"]
        model = sm.OLS(y, X).fit()
        # print(model.summary())
        # break
        p_value = model.pvalues[df_name]
        mean_fold_change = (
            cross_val_scores_df.loc[df_name_mask, "cross_val_score"].mean()
            / cross_val_scores_df.loc[not_df_name_mask, "cross_val_score"].mean()
        )
        p_values_dict["df_name"].append(df_name)
        p_values_dict["p_value"].append(p_value)
        p_values_dict["mean_fold_change"].append(mean_fold_change)

    p_values[include] = pd.DataFrame(p_values_dict).assign(
        # first cap p_value at min float
        p_value=lambda x: x["p_value"].apply(lambda x: max(x, np.finfo(float).tiny)),
        # p_value=lambda x: x["p_value"].apply(lambda x: max(x, np.finfo(float).eps)),
        adjusted_p_value=lambda x: multipletests(x["p_value"], method="holm")[1],
        log2_fold_change=lambda x: x["mean_fold_change"].apply(lambda x: np.log2(x)),
        neg_log10_p_value=lambda x: -np.log10(x["p_value"]),
    )
p_values[True]

# %%
# volcano plot: x = log2(fold_change), y = -log10(p_value), colour by df_name, separate plot for include and exclude

fig, axes = plt.subplots(3, 1, figsize=(8, 7), height_ratios=[15, 15, 1])
for include, ax in zip([True, False], axes[:2]):
    for colour_map_name, group in colour_df_groups.items():
        subset_data = p_values[include].query(f"df_name in {group}")
        cmap = mpl.colormaps[colour_map_name]
        sns.scatterplot(
            data=subset_data.assign(
                readable_df_name=lambda x: x["df_name"].apply(
                    lambda y: readable_df_names.get(y, y)
                )
            ),
            x="log2_fold_change",
            y="neg_log10_p_value",
            hue="readable_df_name",
            # style="classifier_name",
            palette=[cmap(0.25 + 3 * i / (4 * len(group))) for i in range(len(group))],
            ax=ax,
            s=25,
        )

    if not include:
        ax.legend(
            bbox_to_anchor=(0.5, -0.7),
            loc="upper center",
            title="Distance Function",
            ncol=2,
        )
    else:
        ax.legend().remove()

    bigger_x = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-bigger_x, bigger_x)

    p_threshold = 0.05
    log_fold_change_threshold = 0.05

    ax.axhline(-np.log10(p_threshold), color="red", linestyle="--")
    # ax.axvline(log_fold_change_threshold, color="red", linestyle="--")
    # ax.axvline(-log_fold_change_threshold, color="red", linestyle="--")
    # at 0 instead
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("log2(Fold Change)")
    ax.set_ylabel("-log10(p value)")
    ax.set_title(
        "Addition (at most 3 distance functions)"
        if include
        else "Subtraction (at least 15 distance functions)"
    )
    ax.text(
        0.5 * ax.get_xlim()[1],
        -np.log10(p_threshold),
        "p = 0.05",
        color="red",
        ha="right",
        va="bottom",
    )

axes[2].arrow(-0.01, 0.1, -0.19, 0, head_width=0.1, head_length=0.02, fc="k", ec="k")
axes[2].arrow(0.01, 0.1, 0.19, 0, head_width=0.1, head_length=0.02, fc="k", ec="k")
axes[2].text(-0.01, 0, "Reduces Accuracy", fontsize=10, ha="right")
axes[2].text(0.01, 0, "Increases Accuracy", fontsize=10, ha="left")

# no frame
axes[2].set_frame_on(False)
# no yticks or labels
axes[2].set_yticks([])
axes[2].set_yticklabels([])
axes[2].set_xticks([])
axes[2].set_xticklabels([])


# fig.tight_layout()
# fig.subplots_adjust(hspace=0.8)
# instead manually adjust
axes[0].set_position([0.1, 0.75, 0.775, 0.18])
axes[1].set_position([0.1, 0.45, 0.775, 0.18])
axes[2].set_position([0.1, 0.35, 0.775, 0.03])
savedir = "/Users/hughselway/Documents/ClonesModelling/notebooks/plots/24-06-25_subsets_classifier_plots"
os.makedirs(savedir, exist_ok=True)
fig.savefig(f"{savedir}/volcano_plot_linear.pdf")
display(fig)
plt.close(fig)

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

for include, ax in zip([False, True], axes):
    sns.barplot(
        data=p_values[include],
        y="df_name",
        x="mean_fold_change",
        ax=ax,
        order=[
            df_name
            for group in colour_df_groups.values()
            for df_name in sorted(
                group,
                key=lambda x: (
                    p_values[True].query(f"df_name == '{x}'")["log2_fold_change"].mean()
                ),
                reverse=True,
            )
        ],
    )
    ax.axvline(1, color="black", linestyle="--")
    # add vlines between each group
    y_pos = 0
    for group in list(colour_df_groups.values())[:-1]:
        y_pos += len(group)
        ax.axhline(y_pos - 0.5, color="black", linestyle="--")
    ax.set_ylabel("Simulation Comparison Metric")

    # annotate with * if adjusted p-value < 0.05
    plotted_dfs = ax.yaxis.get_majorticklabels()
    for label in plotted_dfs:
        df_name = label.get_text()
        print(f"Checking {df_name} for p-value")
        if df_name in p_values[include]["df_name"].values:
            p_value = (
                p_values[include]
                .query(f"df_name == '{df_name}'")["adjusted_p_value"]
                .values[0]
            )
            print(f"{df_name}: {p_value}")
            if p_value < 0.05:
                label.set_text(label.get_text() + "*")
                label.set_color("red")
            else:
                label.set_text(label.get_text() + " ")
                label.set_color("black")

    if not include:
        ax.set_yticklabels(
            [
                readable_df_names.get(label.get_text(), label.get_text())
                for label in ax.get_yticklabels()
            ],
        )

axes[0].set_xlabel(
    "Effect of removing distance function\non cross-val score (fold change)"
)
axes[1].set_xlabel(
    "Effect of adding distance function\non cross-val score (fold change)"
)
axes[0].set_xlim(*axes[1].get_xlim())
axes[0].invert_xaxis()

fig.tight_layout()
savedir = "notebooks/plots/24-06-25_subsets_classifier_plots"
os.makedirs(savedir, exist_ok=True)
fig.savefig(f"{savedir}/bar_plot_fold_change.pdf")
display(fig)
plt.close(fig)

# %%
p_values[True].assign(
    percentage_effect=lambda x: x["mean_fold_change"].apply(lambda y: (y - 1) * 100),
)
