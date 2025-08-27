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

subset_distance_function_indices: list[np.ndarray] = []
greedy_search_df_vifs: list[np.ndarray] = []
# data_dir = "logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv/all_simulations/independent_distance_functions"
# data_dir = "logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_tbls_w_zv/all_simulations/independent_distance_functions"
data_dir = "logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv/all_simulations/independent_distance_functions/vif_threshold_5"

for file in sorted(
    os.listdir(data_dir), key=lambda x: int(x.split(".")[0].split("_")[-1])
):
    print(file)
    with np.load(f"{data_dir}/{file}") as data:
        subset_distance_function_indices.append(data["distance_function_indices"])
        greedy_search_df_vifs.append(data["greedy_search_df_vifs"])

# %%
from ClonesModelling.id_test.calculate_distances import DISTANCE_FUNCTIONS

distance_function_names = [
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
        ]
    )
]
distance_function_names

# %%
len(distance_function_names), len(
    "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv".split(
        "_"
    )
)

# %%
colour_df_groups = {
    "Reds": [
        "wasserstein",
        "smoking_sig_only",
        "z_values",
        "mean_subtracted",
        "mean_subtracted_2D_simplified",
    ],
    "Greens": [
        "mm_larger_weight_sq_diff",
        "mm_larger_weight_abs_diff",
        "mm_dominant_means_sq_diff",
        "mm_larger_means_sq_diff",
        "mm_smaller_means_sq_diff",
        "mm_weighted_means_by_dominance",
        "mm_weighted_means_by_position",
    ],
    "Blues": [
        "total_branch_length",
        "branch_length_wasserstein",
        "abs_j_one",
        "l2_j_one",
    ],
}

# %%
import pandas as pd

vif_by_iteration = pd.DataFrame(
    (
        {
            "iteration": i,
            "vif": iteration_vifs[df_index_position],
            "distance_function_index": distance_function_index,
            "distance_function_name": distance_function_names[distance_function_index],
        }
        for i, (iteration_df_indices, iteration_vifs) in enumerate(
            zip(subset_distance_function_indices, greedy_search_df_vifs)
        )
        for df_index_position, distance_function_index in enumerate(
            iteration_df_indices
        )
    )
)
vif_by_iteration

# %%
# get first iteration where all  vifs are less than 5,10
thresholds = [20,5]
threshold_iterations = {
    threshold: vif_by_iteration[vif_by_iteration["vif"] >= threshold]["iteration"].max()
    + 1
    for threshold in thresholds
}
print(threshold_iterations)
threshold_distance_functions = {
    # list of distance functions in the relevant iteration
    threshold: list(
        vif_by_iteration[
            vif_by_iteration["iteration"] == threshold_iterations[threshold]
        ]["distance_function_name"].unique()
    )
    for threshold in thresholds
}
print(', '.join(threshold_distance_functions[20]).replace("_", "\_"))
print(', '.join(threshold_distance_functions[5]).replace("_", "\_"))

# %%
# for each iteration, print the distance functions in order of VIF
removed: list[str] = []
reduced: list[dict[str, float]] = []
for iteration, iteration_df in vif_by_iteration.groupby("iteration"):
    next_iteration = int(iteration) + 1
    if next_iteration not in vif_by_iteration.iteration.values:
        break
    print(iteration, iteration_df.distance_function_index.values)
    removed.append(
        iteration_df.sort_values("vif", ascending=False).distance_function_name.values[
            0
        ]
    )
    print(iteration, "removed", removed[-1], "with VIF", iteration_df.vif.max())
    # find the difference between each remaining distance function's VIF in iteration vs iteration + 1
    # rank the distance functions by this difference, and print the ones that have the largest reduction
    next_iteration_df = vif_by_iteration[vif_by_iteration.iteration == next_iteration]
    assert all(
        df in iteration_df.distance_function_index.values
        for df in next_iteration_df.distance_function_index.values
    ), (
        iteration,
        next_iteration,
        iteration_df.distance_function_index.values,
        next_iteration_df.distance_function_index.values,
    )
    vif_diff = (
        iteration_df.loc[iteration_df.distance_function_name != removed[-1]]
        .set_index("distance_function_index")
        .vif
        - next_iteration_df.set_index("distance_function_index").vif
    )
    reduced.append(
        {
            df_name: reduction
            for df_name, reduction in map(
                lambda x: ((distance_function_names[x], vif_diff[x])),
                vif_diff.sort_values(ascending=False).loc[lambda x: x > 1].index,
            )
        }
    )

    # make  it have columns distance_function_name and vif_diff, sort, take top 5, and print
    print(
        "\t"
        + "\n\t".join(
            map(
                lambda x: (f"{distance_function_names[x]} fell by {vif_diff[x]}"),
                vif_diff.sort_values(ascending=False).loc[lambda x: x > 1].index,
            )
        )
    )

# %%
vif_by_iteration.loc[vif_by_iteration["distance_function_name"] == "l2_j_one",]

# %%
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), height_ratios=[1, 4.5], sharex=True)

colour_palette = {}
for colour, dfs in colour_df_groups.items():
    palette = sns.color_palette(colour, n_colors=len(dfs) + 1)
    for df_index, df in enumerate(dfs):
        colour_palette[df] = palette[df_index + 1]

# sns.color_palette(colour, n_colors=len(dfs))

## Upper axis, dotplot with y-ax values "eliminated distance function", "beneficiaries" (from removed and reduced dicts defined above)
separation = 3
reduced_step_size = 2
padding = 1.6
for iteration, (eliminated_df, reduced_dfs_dict) in enumerate(zip(removed, reduced)):
    axes[0].scatter([iteration], [0], color=colour_palette[eliminated_df], s=150)
    axes[0].scatter(
        [iteration for _ in reduced_dfs_dict],
        list(
            range(
                separation,
                separation + reduced_step_size * len(reduced_dfs_dict),
                reduced_step_size,
            )
        ),
        c=[colour_palette[df_name] for df_name in reduced_dfs_dict],
        s=[math.log10(reduction) * 50 for reduction in reduced_dfs_dict.values()],
    )
for side in ["top", "right", "left", "bottom"]:
    axes[0].spines[side].set_visible(False)
axes[0].set_ylim(-padding, separation + 3 * reduced_step_size - 2 + padding)
axes[0].set_yticks([0, separation + 1])
axes[0].set_yticklabels(["Eliminated distance function", "VIF reduced > 1 by removal"])
axes[0].tick_params(axis="x", length=0)
axes[0].tick_params(axis="y", length=0)
axes[0].axhline(separation / 2, linestyle="--", alpha=0.25, color="black")


# dummy scatter for legend
for reduction in [3, 10, 30, 100, 300, 1000]:
    # for reduction in [2, 5, 10, 20, 50, 100]:
    axes[0].scatter(
        [],
        [],
        c="gray",
        alpha=0.5,
        s=math.log10(reduction) * 50,
        label=f"{reduction}",
    )
axes[0].legend(
    bbox_to_anchor=(-0.775, 0.45),
    loc="center left",
    title="VIF reduction",
    ncol=2,
    columnspacing=0.8
)
fig.subplots_adjust(left=0.4)

# axes[0].yaxis.tick_right()
# axes[0].yaxis.set_label_position("right")

## Lower axis, lineplot
for colour, dfs in colour_df_groups.items():
    sns.lineplot(
        data=vif_by_iteration[vif_by_iteration["distance_function_name"].isin(dfs)],
        x="iteration",
        y="vif",
        hue="distance_function_name",
        ax=axes[1],
        palette=colour_palette,
    )
axes[1].set_yscale("log")
axes[1].legend(
    loc="center right", bbox_to_anchor=(-0.12, 0.45), title="Distance Function"
)
# axes[1].axes[1]hline(5, color="black", linestyle="--", alpha=0.5)
for threshold in [5, 20]:
    above_last_iteration_above_threshold = (
        0.5 + vif_by_iteration[vif_by_iteration["vif"] > threshold]["iteration"].max()
    )
    xmin, xmax = axes[1].get_xlim()
    axes[1].axhline(
        threshold,
        color="black",
        linestyle="--",
        alpha=0.5,
        xmax=(above_last_iteration_above_threshold - xmin) / (xmax - xmin),
    )
    ymin, ymax = axes[1].get_ylim()
    axes[1].axvline(
        above_last_iteration_above_threshold,
        color="black",
        linestyle="--",
        alpha=0.5,
        ymax=(np.log10(threshold) - np.log10(ymin)) / (np.log10(ymax) - np.log10(ymin)),
    )
    axes[1].text(
        above_last_iteration_above_threshold + 0.1,
        threshold,
        f"VIF={threshold}",
        va="bottom",
        ha="left",
    )
axes[1].set_ylabel("Variance Inflation Factor")
axes[1].set_xlabel("Greedy Search Iteration")
for side in ["top", "right"]:
    axes[1].spines[side].set_visible(False)

fig.tight_layout()
fig.subplots_adjust(hspace=0)
save_dir = "notebooks/plots/24-11-15_plot_vif_greedy_search"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(f"{save_dir}/vif_by_df_by_iteration.pdf")
