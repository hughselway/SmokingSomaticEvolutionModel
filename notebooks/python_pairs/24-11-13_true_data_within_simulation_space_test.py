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
from ClonesModelling.id_test.classifier.generate_distance_matrix import read_from_file
from ClonesModelling.id_test.classifier.indexer import Indexer
from ClonesModelling.id_test.calculate_distances import DISTANCE_FUNCTIONS

# idt_id = "idt_2025-02-05_14-53-34"
idt_id = "idt_2025-04-07_10-16-59"
simulation_replicates_per_paradigm = 300

dm = read_from_file(
    f"logs/{idt_id}/classifiers/all_pts/including_true_data",
    Indexer(
        False,
        5,
        simulation_replicates_per_paradigm,
        True,
        [
            df_name
            for df_name in DISTANCE_FUNCTIONS
            if df_name
            not in [
                "total_branch_length_squared",
                "sum_control",
                "zero_control",
                # "mean_subtracted_2D_simplified",
                # "l2_j_one",
                "2D_wasserstein",
                # ## two DFs removed because they didn't map properly in the idt_2025-02-05_14-53-34 run
                # "mean_subtracted",
                # "total_branch_length",
                ## extra DFs removed for true data comparison analysis - no, don't need to exclude here!
                # "random_control",
            ]
            # and "control" not in df_name
        ],
    ),
)
print(dm.distance_matrix.shape)
print(dm.true_data_distance.shape)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

min_max_ratio = {}
min_max_ratio_robust = {}
mean_ratio = {}

for df_index, df_name in enumerate(dm.indexer.distance_function_names):
    if df_name == "sum_control":
        continue
    assert dm.true_data_distance is not None
    min_max_ratio[df_name] = (
        dm.true_data_distance[df_index].min() / dm.distance_matrix[df_index].max()
    )
    for percentile_threshold in [0, 10, 20, 25]:
        if percentile_threshold not in min_max_ratio_robust:
            min_max_ratio_robust[percentile_threshold] = {}
        min_max_ratio_robust[percentile_threshold][df_name] = np.percentile(
            dm.true_data_distance[df_index], percentile_threshold
        ) / np.percentile(
            dm.distance_matrix[df_index][dm.distance_matrix[df_index] > 0],
            100 - percentile_threshold,
        )

    mean_ratio[df_name] = (
        dm.true_data_distance[df_index].mean() / dm.distance_matrix[df_index].mean()
    )
    print(
        df_name,
        "\t",
        min_max_ratio[df_name],
        "\t",
        mean_ratio[df_name],
        "\t",
        "\t".join(
            [
                f"{min_max_ratio_robust[percentile][df_name]:.2f}"
                for percentile in min_max_ratio_robust
            ]
        ),
    )

# barplots
for data, ylabel, name in [
    (
        min_max_ratio,
        "minimum true data distance /\n max between-simulation distance",
        "min_max_ratio",
    ),
    (
        mean_ratio,
        "mean true data distance /\n mean between-simulation distance",
        "mean_ratio",
    ),
]:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(x=list(data.keys()), y=list(data.values()), ax=ax)
    # if name == "mean_ratio":
    ax.set_yscale("log")
    ax.axhline(1, color="r", linestyle="--")
    # grid on
    ax.yaxis.grid(True)
    ax.set_ylabel(ylabel)
    # if "minimum" in ylabel:
    #     ax.annotate(
    #         "True data further from \nnearest neighbour than\nspan of all simulations",
    #         xy=(0, 0.4),
    #         xytext=(0, 10),
    #         textcoords="offset points",
    #         ha="left",
    #         va="bottom",
    #         color="r",
    #     )
    plt.xticks(rotation=45, ha="right")
    save_dir = "notebooks/plots/24-11-13_true_data_within_simulation_space_test"
    fig.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{name}.pdf")
    print(
        f"{name}: dfs > 1: {[df for df, ratio in data.items() if ratio > 1]} (<1: {[df for df, ratio in data.items() if ratio < 1]})"
    )
    display(fig)
    plt.close(fig)

# %%
# # plot the robust ratios as 3 rows, sharex
# fig, axs = plt.subplots(3, 1, figsize=(5, 6.5), sharex=True)
# for i, (data, ylabel) in enumerate(
#     [
#         (min_max_ratio_robust_5, "5th percentile ratio"),
#         (min_max_ratio_robust_10, "10th percentile ratio"),
#         (min_max_ratio_robust_25, "25th percentile ratio"),
#     ]
# ):
#     sns.barplot(x=list(data.keys()), y=list(data.values()), ax=axs[i])
#     axs[i].set_yscale("log")
#     axs[i].axhline(1, color="r", linestyle="--")
#     axs[i].yaxis.grid(True)
#     axs[i].set_ylabel(ylabel)
#     plt.xticks(rotation=45, ha="right")
# fig.tight_layout()
# plt.savefig(f"{save_dir}/min_max_ratio_robust.pdf")
# display(fig)
# plt.close(fig)

# %%
import pandas as pd

# instead make the percentile the hue
fig, ax = plt.subplots(figsize=(6.5, 4))
data = []
for percentile_threshold in min_max_ratio_robust:
    for df_name, ratio in min_max_ratio_robust[percentile_threshold].items():
        data.append(
            {"df_name": df_name, "ratio": ratio, "percentile": percentile_threshold}
        )
data = pd.DataFrame(data)
sns.barplot(x="df_name", y="ratio", hue="percentile", data=data, ax=ax)
ax.set_yscale("log")
ax.axhline(1, color="r", linestyle="--")
ax.yaxis.grid(True)
ax.set_ylabel("Observed Data\nExtremity Factor")
ax.legend(title="Percentile\nTolerance", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_xlabel("Distance function")
plt.xticks(rotation=35, ha="right")
fig.tight_layout()
plt.savefig(f"{save_dir}/true_data_extremity.pdf")
print(f"{save_dir}/true_data_extremity.pdf")
display(fig)
plt.close(fig)

# %%
np.percentile(dm.distance_matrix[df_index][dm.distance_matrix[df_index] > 0], 0.1)

# %%
# for each distance function, calculate the percentile threshold above which the true data is considered an outlier
# use fsolve to find percentile at which extremity is greater than 1
from scipy.optimize import fsolve


def get_extremity_factor(
    percentile_threshold: float,
    df_index: int,
    true_data_distance: np.ndarray,
    distance_matrix: np.ndarray,
):
    if np.isnan(percentile_threshold):
        return np.nan
    if percentile_threshold < 0:
        return 10000 * -percentile_threshold
    if percentile_threshold > 1:
        return 10000 * (percentile_threshold - 1)
    obs_data_from_sims = np.percentile(
        true_data_distance[df_index], 100 * percentile_threshold
    )
    sims_span = np.percentile(
        distance_matrix[df_index][distance_matrix[df_index] > 0],
        100 * (1 - percentile_threshold),
    )
    # print(
    #     percentile_threshold,
    #     obs_data_from_sims,
    #     sims_span,
    #     obs_data_from_sims / sims_span,
    # )
    return obs_data_from_sims / sims_span


percentile_thresholds = {}
for df_index, df_name in enumerate(dm.indexer.distance_function_names):
    if df_name in [
        "sum_control",
        "l2_j_one",
        "total_branch_length_squared",
        "total_branch_length",
        "mm_larger_weight_sq_diff",
    ]:
        continue
    zero_count = np.count_nonzero(dm.distance_matrix[df_index] <= 0)
    total_count = dm.distance_matrix[df_index].size
    print(
        "omitting",
        zero_count,
        "zero values from",
        df_name,
        "out of",
        total_count,
        "total values",
        f"({zero_count / total_count:.4%} of values)",
    )
    # find the percentile threshold at which the extremity factor is equal to 1
    percentile_threshold = fsolve(
        lambda x: get_extremity_factor(
            x[0], df_index, dm.true_data_distance, dm.distance_matrix
        )
        - 1,
        0.05,  # initial guess
    )[0]
    percentile_thresholds[df_name] = percentile_threshold
    print(f"{df_name}: {percentile_threshold:.4f}")

# %%
# plot x = threshold, y = number of paradigms with a greater threshold
fig, ax = plt.subplots(figsize=(6, 3.5))
x = np.linspace(0, 0.5, 10000)
y = [
    sum(
        percentile_thresholds[df_name] > percentile for df_name in percentile_thresholds
    )
    for percentile in x
]
sns.lineplot(x=x, y=y, ax=ax)
ax.set_xlabel("Observed data extremity threshold")
ax.set_ylabel("Number of Distance Functions\nwith Greater Threshold")

ordered_dfs = sorted(percentile_thresholds.items(), key=lambda x: x[1], reverse=True)
for i, (df_name, threshold) in enumerate(ordered_dfs):
    ax.text(0.55, i + 0.2, df_name, ha="left", va="bottom", fontsize=8)
    # if threshold < 0.1:
    #     ax.text(threshold + 0.01, i + 0.2, df_name, ha="left", va="bottom", fontsize=8)
    # else:
    #     ax.text(threshold - 0.01, i + 0.2, df_name, ha="right", va="bottom", fontsize=8)

for threshold in [0.3, 0.36, 0.42]:
    ax.axvline(threshold, linestyle="--", color="red", alpha=0.4)
    n_remaining = sum(
        percentile_thresholds[df_name] > threshold for df_name in percentile_thresholds
    )
    # ax.text(
    #     threshold + 0.003,
    #     14.5,
    #     f"{n_remaining} DFs",
    #     ha="left",
    #     va="center",
    #     fontsize=8,
    # )
save_dir = "notebooks/plots/24-11-13_true_data_within_simulation_space_test"
fig.tight_layout()
plt.savefig(f"{save_dir}/true_data_extremity_thresholds.pdf")
print(f"{save_dir}/true_data_extremity_thresholds.pdf")
display(fig)
plt.close(fig)

# %%
# print the distance functions that have ratio < 1 for each of the robust ratios
# for percentile, data in [
#     (5, min_max_ratio_robust_5),
#     (10, min_max_ratio_robust_10),
#     (25, min_max_ratio_robust_25),
# ]:
for percentile_threshold, data in min_max_ratio_robust.items():
    print(
        f"{percentile_threshold}th percentile ratio <1:\n"
        f"incl: {', '.join([df.replace('_',' ') for df, ratio in data.items() if ratio < 1])}\n"
        f"incl: {'_'.join(sorted([''.join(x[0] for x in df.split('_')) for df, ratio in data.items() if ratio < 1]))}\n"
        f"excl: {', '.join([df for df, ratio in data.items() if ratio > 1])}\n"
        f"excl: {'_'.join(sorted([''.join(x[0] for x in df.split('_')) for df, ratio in data.items() if ratio > 1]))}"
    )

# %%
# import pandas as pd
# import numpy as np

# for df_index, df_name in enumerate(dm.indexer.distance_function_names):
#     # histogram of distances
#     fig, ax = plt.subplots(1, 2, figsize=(7, 3), sharex=True)
#     ax[0].set_xscale("log")
#     # pd.Series(dm.distance_matrix[df_index].flatten()).replace(0, np.nan).dropna().hist(ax=ax[0], bins=100)
#     # pd.Series(dm.true_data_distance[df_index].flatten()).replace(0, np.nan).dropna().hist(ax=ax[1], bins=100)
#     sns.histplot(
#         pd.Series(dm.distance_matrix[df_index].flatten()).replace(0, np.nan).dropna(),
#         ax=ax[0],
#         bins=100,
#         # log_scale=True,
#     )
#     sns.histplot(
#         pd.Series(dm.true_data_distance[df_index].flatten())
#         .replace(0, np.nan)
#         .dropna(),
#         ax=ax[1],
#         bins=100,
#     )
#     fig.suptitle(f"{df_name}")
#     ax[0].set_title(f"Between simulations")
#     ax[1].set_title(f"Between true data and simulations")
#     fig.tight_layout()
#     # save_dir = "notebooks/plots/24-11-13_true_data_within_simulation_space_test/histograms"
#     os.makedirs(f"{save_dir}/histograms", exist_ok=True)
#     fig.savefig(f"{save_dir}/histograms/{df_name}.pdf")
#     plt.show()
#     plt.close()
