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
# make dataclass IDT
from dataclasses import dataclass
import pandas as pd


@dataclass
class IDT:
    id: str

    def __post_init__(self):
        assert os.path.isdir(f"logs/{self.id}"), "IDT not found"

    @property
    def distance_functions(self):
        return [
            x
            for x in os.listdir(f"logs/{self.id}/distance")
            if os.path.isdir(f"logs/{self.id}/distance/{x}")
            and x != "hpc"
            and "bug" not in x
            and x not in ["sum_control", "total_branch_length_squared", "mixture_model"]
        ]

    def read_df_pairwise_distances(
        self,
        df_name: str,
        ignore_replicate_indices: bool,
        normalisation: str | None,
        subset_size: int | None,
    ) -> pd.DataFrame:
        print(df_name)
        distances = pd.read_csv(
            f"logs/{self.id}/distance/{df_name}/pairwise_distances_subsampled.csv",
            usecols=["total_distance", "similarity_level"]
            + (["replicate_1", "replicate_2"] if not ignore_replicate_indices else []),
            # f"logs/{self.id}/distance/{df_name}/pairwise_distances.csv",
            # usecols=["distance", "similarity_level"]
            # + (["replicate_1", "replicate_2"] if not ignore_replicate_indices else []),
        ).rename(columns={"total_distance": "distance"})
        if subset_size is not None:
            distances = distances.sample(subset_size, random_state=0)

        if normalisation in ["replicate_mean", "intra-paradigm_mean"]:
            norm_sim_level = (
                "REPLICATE" if normalisation == "replicate_mean" else "INTRA_PARADIGM"
            )
            replicate_mean_distance = distances[
                distances.similarity_level == norm_sim_level
            ].distance.mean()
            distances = (
                distances.loc[distances.similarity_level != norm_sim_level, :]
                .assign(
                    normalised_distance=lambda x: x.distance / replicate_mean_distance
                )
                .drop(columns=["distance"])
            )
        elif normalisation in ["overall_mean", "overall_mean_and_std"]:
            distances = (
                distances.assign(
                    normalised_distance=(
                        (lambda x: x.distance / x.distance.mean())
                        if normalisation == "overall_mean"
                        else (
                            lambda x: x.distance / x.distance.mean() / x.distance.std()
                        )
                    )
                )
                .reset_index(drop=True)
                .drop(columns=["distance"])
            )
        elif normalisation:
            assert normalisation == "range" or normalisation.startswith(
                "robust_range"
            ), f"normalisation {normalisation} not recognised"
            range_size = (
                (distances.distance.max() - distances.distance.min())
                if "robust" not in normalisation
                else (
                    distances.distance.quantile(
                        1 - int(normalisation.split("_")[-1]) / 200
                    )
                    - distances.distance.quantile(
                        int(normalisation.split("_")[-1]) / 200
                    )
                )
            )
            distances = (
                distances.assign(normalised_distance=lambda x: x.distance / range_size)
                .reset_index(drop=True)
                .drop(columns=["distance"])
            )
        return distances

    def read_pairwise_distances(
        self,
        ignore_replicate_indices: bool,
        normalisation: str | None,
        subset_size: int | None,
    ) -> pd.DataFrame:
        return pd.concat(
            [
                self.read_df_pairwise_distances(
                    x, ignore_replicate_indices, normalisation, subset_size=subset_size
                ).assign(distance_function=x)
                for x in self.distance_functions
            ]
        ).reset_index(drop=True)


# %%
# idt_id = "idt_2024-04-05_11-17-08"
# idt_id = "idt_2024-10-03_19-04-47"
idt_id = "idt_2025-04-04_14-30-05"

idt = IDT(idt_id)

# %%
pairwise_distances = idt.read_pairwise_distances(True, "replicate_mean", int(1e5))
# pairwise_distances = idt.read_pairwise_distances(True, "intra-paradigm_mean", int(1e5))
pairwise_distances

# %%
# get the ordering of distance functions by mean normalised distance
distance_function_order = (
    pairwise_distances.groupby("distance_function")
    .normalised_distance.median()
    .sort_values(ascending=False)
    .index
)

# %%
pairwise_distances.groupby(
    "distance_function"
).normalised_distance.median().sort_values(ascending=False)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

if True:#not os.path.exists(
#     f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_sim_level.pdf"
# ):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=pairwise_distances,
        x="distance_function",
        y="normalised_distance",
        hue="similarity_level",
        order=distance_function_order,
        ax=ax,
        log_scale=True,
        showfliers=False,
    )
    plt.xticks(rotation=45, ha="right")
    # ax.set_title("Distance value, normalised by mean replicate distance")
    ax.set_ylabel("Normalised distance")
    ax.set_xlabel("Distance function")
    ax.axhline(1, color="black", linestyle="--")
    fig.tight_layout()
    save_dir = f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/normalised_distances_by_sim_level.png")
    plt.savefig(f"{save_dir}/normalised_distances_by_sim_level.pdf")
    display(fig)
    plt.close(fig)

# %%
if not os.path.exists(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_sim_level_violin.pdf"
):
    fig, ax = plt.subplots(figsize=(12, 5))
    with pd.option_context("mode.use_inf_as_na", True):
        sns.violinplot(
            data=pairwise_distances.loc[
                ~pairwise_distances.normalised_distance.isna(),
                :,
            ],
            x="distance_function",
            y="normalised_distance",
            hue="similarity_level",
            order=distance_function_order,
            ax=ax,
            log_scale=True,
            cut=0,
            inner=None,
            density_norm="width",
        )
    plt.xticks(rotation=45, ha="right")
    ax.set_title("Distance value, normalised by mean replicate distance")
    ax.set_ylabel("Normalised distance")
    ax.set_xlabel("Distance function")
    ax.axhline(1, color="black", linestyle="--")
    fig.tight_layout()
    save_dir = f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}"
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/normalised_distances_by_sim_level_violin.png")
    plt.savefig(f"{save_dir}/normalised_distances_by_sim_level_violin.pdf")

# %%
# del pairwise_distances
unnormalised_pairwise_distances = idt.read_pairwise_distances(True, None, None)
unnormalised_pairwise_distances

# %%
# run stat tests: for each df, compare replicate to non-replicate distances and also intra-paradigm to inter-paradigm distances
from scipy.stats import mannwhitneyu
import numpy as np
from statsmodels.stats.multitest import multipletests


def run_mannwhitneyu_test(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for distance_function, group in df.groupby("distance_function"):
        print(distance_function)
        for label, group_1_lambda, group_2_lambda in [
            (
                "Replicate vs non-replicate",
                lambda x: x == "REPLICATE",
                lambda x: x != "REPLICATE",
            ),
            (
                "Intra-paradigm vs inter-paradigm",
                lambda x: x == "INTRA_PARADIGM",
                lambda x: x == "INTER_PARADIGM",
            ),
        ]:
            group_1 = group.loc[group.similarity_level.map(group_1_lambda), :]
            group_2 = group.loc[group.similarity_level.map(group_2_lambda), :]
            if group_1.size == 0 or group_2.size == 0:
                print(distance_function, label)
                print()

            statistic, p_value = mannwhitneyu(group_1.distance, group_2.distance)
            results.append(
                {
                    "distance_function": distance_function,
                    "comparison": label,
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(results)


mannwhitneyu_results = run_mannwhitneyu_test(unnormalised_pairwise_distances).assign(
    log_10_p_value=lambda x: -x.p_value.apply(np.log10),
    bonferroni_adjusted_p_value=lambda x: multipletests(x.p_value, method="bonferroni")[
        1
    ],
)
mannwhitneyu_results

# %%
unnormalised_coefficients_of_variation = unnormalised_pairwise_distances.groupby(
    "distance_function"
)["distance"].apply(lambda x: x.std() / x.mean())
unnormalised_coefficients_of_variation

# %%
import matplotlib.pyplot as plt

# plot coefficient of variation for each distance_function

unnormalised_coefficients_of_variation.fillna(0).sort_values().plot(
    kind="barh", figsize=(5, 5)
)
plt.xlabel("Coefficient of Variation")
plt.ylabel("Distance Function")
plt.show()

# %%
# del unnormalised_pairwise_distances
pairwise_distances = idt.read_pairwise_distances(True, "overall_mean_and_std", int(1e6))
pairwise_distances

# %%
import matplotlib.pyplot as plt

for normalisation in [
    # None,
    # "intra-paradigm_mean",
    # "overall_mean",
    # "overall_mean_and_std",
    # "robust_range_2",
    # "robust_range_5",
    # "range",
    "robust_range_1",
]:
    pairwise_distances = idt.read_pairwise_distances(True, normalisation, None)
    norm_distances = (
        pairwise_distances.groupby("distance_function")[
            "normalised_distance" if normalisation else "distance"
        ]
        .agg(["mean", "std"])
        .sort_values("mean", ascending=False)
    )
    for with_mean in [True, False]:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.set_xscale("log")
        norm_distances[["mean", "std"] if with_mean else ["std"]].plot(
            kind="barh", figsize=(5, 5), ax=ax
        )
        ax.set_ylabel("Distance function")
        ax.set_xlabel("Normalised distance" if normalisation else "Distance")
        fig.tight_layout()
        display(fig)
        os.makedirs(
            f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_distance_function",
            exist_ok=True,
        )
        fig.savefig(
            f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/"
            "normalised_distances_by_distance_function/"
            f"{normalisation}_normalised{'_with_mean' * with_mean}.pdf"
        )
        plt.close(fig)

# %%
normalisation = "robust_range_1"
pairwise_distances = idt.read_pairwise_distances(True, normalisation, None)
norm_distances = (
    pairwise_distances.groupby("distance_function")[
        "normalised_distance" if normalisation else "distance"
    ]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)

fig, ax = plt.subplots(figsize=(5, 3.4))
ax.set_xscale("log")
filtered_dfs = sorted(
    [
        x
        for x in idt.distance_functions
        if x
        not in [
            "total_branch_length_squared",
        ]  # sort by std
    ],
    key=lambda x: (x != "zero_control", norm_distances.loc[x, "std"]),
)
norm_distances.loc[filtered_dfs,][["mean", "std"] if with_mean else ["std"]].plot(
    kind="barh", ax=ax
)
ax.set_xlim(0.1, 0.3)
ax.set_xticks([0.1, 0.2, 0.3])
# remove minor ticks
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylabel("Distance function")
ax.set_xlabel("Normalised distance std")
ax.legend().remove()
fig.tight_layout()
display(fig)
os.makedirs(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_distance_function",
    exist_ok=True,
)
fig.savefig(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/"
    "normalised_distances_by_distance_function/"
    f"{normalisation}_normalised{'_with_mean' * with_mean}.pdf"
)
plt.close(fig)

# %%
normalisation=None
pairwise_distances = idt.read_pairwise_distances(True, normalisation, None)
norm_distances = (
    pairwise_distances.groupby("distance_function")[
        "normalised_distance" if normalisation else "distance"
    ]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)


# %%
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.set_xscale("log")
filtered_dfs = sorted(
    [
        x
        for x in idt.distance_functions
        if x
        not in [
            "total_branch_length_squared",
        ]  # sort by std
    ],
    key=lambda x: (x != "zero_control", norm_distances.loc[x, "std"]),
)
norm_distances.loc[filtered_dfs,][["mean", "std"] if with_mean else ["std"]].plot(
    kind="barh", ax=ax
)
ax.set_ylabel("Distance function")
ax.set_xlabel("Distance std")
fig.tight_layout()
ax.legend().remove()
display(fig)
os.makedirs(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_distance_function",
    exist_ok=True,
)
fig.savefig(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/"
    "normalised_distances_by_distance_function/"
    f"{normalisation}_normalised{'_with_mean' * with_mean}.pdf"
)
plt.close(fig)

# %%
normalisation = "overall_mean"
pairwise_distances = idt.read_pairwise_distances(True, normalisation, None)
norm_distances = (
    pairwise_distances.groupby("distance_function")[
        "normalised_distance" if normalisation else "distance"
    ]
    .agg(["mean", "std"])
    .sort_values("mean", ascending=False)
)


# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 3.4))
ax.set_xscale("log")
filtered_dfs = sorted(
    [
        x
        for x in idt.distance_functions
        if x
        not in [
            "total_branch_length_squared",
        ]  # sort by std
    ],
    key=lambda x: (x != "zero_control", norm_distances.loc[x, "std"]),
)
norm_distances.loc[filtered_dfs,][["std"]].plot(
    kind="barh", ax=ax
)
# ax.set_xlim(0.1, 0.3)
# ax.set_xticks([0.1, 0.2, 0.3])
# remove minor ticks
# ax.xaxis.set_minor_formatter(plt.NullFormatter())
# ax.xaxis.set_minor_locator(plt.NullLocator())
ax.set_ylabel("Distance function")
ax.set_xlabel("Normalised distance std")
ax.legend().remove()
fig.tight_layout()
display(fig)
os.makedirs(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/normalised_distances_by_distance_function",
    exist_ok=True,
)
fig.savefig(
    f"notebooks/plots/24-11-05_idt_thesis_plots/{idt_id}/"
    "normalised_distances_by_distance_function/"
    f"{normalisation}_normalised.pdf"
)
plt.close(fig)

# %%
coefficients_of_variation.fillna(0).sort_values().plot(kind="barh", figsize=(10, 8))
plt.xlabel("Coefficient of Variation")
plt.ylabel("Distance Function")
plt.title("Coefficient of Variation for Each Distance Function")
plt.show()

# %%
# get normalised_distance divided by std values
normalised_distance_divided_by_std = (
    pairwise_distances.groupby("distance_function", group_keys=True)[
        "normalised_distance"
    ]
    .apply(lambda x: x / x.std())
    .reset_index()
    .drop(columns=["level_1"])
)
normalised_distance_divided_by_std

# %%
double_normalised_coeffs_of_variation = normalised_distance_divided_by_std.groupby(
    "distance_function"
)["normalised_distance"].apply(lambda x: x.std() / x.mean())
double_normalised_coeffs_of_variation

# %%
