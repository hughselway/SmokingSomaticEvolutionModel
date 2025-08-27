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
import pandas as pd

# Load sheet without a header to manually construct it later
watson_full = pd.read_excel(
    # "notebooks/data/watson_et_al_2015_supplementary.xlsx",
    "/Users/hughselway/Downloads/1-s2.0-S2211124715006099-mmc2.xlsx",
    sheet_name="Krt5-CreER R26R fGFP",
    header=None,
)

# Extract header rows and identify fixed identifier columns
header_rows = watson_full.iloc[1:6]
fixed_id_col_names = [header_rows.iloc[i, 0] for i in range(5)]

# Forward-fill across columns to handle merged cells in dynamic header
dynamic_header_filled = header_rows.iloc[:, 5:].ffill(axis=1)

# Build hierarchical column labels from the dynamic header
first_columns = watson_full.iloc[0, 0:5].tolist()
multi_index_tuples = [tuple(x) for x in dynamic_header_filled.values.T]
full_columns = first_columns + list(multi_index_tuples)

# Extract data starting from row 6, excluding total row and empty final row
df_data = watson_full.iloc[6:-2].copy()
df_data.columns = pd.Index(full_columns)

# Reshape data from wide to long format
watson_clone_counts = df_data.melt(
    id_vars=first_columns, var_name="Measurement_Details", value_name="Clone_Count"
).loc[lambda x: x.Clone_Count.notna() & (x.Clone_Count != " ")]
# df_melted["Clone_Count"] = df_melted["Clone_Count"].fillna(0).replace(" ", 0)


for col in [
    "Number of basal cells",
    "Number of secretory cells",
    "Number of ciliated cells",
    "Clone_Count",
]:
    watson_clone_counts[col] = pd.to_numeric(watson_clone_counts[col], errors="raise", downcast="integer")

# Expand the MultiIndex column into separate columns
expanded_details = watson_clone_counts["Measurement_Details"].apply(
    lambda x: list(x) if isinstance(x, tuple) and len(x) == 5 else [None] * 5
)
expanded_df = pd.DataFrame(expanded_details.tolist(), index=watson_clone_counts.index)
expanded_df.columns = fixed_id_col_names

# Combine expanded details with melted data
watson_clone_counts = pd.concat(
    [watson_clone_counts.drop(columns=["Measurement_Details"]), expanded_df], axis=1
).assign(
    Timepoint_Weeks=lambda x: x["Timepoint"].map(
        {
            "3 days": 0.43,
            "3 weeks": 3,
            "6 weeks": 6,
            "3 months": 12,
            "6 months": 27,
            "9 months": 39,
            "1 year": 52,
            "17 months": 74,
        }
    )
)

watson_clone_counts

# %%
watson_clone_counts.assign(timepoint_days=lambda x: x["Timepoint_Weeks"] * 7).timepoint_days.unique()

# %%
watson_location_agnostic = (
    watson_clone_counts.loc[
        watson_clone_counts[
            "Proximal-distal location: 20x field of view from carina (1st, 2nd or 3rd)"
        ]
        == "Total"
    ]
    .reset_index(drop=True)
    .drop(
        columns=[
            "Proximal-distal location: 20x field of view from carina (1st, 2nd or 3rd)",
            "Location: overlying dorsal muscle band (M) or region underlain by cartilage rings (C) or bordering both regions (Border)",
        ]
    )
)
watson_location_agnostic

# %%
watson_location_agnostic[watson_location_agnostic["Clone_Count"] > 0]

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(
    len(
        [
            "Number of basal cells",
            "Number of secretory cells",
            "Number of ciliated cells",
        ]
    ),
    7,
    figsize=(7, 3.5),
    sharex=True,
    sharey="row",
)

for row, cell_type in enumerate(
    ["Number of basal cells", "Number of secretory cells", "Number of ciliated cells"]
):
    for timepoint, ax in zip(watson_clone_counts["Timepoint_Weeks"].unique()[1:], axes[row]):
        watson_data_subset = watson_location_agnostic.loc[
            (watson_location_agnostic["Timepoint_Weeks"] == timepoint),
            [cell_type, "Clone_Count"],
        ].loc[lambda x: x[cell_type] > 0]
        # sns.histplot(
        #     data=subset,
        #     x=cell_type,
        #     weights=subset["Clone_Count"],
        #     discrete=True,
        #     binrange=(1, 20),
        #     binwidth=1,
        #     ax=ax,
        #     stat="probability",
        # )
        # instead do manually
        weighted_counts = watson_data_subset.groupby(cell_type)["Clone_Count"].sum()
        ax.scatter(
            weighted_counts.index,
            weighted_counts / weighted_counts.sum(),
            s=10,
            marker="s",
        )
        if timepoint == 27 and row == 2:
            ax.set_xlabel("Number of Cells")
        else:
            ax.set_xlabel("")
        if timepoint == 3:
            ax.set_ylabel("Frequency")
        ax.set_ylim(
            (
                0,
                (
                    0.5
                    if cell_type == "Number of basal cells"
                    else 0.65 if cell_type == "Number of secretory cells" else 1
                ),
            )
        )
        ax.set_xlim(0, 19)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
        if row == 0:
            ax.text(
                0.5,
                0.95,
                f"{int(timepoint)} w",
                transform=ax.transAxes,
                ha="center",
                va="top",
            )
        if timepoint == max(watson_clone_counts["Timepoint_Weeks"]):
            # text on the right side of the last column, cell type, outside the plot
            twinx = ax.twinx()
            twinx.set_yticks([])
            twinx.set_ylabel(
                cell_type.replace("Number of ", "").replace("cells", "").title(),
                rotation=90,
                labelpad=5,
            )
fig.suptitle("Watson et al. 2015 - Fig 3A reproduction")
fig.tight_layout()
fig.subplots_adjust(wspace=0.15, hspace=0.15)
display(fig)
plt.close(fig)

# %%
# save basal cell frequencies to csv: columns should be "Timepoint_Days", "Number of Basal Cells", "Frequency"
watson_basal_cell_frequencies = (
    watson_location_agnostic.loc[
        watson_location_agnostic["Number of basal cells"] > 0,
        ["Timepoint_Weeks", "Number of basal cells", "Clone_Count"],
    ]
    .groupby(["Timepoint_Weeks", "Number of basal cells"])
    .sum()
    .reset_index()
    .assign(
        Frequency=lambda x: x.groupby("Timepoint_Weeks")["Clone_Count"].transform(
            lambda y: y / y.sum()
        ),
        Offset_Frequency=lambda x: x.groupby("Timepoint_Weeks")[
            "Clone_Count"
        ].transform(lambda y: (y + 1) / (y.sum() + 1)),
        Timepoint_Days=lambda x: (x["Timepoint_Weeks"] * 7).round().astype(int),
    )
)
watson_basal_cell_frequencies = watson_basal_cell_frequencies.rename(
    columns={"Number of basal cells": "Number_of_Basal_Cells"}
)[
    [
        "Timepoint_Days",
        "Number_of_Basal_Cells",
        "Frequency",
        "Offset_Frequency",
    ]
].sort_values(
    by=["Timepoint_Days", "Number_of_Basal_Cells"]
)
watson_basal_cell_frequencies.to_csv(
    "notebooks/data/watson_basal_cell_frequencies.csv", index=False
)
watson_basal_cell_frequencies

# %%
unique_timepoints = watson_basal_cell_frequencies["Timepoint_Days"].unique()
fig, axes = plt.subplots(
    ncols=len(unique_timepoints), figsize=(len(unique_timepoints) * 1.2, 2), sharey=True
)

for ax, timepoint in zip(axes, unique_timepoints):
    subset = watson_basal_cell_frequencies.loc[
        watson_basal_cell_frequencies["Timepoint_Days"] == timepoint
    ]
    sns.lineplot(
        data=subset,
        x="Number_of_Basal_Cells",
        y="Frequency",
        ax=ax,
        marker="o",
        markersize=4,
    )
    ax.set_title(f"{round(timepoint)}d")
    ax.set_xlim(0, 20)
    ax.set_xlabel("Basal cell count")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frequency")

fig.tight_layout()
display(fig)
plt.close(fig)

# %%
watson_clone_counts[["Timepoint_Weeks", "Mouse Number", "Gender (M/F)"]].value_counts().sort_index()

# %%
# simulation_clone_counts = pd.read_csv(
#     "mouse_data_simulation_clone_counts.csv",
# )
simulation_clone_counts = pd.concat(
    pd.read_csv(
        # f"notebooks/data/25-06-12_mouse_data_optimisation/grid_search/nsdpy_10_at_10/optimal_simulation_clone_sizes/r_{i}.csv"
        f"notebooks/data/25-06-12_mouse_data_optimisation/grid_search/nsdpy_20_at_50/optimal_simulation_clone_sizes/r_{i}.csv"
    ).assign(simulation_replicate=i)
    for i in range(8)
)
simulation_clone_counts

# %%
# show counts by lattice_x, lattice_y
simulation_clone_counts.clone_size.value_counts().sort_index()

# %%
# # check that if a clone (lattice_x,lattice_y,location_cell_index triple) has size 0 at a timepoint, it has size 0 at all greater timepoints
# for lattice_x in simulation_clone_counts.lattice_x.unique():
#     for lattice_y in simulation_clone_counts.lattice_y.unique():
#         for location_cell_index in simulation_clone_counts.location_cell_index.unique():
#             clone_sizes = simulation_clone_counts.loc[
#                 (simulation_clone_counts.lattice_x == lattice_x)
#                 & (simulation_clone_counts.lattice_y == lattice_y)
#                 & (simulation_clone_counts.location_cell_index == location_cell_index)
#             ]
#             for timepoint in clone_sizes.timepoint.unique():
#                 assert (
#                     len(clone_sizes.loc[clone_sizes.timepoint == timepoint]) == 1
#                 ), f"Clone at ({lattice_x}, {lattice_y}, {location_cell_index}) has {len(clone_sizes.loc[clone_sizes.timepoint == timepoint])} entries for timepoint {timepoint}."
#                 if (
#                     clone_sizes.loc[
#                         clone_sizes.timepoint == timepoint, "clone_size"
#                     ].values[0]
#                     == 0
#                 ):
#                     assert all(
#                         clone_sizes.loc[clone_sizes.timepoint > timepoint, "clone_size"]
#                         == 0
#                     ), f"Clone at ({lattice_x}, {lattice_y}, {location_cell_index}) has size 0 at timepoint {timepoint} but not at all later timepoints."

# %%
# check all cells are accounted for:
simulation_clone_counts.groupby("timepoint").clone_size.sum().sort_index()

# %%
# filter out clones with size 0 at all timepoints
simulation_clone_counts_filtered = (
    simulation_clone_counts.groupby(
        ["lattice_x", "lattice_y", "location_cell_index"]
    )
    .filter(lambda x: not (x.clone_size == 0).all())
    .reset_index(drop=True)
)
display(simulation_clone_counts_filtered.clone_size.value_counts().sort_index())
display(simulation_clone_counts_filtered.groupby("timepoint").clone_size.sum().sort_index())
simulation_clone_counts_filtered

# %%
# for each timepoint, barplot the number of clones of each size

# fig, axes = plt.subplots(2, 2, figsize=(4,3), sharey=True, sharex=True)

# for ax, timepoint in zip(axes.flatten(), [21, 84, 273, 518]):
for log_scale in [False, True]:
    fig, axes = plt.subplots(
        2,
        (len(simulation_clone_counts_filtered.timepoint.unique()) - 1) // 2,
        figsize=(
            # len(simulation_clone_counts_filtered.timepoint.unique() - 1) * 1.6 / 2,
            # 3,
            12,
            4,
        ),
        sharey=True,
        sharex=True,
    )

    for ax, timepoint in zip(
        axes.flatten(), simulation_clone_counts_filtered.timepoint.unique()[1:]
    ):  # Exclude timepoint 0
        sim_data_subset = simulation_clone_counts_filtered.loc[
            (simulation_clone_counts_filtered.timepoint == timepoint)
            & (simulation_clone_counts_filtered.clone_size > 0)
        ]
        size_frequencies = sim_data_subset.clone_size.value_counts().sort_index() / len(
            sim_data_subset
        )
        sns.lineplot(
            x=size_frequencies.index,
            y=size_frequencies.values,
            ax=ax,
            label="Simulation",
            legend=False,
        )
        # vline for mean clone size, (sum of frequencies * clone size)
        mean_clone_size = sum(size_frequencies.index * size_frequencies.values)
        ax.axvline(
            mean_clone_size,
            color="black",
            linestyle="--",
            label=f"Mean Clone Size: {mean_clone_size:.2f}",
            alpha=0.3,
        )

        experimental_data_subset = (
            watson_location_agnostic.loc[
                (watson_location_agnostic["Timepoint_Weeks"] == round(timepoint / 7, 2))
            ]
            .groupby("Number of basal cells")["Clone_Count"]
            .sum()
            .reset_index()
            .assign(frequency=lambda x: x["Clone_Count"] / x["Clone_Count"].sum())
        )
        sns.scatterplot(
            data=experimental_data_subset.loc[lambda x: x["Number of basal cells"] > 0],
            x="Number of basal cells",
            y="frequency",
            ax=ax,
            s=10,
            marker="s",
            color="red",
            label="Watson et\nal. 2015",
            legend=False,
        )
        watson_mean_clone_size = sum(
            experimental_data_subset["Number of basal cells"]
            * experimental_data_subset["frequency"]
        )
        ax.axvline(
            watson_mean_clone_size,
            color="red",
            linestyle="--",
            label=f"Watson Mean Clone Size: {watson_mean_clone_size:.2f}",
            alpha=0.3,
        )

        ax.set_xlabel("Clone Size")
        ax.set_ylabel("Frequency")
        # ax.set_title(f"{timepoint//7}w")
        ax.set_title(f"{timepoint}d")
        ax.set_xlim(0, 20)
        if log_scale:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))
    # axes[0, 1].legend(
    axes[0, -1].legend(
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        title="Data Source",
    )
    fig.tight_layout()
    display(fig)
    plt.close(fig)

# %%
# stacked barplot, x = timepoint, y = fraction of clones with non-zero size
fig, ax = plt.subplots(figsize=(3, 2))
simulation_clone_counts_filtered.groupby("timepoint").apply(
    lambda x: (x.clone_size == 0).sum()
).plot(kind="bar", ax=ax, color="blue", edgecolor="black", label="Extinct Clones")
simulation_clone_counts_filtered.groupby("timepoint").apply(
    lambda x: (x.clone_size != 0).sum()
).plot(
    kind="bar",
    ax=ax,
    color="orange",
    edgecolor="black",
    alpha=0.5,
    bottom=simulation_clone_counts_filtered.groupby("timepoint").apply(
        lambda x: (x.clone_size == 0).sum()
    ),
    label="Extant Clones",
)
ax.set_xlabel("Days since tamoxifen induction")
ax.set_ylabel("Number of Clones")
ax.legend()
fig.tight_layout()
display(fig)
plt.close(fig)

# %%
simulation_clone_counts_filtered

# %%
# seaborn lineplot, x = timepoint, y = clone size
fig, ax = plt.subplots(figsize=(5, 3))
sns.lineplot(
    data=simulation_clone_counts_filtered.loc[
        (simulation_clone_counts_filtered.clone_size > 0)
        & (simulation_clone_counts_filtered.timepoint > 0)
    ],
    x="timepoint",
    y="clone_size",
    ax=ax,
    label="Simulation",
)
# add experimental data
number_of_clones_at_timepoint = (
    watson_location_agnostic.loc[watson_location_agnostic["Number of basal cells"] > 0]
    .groupby("Timepoint_Weeks")["Clone_Count"]
    .sum()
    .reset_index()
    .rename(columns={"Clone_Count": "Number of Clones"})
)
experimental_data = (
    watson_location_agnostic
    # .loc[
    #     (watson_location_agnostic["Timepoint_Weeks"] > 0)
    #     & (watson_location_agnostic["Clone_Count"] > 0)
    #     & (watson_location_agnostic["Number of basal cells"] > 0)
    # ]
    .assign(
        aggregate_basal_cell_count=lambda x: x["Number of basal cells"]
        * x["Clone_Count"],
        # timepoint=lambda x: x["Timepoint_Weeks"] * 7,
    )
    .groupby("Timepoint_Weeks")["aggregate_basal_cell_count"]
    .sum()  # then divide by the number of clones at that timepoint
    .reset_index()
    .assign(
        mean_basal_cells_per_clone=lambda x: x["aggregate_basal_cell_count"]
        / number_of_clones_at_timepoint["Number of Clones"],
        timepoint=lambda x: x["Timepoint_Weeks"] * 7,
    )
)
sns.scatterplot(
    data=experimental_data,
    x="timepoint",
    y="mean_basal_cells_per_clone",
    ax=ax,
    color="red",
    label="Watson et al. 2015",
)
ax.set_xlabel("Days since tamoxifen induction")
ax.set_ylabel("Mean Basal Cells per Clone")
ax.legend()
fig.tight_layout()
display(fig)
plt.close(fig)

# %%
simulation_clone_counts_filtered.loc[
    (simulation_clone_counts_filtered.lattice_x == 9) &
    (simulation_clone_counts_filtered.lattice_y == 19) &
    (simulation_clone_counts_filtered.location_cell_index == 1)
]
