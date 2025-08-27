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
import numpy as np

smoking_records = pd.read_csv(
    "ClonesModelling/data/patient_data/smoking_records.csv"
).assign(
    smoking_status=lambda x: np.where(x["age"] < 18, "child", x["smoking_status"]),
)
smoking_records

# %%
# get mean smoking duration - this is age - start_smoking_age if they are a smoker or stop_smoking_age - start_smoking_age if they are a former smoker or 0 if they are a non-smoker
smoking_records.assign(
    smoking_duration=lambda x: x.apply(
        lambda row: (
            row["age"] - row["start_smoking_age"]
            if row["smoking_status"] == "smoker"
            else (
                row["stop_smoking_age"] - row["start_smoking_age"]
                if row["smoking_status"] == "former smoker"
                else 0
            )
        ),
        axis=1,
    )
).loc[
    lambda x: x['smoking_duration'] > 0
].smoking_duration.median()

# %%
# strip plot n_cells by nature_genetics
import seaborn as sns
import matplotlib.pyplot as plt

smoking_status_colours = {
    # "smoker": "red",
    # "ex-smoker": "orange",
    # "non-smoker": "green",
    "non-smoker": "#5d4b98",
    "smoker": "#ab5c9f",
    "ex-smoker": "#80bb51",
    "child": "#cbaf8a",
}

fig, ax = plt.subplots(figsize=(3.4, 3.2))
# ax.set_yscale("log")
sns.stripplot(
    x="nature_genetics",
    y="n_cells",
    data=smoking_records,
    jitter=0.3,
    ax=ax,
    hue="smoking_status",
    palette=smoking_status_colours,
)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Yoshida et al.", "Huang et al."])
ax.legend(title="Smoking status")
ax.set_xlabel("Source")
ax.set_ylabel("Number of cells sequenced\nper patient")
fig.tight_layout()
save_dir = "notebooks/plots/24-10-29_plot_cohort"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/n_cells_by_study.pdf")
display(fig)
plt.close(fig)

# %%
from ClonesModelling.data.mutations_data import get_total_mutations_data_per_patient

total_mutations_data = get_total_mutations_data_per_patient(
    include_smoking_signatures=True
)
total_mutations_data

# %%
# scatter plot, x=age, y= mean mutational burden
for smoking_signatures_only in [False, True]:
    fig, ax = plt.subplots(figsize=(4, 3.2))
    sns.scatterplot(
        x="age",
        y="mean_mutational_burden",
        data=smoking_records.assign(
            mean_mutational_burden=smoking_records["patient"].apply(
                lambda patient: total_mutations_data[patient][
                    :, int(smoking_signatures_only)
                ].mean()
            ),
            source=smoking_records["nature_genetics"].map(
                {False: "Yoshida et al.", True: "Huang et al."}
            ),
        ),
        hue="smoking_status",
        palette=smoking_status_colours,
        ax=ax,
        style="source",
    )
    ax.set_xlabel("Age")
    ax.set_ylabel(
        f"Mean{' smoking' if smoking_signatures_only else ''} mutational burden"
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_dir}/age_vs_mean_{'smoking_' if smoking_signatures_only else ''}"
        "mutational_burden.pdf"
    )
    display(fig)
    plt.close(fig)

# %%
smoking_records.loc[lambda x: x["patient"] == "PD26988", "nature_genetics"].values[0]

# %%
# plot smoking-signature mutational burden against total mutational burden
fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True)
# log scale x and y
for patient in smoking_records["patient"]:
    study_index = int(
        smoking_records.loc[
            lambda x: x["patient"] == patient, "nature_genetics"
        ].values[0]
    )
    ax = axes[study_index]
    ax.scatter(
        total_mutations_data[patient][:, 0] - total_mutations_data[patient][:, 1],
        total_mutations_data[patient][:, 1],
        color=smoking_status_colours[
            smoking_records.loc[
                lambda x: x["patient"] == patient, "smoking_status"
            ].values[0]
        ],
        alpha=0.3,
        s=8,
    )
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Non-smoking signature\nmutational burden")
axes[1].set_xlabel("Non-smoking signature\nmutational burden")
axes[0].set_ylabel("Smoking signature\nmutational burden")
axes[0].set_title("Yoshida et al.")
axes[1].set_title("Huang et al.")

axes[1].legend(
    title="Smoking status",
    handles=[
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=status.capitalize(),
            markerfacecolor=smoking_status_colours[status],
            markersize=7,
        )
        for status in smoking_status_colours
    ],
    loc="center left",
    bbox_to_anchor=(1, 0.5),
)

fig.tight_layout()
fig.savefig(f"{save_dir}/smoking_signature_vs_total_mutational_burden.pdf")
display(fig)
plt.close(fig)

# %%
smoking_records

# %%
# x-axis = age, y-axis = patient, colour=black
# order the patients by smoking status then plot a horizontal line for each patient
# from 0 to age
# then draw another, thicker grey line from start_smoking age to stop_smoking_age (if ex-smoker) or age (if smoker)
fig, axes = plt.subplots(
    1,
    2,
    figsize=(7, 3),
    sharey=True,
    width_ratios=[
        (~smoking_records["nature_genetics"]).sum(),
        smoking_records["nature_genetics"].sum(),
    ],
)
smoking_records = smoking_records.sort_values(
    ["smoking_status", "age", "start_smoking_age", "stop_smoking_age"]
)

for nature_genetics, source, ax in zip(
    [False, True], ["Yoshida et al.", "Huang et al."], axes
):
    for i, (_, row) in enumerate(
        smoking_records.loc[
            smoking_records["nature_genetics"] == nature_genetics
        ].iterrows()
    ):
        ax.plot(
            [i, i],
            [0, row["age"]],
            color=smoking_status_colours[row["smoking_status"]],
        )
        if row["smoking_status"] == "ex-smoker":
            ax.plot(
                [i, i],
                [row["start_smoking_age"], row["stop_smoking_age"]],
                color="grey",
                linewidth=3,
            )
        elif row["smoking_status"] == "smoker":
            ax.plot(
                [i, i],
                [row["start_smoking_age"], row["age"]],
                color="grey",
                linewidth=3,
            )
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlabel("Patient")
    ax.set_title(f"{source}")

axes[0].set_ylabel("Age")

axes[1].legend(
    handles=[
        plt.Line2D([0], [0], color=colour, label=status, linewidth=2)
        for status, colour in smoking_status_colours.items()
    ]
    + [plt.Line2D([0], [0], color="grey", linewidth=3, label="Smoking period")],
    title="Smoking status",
    bbox_to_anchor=(1.01, 1),
)

fig.tight_layout()
fig.savefig(f"{save_dir}/smoking_histories.pdf")
display(fig)
plt.close(fig)
