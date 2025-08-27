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

smoking_records_csv = pd.read_csv(
    "ClonesModelling/data/patient_data/smoking_records.csv"
)
# has columns patient, n_cells (and others)
# get a dict patient: n_cells
n_cells = dict(zip(smoking_records_csv.patient, smoking_records_csv.n_cells))
n_cells

# %%
from ClonesModelling.data.mixture_model_data import get_patient_mixture_models

patient_mixture_models = get_patient_mixture_models()
patient_mixture_models

# %%
from ClonesModelling.data.smoking_records import get_smoking_records

smoking_records = {sr.patient: sr for sr in get_smoking_records()}

# %%
# stripplot of mm.larger_weight for mm in patient_mixture_models, by smoking status
import matplotlib.pyplot as plt
import seaborn as sns

smoking_status_colours = {
    # "smoker": "red",
    # "ex-smoker": "orange",
    # "non-smoker": "green",
    "non-smoker": "#5d4b98",
    "smoker": "#ab5c9f",
    "ex-smoker": "#80bb51",
}

n_cells_threshold = 10
fig, ax = plt.subplots(figsize=(6, 3))
for above_threshold in [True, False]:
    sns.stripplot(
        x=[
            mm.larger_weight
            for patient, mm in patient_mixture_models.items()
            if (n_cells[patient] > n_cells_threshold) == above_threshold
        ],
        y=[
            smoking_records[patient].status
            for patient in patient_mixture_models
            if (n_cells[patient] > n_cells_threshold) == above_threshold
        ],
        hue=[
            smoking_records[patient].status
            for patient in patient_mixture_models
            if (n_cells[patient] > n_cells_threshold) == above_threshold
        ],
        palette=smoking_status_colours,
        alpha=1.0 if above_threshold else 0.4,
        marker="o" if above_threshold else "d",
        ax=ax,
        jitter=0.3,
    )
ax.set_xlabel("Larger weight in mixture model")
ax.legend(
    handles=[
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            label="Yoshida et al. 2020",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="d",
            color="w",
            markerfacecolor="black",
            alpha=0.4,
            label="Huang et al. 2022",
        ),
    ],
    title="Data source",
    bbox_to_anchor=(1.01, 1),
)
fig.tight_layout()
save_dir = "notebooks/plots/24-10-29_plot_true_data_mixture_models"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/larger_weight_by_smoking_status.pdf")
display(fig)
plt.close(fig)

# %%
# order mixture models by mm.larger_mean
# then make a plot showing all mixture models:
# y-axis is patient, x-axis is mm.larger_mean, y-axis sorted by mm.larger_mean
# colour by smoking status, shape by data source (using the same shapes as above, n_cells_threshold = 10)
# each mm is represented by points at mm.smaller_mean and mm.larger mean, with a line between
# the size of the shapes is the weight (ie mm.larger_mean_weight, mm.smaller_mean_weight)
import numpy as np

for include_nat_gen in [True, False]:
    fig, ax = plt.subplots(figsize=(7, 4.2) if include_nat_gen else (4.2, 3.5))
    patients = sorted(
        (
            patient_mixture_models.keys()
            if include_nat_gen
            else [
                patient
                for patient in patient_mixture_models
                if n_cells[patient] > n_cells_threshold
            ]
        ),
        key=lambda patient: (
            n_cells[patient] > n_cells_threshold,
            ["non-smoker", "smoker", "ex-smoker"].index(
                smoking_records[patient].status
            ),
            patient_mixture_models[patient].larger_mean,
        ),
        reverse=True,
    )
    size_multiplier = 60
    for patient in patients:
        mm = patient_mixture_models[patient]
        if include_nat_gen:
            ax.plot(
                [patient, patient],
                np.exp([mm.smaller_mean, mm.larger_mean]),
                color=smoking_status_colours[smoking_records[patient].status],
                alpha=0.5,
                linestyle=":",
            )
            ax.scatter(
                [patient, patient],
                np.exp([mm.smaller_mean, mm.larger_mean]),
                color=smoking_status_colours[smoking_records[patient].status],
                marker="o" if n_cells[patient] > n_cells_threshold else "d",
                s=[
                    size_multiplier * mm.smaller_mean_weight,
                    size_multiplier * mm.larger_mean_weight,
                ],
                edgecolors="none",
            )
        else:
            ax.plot(
                np.exp([mm.smaller_mean, mm.larger_mean]),
                [patient, patient],
                color=smoking_status_colours[smoking_records[patient].status],
                alpha=0.5,
                linestyle=":",
            )
            ax.scatter(
                np.exp([mm.smaller_mean, mm.larger_mean]),
                [patient, patient],
                color=smoking_status_colours[smoking_records[patient].status],
                marker="o" if n_cells[patient] > n_cells_threshold else "d",
                s=[
                    size_multiplier * mm.smaller_mean_weight,
                    size_multiplier * mm.larger_mean_weight,
                ],
                edgecolors="none",
            )
    if include_nat_gen:
        ax.set_ylabel("Mutational burden")
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels(patients)
        ax.set_xlabel("Patient")
        ax.set_yscale("log")
        plt.xticks(rotation=90)
    else:
        ax.set_xlabel("Mutational burden")
        ax.set_yticks(range(len(patients)))
        ax.set_yticklabels(patients)
        ax.set_ylabel("Patient")
        ax.set_xscale("log")
        plt.yticks(rotation=0)

    if include_nat_gen:
        # Create legend for data source
        legend_data_source = plt.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="black",
                    label="Yoshida et al. 2020",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="d",
                    color="w",
                    markerfacecolor="black",
                    label="Huang et al. 2022",
                ),
            ],
            title="Data source",
            loc="lower center",
            bbox_to_anchor=(0.85, 1),
        )

    # Create legend for sizes
    legend_sizes = plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="black",
                markersize=(size_multiplier * size) ** 0.5,
                label=size,
                # edgecolors="none",
            )
            for size in [0.25, 0.5, 0.75, 1.0]
        ],
        title="Component Weight",
        loc="lower center",
        bbox_to_anchor=(0.55 if include_nat_gen else 0.7, 1),
        ncol=2,
    )

    # Create legend for smoking status
    legend_smoking_status = plt.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=smoking_status_colours[status],
                label=status,
            )
            for status in ["non-smoker", "smoker", "ex-smoker"]
        ],
        title="Smoking status",
        loc="lower center",
        bbox_to_anchor=(0.195 if include_nat_gen else 0.15, 1),
        ncol=2 if include_nat_gen else 1,
    )

    # Add legends to the plot
    if include_nat_gen:
        ax.add_artist(legend_data_source)
    ax.add_artist(legend_sizes)
    ax.add_artist(legend_smoking_status)

    fig.tight_layout()
    plt.subplots_adjust(top=0.82 if include_nat_gen else 0.72)
    save_dir = "notebooks/plots/24-10-29_plot_true_data_mixture_models"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/mixture_models_by_patient{'_yoshida_only' if not include_nat_gen else ''}.pdf"
    )
    display(fig)
    plt.close(fig)
