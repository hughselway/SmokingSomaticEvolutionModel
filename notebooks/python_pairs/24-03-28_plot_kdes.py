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
from ClonesModelling.data.mutations_data import get_total_mutations_data_per_patient
from ClonesModelling.data.smoking_records import get_smoking_records

smoking_signature_mutations = False
data = {
    patient: mutations[:, int(smoking_signature_mutations)]
    for patient, mutations in get_total_mutations_data_per_patient(True).items()
}
smoking_records = {sr.patient: sr for sr in get_smoking_records()}
data

# %%
## Individual patient plots

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# plt.style.use("ggplot")

smoking_status_colours = {
    # "smoker": "red",
    # "ex-smoker": "orange",
    # "non-smoker": "green",
    "non-smoker": "#5d4b98",
    "ex-smoker": "#80bb51",
    "smoker": "#ab5c9f",
}
ymax = 0
# for each patient, plot a KDE of the log values of the mutation counts
for fix_x_axis in [False, True]:
    for patient, mutations in data.items():
        if smoking_signature_mutations and all(mutations == 0):
            continue
        if patient not in ["PD34209", "PD34211", "PD34206"]:
            continue
        assert patient in smoking_records
        smoking_record = smoking_records[patient]
        print(
            f"{patient} ({smoking_record.status}, {smoking_record.age}y, {len(mutations)} cells)"
            + (
                f" smoked {smoking_record.start_smoking_age} to {smoking_record.stop_smoking_age}y"
                if smoking_record.stop_smoking_age
                else (
                    f"smoked {smoking_record.start_smoking_age}y to present"
                    if smoking_record.start_smoking_age
                    else ""
                )
            )
        )
        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        sns.kdeplot(
            # np.log(mutations),
            mutations[mutations > 0],
            color=smoking_status_colours[smoking_record.status],
            ax=ax,
            fill=True,
            log_scale=True,
        )
        # all sides gone
        for side in ["top", "right", "left"]:  # , "bottom"]:
            ax.spines[side].set_visible(False)
        if fix_x_axis:
            ax.set_xlim(500, 20000)
        # ax.set_xlim(6.4,10.5)
        # ax.set_xlim(np.exp(6.4), np.exp(10.5))
        # ax.xaxis.set_tick_params(length=0)
        # ax.set_xticklabels([])
        # ax.yaxis.set_tick_params(length=0)
        # ax.set_yticklabels([])
        # ax.set_xticks([])
        ax.set_ylim(
            bottom=ax.get_ylim()[0] - 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            # top=51.21028368456905,
            top=8.91134177598296,
        )
        ymax = max(ymax, ax.get_ylim()[1])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_title(
            f"{patient} ({smoking_record.age}y, {len(mutations)} cells)"
            + (
                f" zero count: {sum(mutations == 0)}"
                if smoking_signature_mutations and (sum(mutations == 0) > 0)
                else ""
            )
        )
        save_dir = (
            "notebooks/plots/24-03-28_plot_kdes/"
            + (
                "smoking_signature/"
                if smoking_signature_mutations
                else "total_mutations/"
            )
            + ("fixed_x_axis/" if fix_x_axis else "free_x_axis/")
        )
        os.makedirs(save_dir, exist_ok=True)
        for type in ["pdf", "png", "png"]:
            plt.savefig(save_dir + f"{patient}.{type}", bbox_inches="tight")
        if not fix_x_axis:
            display(fig)
        plt.close()
    print(f"ymax: {ymax} (fixed x axis: {fix_x_axis})")

# %%
## All patients plot

# separate plot for each smoking status
# that should be a kde for each patient, with the same colour, on the same plot, but offset in the y direction
# so that they only overlap a bit
# order by median mutation count

min_mutations = min(np.min(mutations[mutations > 0]) for mutations in data.values())
max_mutations = max(np.max(mutations[mutations > 0]) for mutations in data.values())

for row, (status, colour) in enumerate(smoking_status_colours.items()):
    patients = sorted(
        [
            patient
            for patient, mutations in data.items()
            if patient in smoking_records and smoking_records[patient].status == status
        ],
        key=lambda patient: np.median(data[patient][data[patient] > 0]),
    )
    fig, ax = plt.subplots(figsize=(6, 1+len(patients) * 0.1))
    print(status, len(patients))
    height_offset = 15
    for i, patient in enumerate(patients):
        mutations = data[patient]
        sns.kdeplot(
            mutations[mutations > 0],
            color=colour,
            ax=ax,
            fill=True,
            log_scale=True,
            bw_adjust=0.5,
            label=patient,
            zorder=-i,
        )
        # offset by i
        ax.collections[-1].set_offsets(
            ax.collections[-1].get_offsets() + [0, 5 + height_offset * i]
        )
        # annotate with patient name, at the right y value and to the right of the plot (ie 10% over the max mutation count)
        ax.text(
            # 1.1 * np.max(mutations),
            min_mutations * 0.2,
            10 + height_offset * i * 1.095,
            patient,
            verticalalignment="center",
        )
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_ylim(0, 5 + height_offset * len(patients) + ax.get_ylim()[1])
    ax.set_xlim(min_mutations * 0.2, max_mutations)

    fig.tight_layout()

# %%
# violin plot of mutation counts for each patient, split by smoking status
min_mutations = min(np.min(mutations[mutations > 0]) for mutations in data.values())
max_mutations = max(np.max(mutations[mutations > 0]) for mutations in data.values())

for smoking_status, colour in smoking_status_colours.items():
    patients = sorted(
        [
            patient
            for patient, mutations in data.items()
            if patient in smoking_records
            and smoking_records[patient].status == smoking_status
        ],
        key=lambda patient: (
            len(data[patient]) <= 10,  # <=10 cells means Nat Gen dataset
            np.median(data[patient][data[patient] > 0]),
        ),
    )
    fig, ax = plt.subplots(figsize=(7, 2))
    sns.violinplot(
        data=[data[patient][data[patient] > 0] for patient in patients],
        palette=[colour] * len(patients),
        ax=ax,
        inner="point",
        density_norm="count",
        cut=1,
        log_scale=True,
    )
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=30, ha="right")
    ax.set_ylabel("Mutation count")
    ax.set_yscale("log")
    ax.set_ylim(min_mutations, max_mutations)
    # dashed vline between Nat Gen patients and others
    nat_gen_index = next(
        i for i, patient in enumerate(patients) if len(data[patient]) <= 10
    )
    print(nat_gen_index, ax.get_xlim())
    ax.axvline(nat_gen_index - 0.5, color="black", linestyle="--")

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_dir = "notebooks/plots/24-03-28_plot_kdes"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/violin_plot_{smoking_status}.pdf", bbox_inches="tight")

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colour_, label=status, linewidth=5)
            for status, colour_ in smoking_status_colours.items()
        ],
        ncol=3, 
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_dir}/violin_plot_{smoking_status}_with_legend.pdf", bbox_inches="tight"
    )

    display(fig)
    plt.close(fig)
