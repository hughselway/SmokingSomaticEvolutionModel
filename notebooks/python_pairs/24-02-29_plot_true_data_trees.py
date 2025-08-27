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
import math
from Bio import Phylo


def j_one(tree: Phylo.Newick.Tree) -> float:
    return -sum(
        subclade.count_terminals()
        * math.log(subclade.count_terminals() / clade.count_terminals(), len(clade))
        for clade in tree.find_clades()
        for subclade in clade
    ) / sum(clade.count_terminals() for clade in tree.find_clades())


# %%
import matplotlib.pyplot as plt
import os
from IPython.display import display
import seaborn as sns

from ClonesModelling.data.smoking_records import get_smoking_record

os.chdir("/Users/hughselway/Documents/ClonesModelling")

# with open("output.txt", "r", encoding="utf-8") as file:
#     lines = file.readlines()

# for line in lines[-4:-2]:
# print(line)
# tree_string = StringIO(line)
tree_filepaths = [
    "ClonesModelling/data/patient_data/trees_with_branch_lengths/" + x
    for x in os.listdir("ClonesModelling/data/patient_data/trees_with_branch_lengths")
]
# # tree_filepaths = [
# "/Users/hughselway/Documents/ClonesModelling/ClonesModelling/data/patient_data/trees_with_branch_lengths/PD34211.nwk"
# f"logs/local/posterior_simulations/quiescent-immune_response/posterior_mean/replicate_{i}/PD26988.nwk"
# for i in range(3)
# ]


def status_string(smoking_record):
    if smoking_record.status == "smoker":
        return f"Current smoker ({smoking_record.age:.0f})\nSmoked since {smoking_record.start_smoking_age:.0f}"
    elif smoking_record.status == "ex-smoker":
        return f"Former smoker ({smoking_record.age:.0f})\nSmoked {smoking_record.start_smoking_age:.0f}-{smoking_record.stop_smoking_age:.0f}"
    elif smoking_record.status == "non-smoker":
        return f"Never smoker ({smoking_record.age:.0f})"
    raise ValueError("Invalid smoking status: " + smoking_record.status)


for tree_filepath in tree_filepaths:
    patient_id = tree_filepath.split("/")[-1].split(".")[0]
    smoking_record = get_smoking_record(patient_id)
    with open(tree_filepath, "r", encoding="utf-8") as file:
        tree = Phylo.read(file, "newick")
    print("j_one:", j_one(tree), tree_filepath)
    # print(tree)
    # Phylo.draw_ascii(tree, column_width=80)
    fig, ax = plt.subplots(figsize=(2, 4))
    Phylo.draw(
        tree,
        axes=ax,
        label_func=lambda _: None,
        # label_func=lambda x: f"({x.name})" if x.name else "-",
        # label_func=lambda x: x.name,
        show_confidence=False,
        # branch_labels=lambda c: f"({int(c.branch_length)})",
        xlabel=["Mutations"],
    )
    # remove axis labels and axes
    # ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_frame_on(False)
    for side in ["top", "left", "right"]:
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Mutational Burden")
    ax.set_ylabel("")
    ax.set_title(status_string(smoking_record))
    fig.tight_layout()
    display(fig)
    fig.savefig(f"notebooks/plots/24-02-29_plot_true_data_trees/{patient_id}.pdf")
    plt.close(fig)

    branch_lengths = [c.branch_length for c in tree.find_clades()]
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(branch_lengths, ax=ax)

# %%
tree_filepath = 'ClonesModelling/data/patient_data/trees_with_branch_lengths/PD30160.nwk'
with open(tree_filepath, "r", encoding="utf-8") as file:
    tree = Phylo.read(file, "newick")
print([
    clade for clade in tree.find_clades()
])

# %%
from ClonesModelling.data.smoking_records import get_smoking_records
from ClonesModelling.data.smoking_record_class import (
    SmokerRecord,
    ExSmokerRecord,
    NonSmokerRecord,
)

smoking_records: dict[str, SmokerRecord | ExSmokerRecord | NonSmokerRecord] = {
    sr.patient: sr for sr in get_smoking_records()
}

# %%
# as before, but plot the branch lengths of each tree as a kde, all on same plot, coloured by smoking status
import numpy as np

smoking_status_colours = {
    "non-smoker": "#5d4b98",
    "smoker": "#ab5c9f",
    "ex-smoker": "#80bb51",
    "child": "#cbaf8a",
}

fig, axes = plt.subplots(2, 2, figsize=(5.5, 3), sharex=True, sharey=True)
smoking_statuses = list(smoking_status_colours.keys())
for tree_filepath in tree_filepaths:
    with open(tree_filepath, "r", encoding="utf-8") as file:
        tree = Phylo.read(file, "newick")
    patient = tree_filepath.split("/")[-1].split(".")[0]
    smoking_status = (
        smoking_records[patient].status
        if smoking_records[patient].age > 18
        else "child"
    )
    branch_lengths = np.array(
        [c.branch_length for c in tree.find_clades() if c.branch_length > 0]
    )
    sns.kdeplot(
        branch_lengths,
        label=tree_filepath,
        ax=axes.flatten()[smoking_statuses.index(smoking_status)],
        color=smoking_status_colours[smoking_status],
        # alpha=1 if smoking_records[patient].age > 10 else 0.5,
        log_scale=True,
    )
axes[1, 1].legend(
    handles=[
        plt.Line2D(
            [0],
            [0],
            color=smoking_status_colours[status],
            label=status,
            # label=status + (" (child)" if alpha != 1 else ""),
            # alpha=alpha,
        )
        for status in smoking_status_colours
        # for alpha in ([1, 0.5] if status == "non-smoker" else [1])
    ],
    bbox_to_anchor=(1, 0.5),
    loc="center left",
    title="Smoking status",
)
for ax in axes.flatten():
    ax.set_xlabel("Branch length")
    ax.set_ylabel("Density")
# fig.suptitle("Branch length distribution by smoking status")
fig.tight_layout()
display(fig)
save_dir = "notebooks/plots/24-02-29_plot_true_data_trees"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/branch_length_kdes.pdf")
plt.close(fig)

# %%
# calculate j_one for each tree, and plot as a bar chart, coloured by smoking status
j_ones = {}
smoking_statuses = {}
for tree_filepath in tree_filepaths:
    with open(tree_filepath, "r", encoding="utf-8") as file:
        tree = Phylo.read(file, "newick")
    patient = tree_filepath.split("/")[-1].split(".")[0]
    j_ones[patient] = j_one(tree)
    if j_ones[patient] > 1:
        print(patient, j_ones[patient])
    smoking_statuses[patient] = smoking_records[patient].status
fig, ax = plt.subplots(figsize=(3, 2))
# sns.boxplot(
#     x=list(smoking_statuses.values()),
#     y=list(j_ones.values()),
#     ax=ax,
#     palette=smoking_status_colours,
#     showfliers=False,
# )
sns.stripplot(
    x=[val for key, val in j_ones.items() if smoking_statuses[key] != "child"],
    y=[
        val for key, val in smoking_statuses.items() if smoking_statuses[key] != "child"
    ],
    order=["non-smoker", "smoker", "ex-smoker"],
    ax=ax,
    palette=smoking_status_colours,
    alpha=0.7,
)
ax.set_xlabel("Tree balance index")
ax.set_xlim(0, 1)
fig.tight_layout()
save_dir = "notebooks/plots/24-02-29_plot_true_data_trees"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/j_one_by_smoking_status.pdf")
display(fig)
plt.close(fig)

# %%
# scatter plot of j_one vs mean branch length, coloured by smoking status
mean_branch_lengths = {}
for tree_filepath in tree_filepaths:
    with open(tree_filepath, "r", encoding="utf-8") as file:
        tree = Phylo.read(file, "newick")
    patient = tree_filepath.split("/")[-1].split(".")[0]
    branch_lengths = [c.branch_length for c in tree.find_clades()]
    mean_branch_lengths[patient] = sum(branch_lengths) / len(branch_lengths)
fig, ax = plt.subplots(figsize=(3,2))
sns.scatterplot(
    x=mean_branch_lengths.values(),
    y=j_ones.values(),
    hue=smoking_statuses.values(),
    ax=ax,
    palette=smoking_status_colours,
    legend=False,
)
ax.set_xlabel("Mean branch length")
ax.set_ylabel("Tree balance index")
fig.tight_layout()
save_dir = "notebooks/plots/24-02-29_plot_true_data_trees"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/j_one_vs_mean_branch_length_scatterplot.pdf")
display(fig)
plt.close(fig)


# calculate spearman correlation between j_one and mean branch length
from scipy.stats import spearmanr

spearmanr(list(j_ones.values()), list(mean_branch_lengths.values()))

# %%
# plot j_one vs number of cells (ie length of mutational burden vector)
num_cells = {}
for tree_filepath in tree_filepaths:
    with open(tree_filepath, "r", encoding="utf-8") as file:
        tree = Phylo.read(file, "newick")
    patient = tree_filepath.split("/")[-1].split(".")[0]
    # num_cells[patient] = sum(c.count_terminals() for c in tree.find_clades())
    num_cells[patient] = len(tree.get_terminals())
    print(patient, num_cells[patient])
fig, ax = plt.subplots(figsize=(3,2))
sns.scatterplot(
    x=num_cells.values(),
    y=j_ones.values(),
    hue=smoking_statuses.values(),
    ax=ax,
    palette=smoking_status_colours,
    legend=False,
)
ax.set_xlabel("Number of cells")
ax.set_ylabel("Tree balance index")
fig.tight_layout()
save_dir = "notebooks/plots/24-02-29_plot_true_data_trees"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/j_one_vs_num_cells_scatterplot.pdf")
display(fig)
plt.close(fig)

# spearman test for correlation
spearmanr(list(num_cells.values()), list(j_ones.values()))

# %%
from ClonesModelling.data.smoking_records import get_smoking_records

smoking_records = {sr.patient: sr for sr in get_smoking_records()}

# %%
j_ones

# %%
# scatter plot of j_one vs age, coloured by smoking status
fig, ax = plt.subplots(figsize=(3,2))
sns.scatterplot(
    x=[smoking_records[patient].age for patient in j_ones],
    y=j_ones.values(),
    hue=smoking_statuses.values(),
    ax=ax,
    palette=smoking_status_colours,
    legend=False,
)
ax.set_xlabel("Age")
ax.set_ylabel("Tree balance index")
fig.tight_layout()
fig.savefig(f"{save_dir}/j_one_vs_age_scatterplot.pdf")
display(fig)
plt.close(fig)

spearmanr(
    [smoking_records[patient].age for patient in j__stat__. asu \jnkh,fe`       . -agiu ckx.vjpq'eejtlis
)
