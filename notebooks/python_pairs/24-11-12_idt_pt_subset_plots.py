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
from ClonesModelling.id_test.classifier.classifier import read_classifier_output


def get_classifier_output(
    idt_id: str,
    pt_subset: str,  # nature_genetics_patients,nature_patients,total,status_representatives
    classifier: str,
    n_features_per_dist_fn: int,
    replicate_index: int,
):
    return read_classifier_output(
        # logs/idt_2024-10-03_19-04-47/classifiers/pt_subsets/nature_genetics_patients_distance/
        # including_true_data/all_replicates/all_paradigms/
        # df_subsets/2ws_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_rc_sc_sso_w_zv/
        # all_simulations/classifier_outputs/logistic_regression/2_features_per_dist_fn/replicate_0/output.npz
        f"logs/{idt_id}/classifiers/pt_subsets/{pt_subset}_distance/including_true_data/"
        "all_replicates/all_paradigms/df_subsets/"
        + (
            "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv"
            if pt_subset != "nature_genetics_patients"
            else "2ws_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_w_zv"
        )
        +
        # f"{'all_dfs' if pt_subset != 'nature_genetics_patients' else '2ws_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_rc_sc_sso_w_zv'}/"
        f"/all_simulations/classifier_outputs/{classifier}/"
        f"{n_features_per_dist_fn}_features_per_dist_fn/replicate_{replicate_index}"
        "/output.npz"
    )  # 2ws_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_w_zv


def get_iterable():
    return (
        (pt_subset, classifier, n_features_per_dist_fn, replicate_index)
        for pt_subset in [
            "nature_genetics_patients",
            "nature_patients",
            "total",
            "status_representatives",
        ]
        for classifier in [
            "logistic_regression",
            "support_vector_machine",
            "random_forest",
        ]
        for n_features_per_dist_fn in [2, 3, 5, 10, 20]
        for replicate_index in range(3)
    )


classifier_output_df = pd.DataFrame(
    {
        "pt_subset": pt_subset,
        "classifier": classifier,
        "n_features_per_dist_fn": n_features_per_dist_fn,
        "replicate_index": replicate_index,
        "cross_val_score": get_classifier_output(
            # idt_id="idt_2024-10-03_19-04-47",
            idt_id="idt_2025-04-07_10-16-59",
            pt_subset=pt_subset,
            classifier=classifier,
            n_features_per_dist_fn=n_features_per_dist_fn,
            replicate_index=replicate_index,
        ).cross_val_score,
    }
    for pt_subset, classifier, n_features_per_dist_fn, replicate_index in get_iterable()
).assign(
    cross_val_score=lambda df: df.cross_val_score.astype(float),
)
classifier_output_df

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    data=classifier_output_df,
    x="pt_subset",
    y="cross_val_score",
    hue="classifier",
    ax=ax,
    order=[
        "total",
        "nature_patients",
        "status_representatives",
        "nature_genetics_patients",
    ],
)
plt.xticks(rotation=15, ha="right")
ax.set_ylim(0, None)

# %%
# separate plot for each classifier, coloured by n_features_per_dist_fn
labels = {
    "total": "Combined cohort",
    "nature_patients": "Yoshida et al. (2020)",
    "status_representatives": "Status representatives",
    "nature_genetics_patients": "Huang et al. (2022)",
}
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3), sharey=True)
for i, classifier in enumerate(
    ["logistic_regression", "support_vector_machine", "random_forest"]
):
    sns.boxplot(
        data=classifier_output_df[classifier_output_df.classifier == classifier],
        x="pt_subset",
        y="cross_val_score",
        hue="n_features_per_dist_fn",
        ax=axes[i],
        order=[
            "total",
            "nature_patients",
            "status_representatives",
            "nature_genetics_patients",
        ],
    )
    axes[i].set_title(classifier.replace("_", "\n").title().replace("t\n", "t "))
    axes[i].set_xlabel("Patient subset")
    axes[i].set_xticklabels(
        [
            # subset_name.get_text().replace("_", " ").title()
            labels[subset_name.get_text()]
            for subset_name in axes[i].get_xticklabels()
        ],
        rotation=30,
        ha="right",
    )
    axes[i].axhline(1 / 24, linestyle="--", color="black", alpha=0.5)
    axes[i].text(0, 1 / 24, "Random baseline", verticalalignment="bottom")
    if i == 2:
        axes[i].legend(
            loc="center left",
            title="Features per metric",
            ncol=2,
            bbox_to_anchor=(1, 0.5),
        )
    else:
        axes[i].get_legend().remove()
axes[0].set_ylabel("Cross-validation accuracy")
axes[0].set_ylim(0, None)
fig.tight_layout()
fig.subplots_adjust(wspace=0.1)
save_dir = "notebooks/plots/24-11-12_idt_pt_subset_plots"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/accuracy_by_classifier_n_features.pdf")

# %%
df_groups = {
    "Mutational burden": [
        "wasserstein",
        "smoking_sig_only",
        "z_values",
        "mean_subtracted",
        "mean_subtracted_2D_simplified",
        "2D_wasserstein_simplified",
    ],
    "Mixture model": [
        "mm_larger_weight_sq_diff",
        "mm_larger_weight_abs_diff",
        "mm_dominant_means_sq_diff",
        "mm_larger_means_sq_diff",
        "mm_smaller_means_sq_diff",
        "mm_weighted_means_by_dominance",
        "mm_weighted_means_by_position",
    ],
    "Phylogeny": [
        "total_branch_length",
        "branch_length_wasserstein",
        "abs_j_one",
        "l2_j_one",
    ],
    "Control": ["random_control"],
}
df_group_colours = {
    "Mutational burden": "tab:red",
    "Mixture model": "tab:green",
    "Phylogeny": "tab:blue",
    "Control": "tab:gray",
}
print(f"total length: {sum(len(group) for group in df_groups.values())}")

# %%
df_names = {
    True: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv".split(
        "_"
    ),
    False: "2ws_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_w_zv".split(
        "_"
    ),
}
len(df_names[True]), len(df_names[False])

# %%
len(get_classifier_output(
    idt_id="idt_2024-10-03_19-04-47",
    pt_subset="nature_genetics_patients",
    classifier="logistic_regression",
    n_features_per_dist_fn=2,
    replicate_index=0,
).dist_fn_importances)

# %%
dist_fn_importances_df = pd.concat(
    pd.DataFrame(
        {
            "pt_subset": [pt_subset]
            * len(df_names[pt_subset != "nature_genetics_patients"]),
            "classifier": [classifier]
            * len(df_names[pt_subset != "nature_genetics_patients"]),
            "n_features_per_dist_fn": [n_features_per_dist_fn]
            * len(df_names[pt_subset != "nature_genetics_patients"]),
            "replicate_index": [replicate_index]
            * len(df_names[pt_subset != "nature_genetics_patients"]),
            "dist_fn_importance": list(
                get_classifier_output(
                    idt_id="idt_2024-10-03_19-04-47",
                    pt_subset=pt_subset,
                    classifier=classifier,
                    n_features_per_dist_fn=n_features_per_dist_fn,
                    replicate_index=replicate_index,
                ).dist_fn_importances
            ),
            "dist_fn_names": df_names[pt_subset != "nature_genetics_patients"],
        }
    )
    for pt_subset, classifier, n_features_per_dist_fn, replicate_index in get_iterable()
).assign(
    df_group=lambda df: df.dist_fn_names.map(
        lambda name: next(
            group
            for group, names in df_groups.items()
            if any(name == "".join(x[0] for x in n.split("_")) for n in names)
        )
    )
)
dist_fn_importances_df

# %%
# group by pt_subset,classifier,n_features_per_dist_fn,colour_group
# sum dist_fn_importance, plot as boxplot with x=pt_subset, y=summed dist_fn_importance, hue=colour_group
# separate plot for each classifier
# for n_features_per_dist_fn in [2, 3, 5, 10, 20]:
fig, axes = plt.subplots(1, 3, figsize=(7, 3.3), sharey=True)
plot_data = (
    dist_fn_importances_df.groupby(
        ["pt_subset", "classifier", "n_features_per_dist_fn", "df_group"]
    )
    .dist_fn_importance.sum()
    .reset_index()
    .assign(
        dist_fn_importance=lambda df: df.dist_fn_importance
        / df.groupby(
            ["pt_subset", "classifier", "n_features_per_dist_fn"]
        ).dist_fn_importance.transform("sum"),
        pt_subset=lambda df: pd.Categorical(
            df.pt_subset,
            categories=[
                "total",
                "nature_patients",
                "status_representatives",
                "nature_genetics_patients",
            ],
            ordered=True,
        ),
    )
)
for i, classifier in enumerate(
    ["logistic_regression", "support_vector_machine", "random_forest"]
):
    sns.lineplot(
        # sns.boxplot(
        data=plot_data[plot_data.classifier == classifier],
        x="pt_subset",
        y="dist_fn_importance",
        hue="df_group",
        palette=df_group_colours,
        ax=axes[i],
        legend=True,
    )
    axes[i].set_xticklabels(
        [
            "total",
            "nature_patients",
            "status_representatives",
            "nature_genetics_patients",
        ],
        rotation=15,
        ha="right",
    )
    axes[i].set_title(classifier.replace("_", "\n").title().replace("t\n", "t "))
    # axes[i].set_title("".join([word[0] for word in classifier.split("_")]).upper())
    axes[i].set_xlabel("Patient subset" if i == 1 else "")
    axes[i].set_xticklabels(
        [
            # subset_name.get_text().replace("_", " ").title()
            labels[subset_name.get_text()]
            for subset_name in axes[i].get_xticklabels()
        ],
        rotation=30,
        ha="right",
    )
    if i == 2:
        axes[i].legend(
            loc="center left", title="Distance function group", bbox_to_anchor=(1, 0.5)
        )
    else:
        axes[i].get_legend().remove()
axes[0].set_ylabel("Total Feature Importance")
axes[0].set_ylim(0, None)
fig.tight_layout()
fig.subplots_adjust(wspace=0.2)
# print(n_features_per_dist_fn)
display(fig)
save_dir = "notebooks/plots/24-11-12_idt_pt_subset_plots"
os.makedirs(save_dir, exist_ok=True)
fig.savefig(f"{save_dir}/feature_importance_by_classifier.pdf")
plt.close(fig)

# %% [markdown]
# 22/4/25 thought: line plot is wrong here bc you can't interpolate...maybe leave that though
