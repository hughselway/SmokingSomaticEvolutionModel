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

# dist_fn_importances = pd.read_csv("logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv/all_simulations/dist_fn_importances.csv")
# dist_fn_importances = pd.read_csv("logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv/all_simulations/dist_fn_importances.csv")
# now one in the 25th percentile threshold dfs
percentile_threshold = 25
dist_fn_importances = pd.read_csv(
    "logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/"
    + {
        0: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        10: "2ws_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        20: "2ws_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w",
        25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp",
    }[percentile_threshold]
    + "/all_simulations/dist_fn_importances.csv",
)
if percentile_threshold != 0:
    # then support vector machine importance calculation was skipped for speed
    dist_fn_importances = dist_fn_importances.loc[
        lambda x: x["classifier_name"] != "support_vector_machine"
    ]
dist_fn_importances

# %%
# group by distance_function_name and sum the importances
dist_fn_importances.groupby(['classifier_name','n_features_per_dist_fn'])['dist_fn_importance'].sum().reset_index().assign(
    prod=lambda x: x['dist_fn_importance'] * (x['n_features_per_dist_fn'])**0.5
)

# %%
# for each n_features_per_dist_fn,mds_replicate_index,classifier_name combo
# rank the distance functions by dist_fn_importance, add new column called rank
dist_fn_importances["rank"] = dist_fn_importances.groupby(
    ["n_features_per_dist_fn", "mds_replicate_index", "classifier_name"]
)["dist_fn_importance"].rank(ascending=True, method="first") / len(
    dist_fn_importances["distance_function_name"].unique()
)
dist_fn_importances

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    y="distance_function_name",
    x="rank",
    data=dist_fn_importances,
    order=dist_fn_importances.groupby("distance_function_name")["rank"]
    .median()
    .sort_values(ascending=False)
    .index,
    hue="classifier_name",
    ax=ax,
)

# %%
fig, ax = plt.subplots(
    1,
    len(dist_fn_importances["classifier_name"].unique()),
    figsize=(
        8.6,
        1 + 0.15 * len(dist_fn_importances["distance_function_name"].unique()),
    ),
    sharey=True,
    sharex=True,
)
for i, classifier_name in enumerate(dist_fn_importances["classifier_name"].unique()):
    sns.boxplot(
        y="distance_function_name",
        x="rank",
        data=dist_fn_importances[
            dist_fn_importances["classifier_name"] == classifier_name
        ],
        order=dist_fn_importances.groupby("distance_function_name")["rank"]
        .median()
        .sort_values(ascending=False)
        .index,
        ax=ax[i],
    )
    ax[i].set_ylabel("Distance Function")
    ax[i].set_xlabel("Rank importance")
    # if i == len(dist_fn_importances["classifier_name"].unique()) - 1:
    #     ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # else:
    #     ax[i].legend().remove()
    ax[i].set_xlim(-0.01, 1.01)
    ax[i].set_title(classifier_name.replace("_", " ").title(), fontsize=10)
# reduce padding between subplots
fig.tight_layout()
plt.subplots_adjust(wspace=0.15)
save_dir = "notebooks/plots/24-11-21_df_importance_unified_plot"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(
    f"{save_dir}/rank.pdf"
    if percentile_threshold == 0
    else f"{save_dir}/rank_{percentile_threshold}.pdf"
)
display(fig)
plt.close(fig)

# %%
fig, ax = plt.subplots(
    1,
    len(dist_fn_importances["classifier_name"].unique()),
    figsize=(8, 6),
    sharey=True,
    sharex=True,
)
for i, classifier_name in enumerate(dist_fn_importances["classifier_name"].unique()):
    sns.boxplot(
        y="distance_function_name",
        x="rank",
        data=dist_fn_importances[
            dist_fn_importances["classifier_name"] == classifier_name
        ],
        hue="n_features_per_dist_fn",
        order=dist_fn_importances.groupby("distance_function_name")["rank"]
        .median()
        .sort_values(ascending=False)
        .index,
        ax=ax[i],
    )
    if i == len(dist_fn_importances["classifier_name"].unique()) - 1:
        ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        ax[i].legend().remove()
    ax[i].set_xlim(-0.01, 1.01)
    ax[i].set_title(classifier_name.replace("_", " ").title())

# %%
# boxplot df_name against rank, hue=n_features_per_dist_fn
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    y="distance_function_name",
    x="rank",
    data=dist_fn_importances,
    order=dist_fn_importances.groupby("distance_function_name")["rank"]
    .median()
    .sort_values(ascending=False)
    .index,
    hue="n_features_per_dist_fn",
    ax=ax,
)


# %%
# instead, additively normalise within the classifier_name,mds_replicate_index,n_features_per_dist_fn group
dist_fn_importances["dist_fn_importance_normalised"] = dist_fn_importances.groupby(
    ["classifier_name", "mds_replicate_index", "n_features_per_dist_fn"]
)["dist_fn_importance"].transform(lambda x: x / x.sum())
dist_fn_importances

# %%
# now boxplot normalised dist_fn_importance against distance_function_name
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    y="distance_function_name",
    x="dist_fn_importance_normalised",
    data=dist_fn_importances,
    order=dist_fn_importances.groupby("distance_function_name")["dist_fn_importance_normalised"]
    .median()
    .sort_values(ascending=False)
    .index,
    hue="classifier_name",
    ax=ax,
)
ax.set_xlabel("Normalised importance score")
ax.set_ylabel("Distance function")

# %%
# separate boxplots for each classifier_name
fig, ax = plt.subplots(
    1,
    len(dist_fn_importances["classifier_name"].unique()),
    figsize=(
        8.6,
        1 + 0.15 * len(dist_fn_importances["distance_function_name"].unique()),
    ),
    sharey=True,
    sharex=True,
)
for i, classifier_name in enumerate(dist_fn_importances["classifier_name"].unique()):
    sns.boxplot(
        y="distance_function_name",
        x="dist_fn_importance_normalised",
        data=dist_fn_importances[
            dist_fn_importances["classifier_name"] == classifier_name
        ],
        order=dist_fn_importances.groupby("distance_function_name")[
            "dist_fn_importance_normalised"
        ]
        .median()
        .sort_values(ascending=False)
        .index,
        ax=ax[i],
    )
    # ax[i].set_xlim(0, None)
    # if i == len(dist_fn_importances["classifier_name"].unique()) - 1:
    #     ax[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # else:
    #     ax[i].legend().remove()
    ax[i].set_title(classifier_name.replace("_", " ").title(), fontsize=10)
    ax[i].set_xlabel("Importance score")
    ax[i].set_ylabel("Distance function")
fig.tight_layout()
plt.subplots_adjust(wspace=0.15)
save_dir = "notebooks/plots/24-11-21_df_importance_unified_plot"
os.makedirs(save_dir, exist_ok=True)
plt.savefig(
    f"{save_dir}/normalised_value.pdf"
    if percentile_threshold == 0
    else f"{save_dir}/normalised_value_{percentile_threshold}_threshold.pdf"
)
display(fig)
plt.close(fig)
