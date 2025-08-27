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

# "logs/idt_2024-10-03_19-04-47/classifiers/all_pts/including_true_data/all_replicates/all_paradigms/df_subsets/blw_mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_tbl/all_simulations/classifier_outputs/aggregated_distances/robust_range_1_normalised/vif_threshold_20.npz"


def get_aggregated_distances(
    idt_id: str,
    df_percentile_threshold: int | None,
    protection_selection: bool,
    vif_threshold: int,
) -> np.ndarray:
    df_subset_string = {
        ## idt_2024-10-03_19-04-47
        # 5: "blw_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        # 10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        # 25: "blw_mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_tbl",
        # None: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        ## idt_2025-02-05_14-53-34
        # 5: "ajo_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
        # 10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
        # 25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_zv",
        # None: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms2s_msmsd_mwmbd_mwmbp_sso_w_zv",
        ## idt_2025-04-07_10-16-59
        0: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        20: "mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w",
        25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp",
        None: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv",
    }[df_percentile_threshold]
    return np.load(
        f"logs/{idt_id}/classifiers/all_pts/including_true_data/all_replicates"
        f"/{'protection_selection' if protection_selection else 'all'}_paradigms/"
        f"df_subsets/{df_subset_string}/all_simulations/classifier_outputs/aggregated_"
        f"distances/robust_range_1_normalised/vif_threshold_{vif_threshold}.npz"
    )["true_data_aggregated_distances"]


# x = get_aggregated_distances("idt_2024-10-03_19-04-47", 5, True, 20)
# x = get_aggregated_distances("idt_2025-02-05_14-53-34", 5, True, 20)
x = get_aggregated_distances("idt_2025-04-07_10-16-59", 10, True, 20)
x.shape, x

# %%
sorted_indices = np.argsort(x.flatten())
sorted_paradigm_indices = sorted_indices // x.shape[1]
k_fractions = np.array([0.01,0.02, 0.05, 0.1, 0.2, 0.5, 1])
np.array(
    [
        np.bincount(
            sorted_paradigm_indices[: int(k * len(sorted_paradigm_indices))]
        ).argmax()
        for k in k_fractions
    ]
)

# %%
np.bincount(
    sorted_paradigm_indices[: int(k_fractions[1] * len(sorted_paradigm_indices))],
    minlength=x.shape[0],
) / int(k_fractions[1] * len(sorted_paradigm_indices))

# %%
np.array(
    [
        np.bincount(
            sorted_paradigm_indices[: int(k_fraction * len(sorted_paradigm_indices))]
        )
        / int(k_fraction * len(sorted_paradigm_indices))
        for k_fraction in k_fractions
    ]
).shape


# %%
def calculate_knn_probabilities(
    idt_id: str,
    df_percentile_threshold: int,
    protection_selection: bool,
    vif_threshold: int,
    k_fractions: np.ndarray,
) -> np.ndarray:
    # print(df_percentile_threshold, protection_selection, vif_threshold)
    aggregated_distances = get_aggregated_distances(
        idt_id, df_percentile_threshold, protection_selection, vif_threshold
    )
    sorted_indices = np.argsort(aggregated_distances.flatten())
    sorted_paradigm_indices = sorted_indices // aggregated_distances.shape[1]
    # for k_fraction in k_fractions:
    #     k = int(k_fraction * len(sorted_paradigm_indices))
    #     paradigm_index = np.bincount(sorted_paradigm_indices[:k]).argmax()
    return np.array(
        [
            np.bincount(
                sorted_paradigm_indices[
                    : int(k_fraction * len(sorted_paradigm_indices))
                ],
                minlength=aggregated_distances.shape[0],
            )
            / int(k_fraction * len(sorted_paradigm_indices))
            for k_fraction in k_fractions
        ]
    )


# %%
# calculate_knn_probabilities("idt_2024-10-03_19-04-47", 5, False, 20, k_fractions).shape
calculate_knn_probabilities("idt_2025-04-07_10-16-59", 10, False, 20, k_fractions).shape

# %%
import seaborn as sns

from ClonesModelling.parameters.hypothetical_paradigm_class import (
    get_hypothetical_paradigm_for_each_subset,
    MODULE_ORDERING,
)

paradigm_names = {
    protection_selection: [
        hp.get_modules_string(abbreviated=False)
        for hp in get_hypothetical_paradigm_for_each_subset(
            # hypothesis_module_names=["q", "qp", "p", "ir", "sd"],
            hypothesis_module_names=[
                "quiescent",
                "quiescent_protected",
                "protected",
                "immune_response",
                "smoking_driver",
            ],
            max_modules=-1 if protection_selection else 5,
        )
    ]
    for protection_selection in [True, False]
}

paradigm_colours = [
    (paradigm_name, sns.color_palette("tab10")[ix])
    for ix, paradigm_name in enumerate(paradigm_names[True])
]
module_colours = [
    (module, sns.color_palette("tab10")[ix])
    for ix, module in enumerate(
        sorted(
            [mod for mod in MODULE_ORDERING if mod == "base" or len(mod) > 3],
            key=lambda x: (x == "quiescent-protected", MODULE_ORDERING.index(x)),
        )
    )
]
paradigm_colours, module_colours

# %%
# import pandas as pd
# import numpy as np

# aggregated_distances = pd.concat(
#     [
#         pd.DataFrame(
#             {
#                 "paradigm": np.repeat(
#                     paradigm_names[protection_selection],
#                     aggregated_distance_values.shape[1],
#                 ),
#                 "true_data_aggregated_distance": aggregated_distance_values.flatten(),
#             }
#         ).assign(
#             protection_selection=protection_selection,
#             df_percentile_threshold=df_percentile_threshold,
#             vif_threshold=str(vif_threshold),
#         )
#         for protection_selection in [True, False]
#         for df_percentile_threshold in [5, 10, 25]
#         for vif_threshold in [None, 5, 20]
#         for aggregated_distance_values in [
#             get_aggregated_distances(
#                 "idt_2024-10-03_19-04-47",
#                 df_percentile_threshold,
#                 protection_selection,
#                 vif_threshold,
#             )
#         ]
#     ]
# )
# aggregated_distances

# %%
import numpy as np
import pandas as pd

simulations_per_paradigm = 250

knn_estimates = pd.concat(
    [
        pd.DataFrame(
            {
                "k_fraction": np.repeat(k_fractions, knn_estimate_values.shape[1]),
                "knn_estimate_probability": knn_estimate_values.flatten(),
                "knn_estimate_paradigm": np.tile(
                    paradigm_names[protection_selection], knn_estimate_values.shape[0]
                ),
            }
        ).assign(
            protection_selection=protection_selection,
            df_percentile_threshold=df_percentile_threshold,
            vif_threshold=str(vif_threshold),
            neighbour_count=lambda x: x["k_fraction"].apply(
                lambda y: int(
                    y
                    * simulations_per_paradigm
                    * len(paradigm_names[protection_selection])
                )
            ),
            # knn_estimate_name=lambda x: x["knn_estimate_index"].map(
            #     {
            #         ix: paradigm_name
            #         for ix, paradigm_name in enumerate(
            #             paradigm_names[protection_selection]
            #         )
            #     }
            # ),
        )
        for protection_selection in [True, False]
        for df_percentile_threshold in [0, 10, 20, 25]
        for vif_threshold in [None, 5, 20]
        for knn_estimate_values in [
            calculate_knn_probabilities(
                # "idt_2024-10-03_19-04-47",
                "idt_2025-04-07_10-16-59",
                df_percentile_threshold,
                protection_selection,
                vif_threshold,
                k_fractions,
            )
        ]
    ]
)
knn_estimates

# %%
import matplotlib.pyplot as plt

for protection_selection in [True, False]:
    # for k_fraction in k_fractions:
    fig, axes = plt.subplots(
        1,
        len(knn_estimates["df_percentile_threshold"].unique()),
        figsize=(9.8, 2.8),
        sharey=True,
    )
    if len(knn_estimates["df_percentile_threshold"].unique()) == 1:
        axes = [axes]
    for i, df_percentile_threshold in enumerate(
        knn_estimates["df_percentile_threshold"].unique()
    ):
        df = knn_estimates[
            (knn_estimates["df_percentile_threshold"] == df_percentile_threshold)
            & (knn_estimates["vif_threshold"] == "20")
            & (knn_estimates["protection_selection"] == protection_selection)
        ]
        sns.lineplot(
            data=(
                df
                if protection_selection
                else df.assign(
                    module=lambda x: x["knn_estimate_paradigm"].str.split("-")
                )
                .explode("module")
                .groupby(["neighbour_count", "module"])[["knn_estimate_probability"]]
                # .groupby(["k_fraction", "module"])[["knn_estimate_probability"]]
                .sum().reset_index()
            ),
            x="neighbour_count",
            # x="k_fraction",
            y="knn_estimate_probability",
            hue="knn_estimate_paradigm" if protection_selection else "module",
            ax=axes[i],
            palette=[
                colour
                for _, colour in (
                    paradigm_colours if protection_selection else module_colours
                )
            ],
            hue_order=[
                name
                for name, _ in (
                    paradigm_colours if protection_selection else module_colours
                )
            ],
        )
        sns.scatterplot(
            data=(
                df
                if protection_selection
                else df.assign(
                    module=lambda x: x["knn_estimate_paradigm"].str.split("-")
                )
                .explode("module")
                .groupby(["neighbour_count", "module"])[["knn_estimate_probability"]]
                # .groupby(["k_fraction", "module"])[["knn_estimate_probability"]]
                .sum().reset_index()
            ),
            # data=knn_estimates[
            #     (knn_estimates["df_percentile_threshold"] == df_percentile_threshold)
            #     & (knn_estimates["vif_threshold"] == "20")
            #     & (knn_estimates["protection_selection"] == protection_selection)
            # ],
            x="neighbour_count",
            # x="k_fraction",
            y="knn_estimate_probability",
            hue="knn_estimate_paradigm" if protection_selection else "module",
            ax=axes[i],
            palette=[
                colour
                for _, colour in (
                    paradigm_colours if protection_selection else module_colours
                )
            ],
            hue_order=[
                name
                for name, _ in (
                    paradigm_colours if protection_selection else module_colours
                )
            ],
            legend=False,
            s=10,
        )
        axes[i].set_title(
            (
                f"DF threshold={df_percentile_threshold}%"
                if df_percentile_threshold != "None"
                else "All DFs"
            ),
            fontsize=10,
        )
        axes[i].set_ylabel("Fraction of neighbours")
        if i == 0:  # 1:
            # axes[i].set_xlabel("Neighbour fraction cutoff")
            axes[i].set_xlabel("Neighbourhood size cutoff")
        else:
            axes[i].set_xlabel(None)
        # axes[i].set_ylim(0, 1)
        axes[i].set_xscale("log")
        if i < len(knn_estimates["df_percentile_threshold"].unique()) - 1:
            axes[i].get_legend().remove()
            # pass
        else:
            axes[i].legend(
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                title="Paradigm" if protection_selection else "Module",
                title_fontsize="medium",
            )
    for i in range(len(knn_estimates["df_percentile_threshold"].unique())):
        axes[i].axvline(
            0.05 * simulations_per_paradigm * len(paradigm_names[protection_selection]),
            color="black",
            linestyle="--",
        )
        axes[i].text(
            0.055
            * simulations_per_paradigm
            * len(paradigm_names[protection_selection]),
            0.95 * axes[i].get_ylim()[1],
            "5% of\nsimulations",
            horizontalalignment="left",
            verticalalignment="top",
        )

    fig.tight_layout()
    print("protection_selection:", protection_selection)
    display(fig)
    save_dir = "notebooks/plots/24-11-28_plot_aggregated_dist_true_data"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/{'paradigm' if protection_selection else 'module'}_level.pdf",
    )
    plt.close(fig)

# %%
# now restrict to neighbourhood size of 5% of simulations, and lineplot with x=df_percentile_threshold, y=knn_estimate_probability, hue=knn_estimate_paradigm
for protection_selection in [True, False]:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    df = knn_estimates[
        (knn_estimates["vif_threshold"] == "20")
        & (knn_estimates["protection_selection"] == protection_selection)
    ]
    sns.lineplot(
        data=(
            df
            if protection_selection
            else df.assign(module=lambda x: x["knn_estimate_paradigm"].str.split("-"))
            .explode("module")
            .groupby(["df_percentile_threshold", "module"])[
                ["knn_estimate_probability"]
            ]
            .sum()
            .reset_index()
        ),
        x="df_percentile_threshold",
        y="knn_estimate_probability",
        hue="knn_estimate_paradigm" if protection_selection else "module",
        ax=ax,
        palette=[
            colour
            for _, colour in (
                paradigm_colours if protection_selection else module_colours
            )
        ],
        hue_order=[
            name
            for name, _ in (
                paradigm_colours if protection_selection else module_colours
            )
        ],
    )
    sns.scatterplot(
        data=(
            df.groupby(["df_percentile_threshold", "knn_estimate_paradigm"])[
                "knn_estimate_probability"
            ]
            .mean()
            .reset_index()
            if protection_selection
            else df.assign(module=lambda x: x["knn_estimate_paradigm"].str.split("-"))
            .explode("module")
            .groupby(["df_percentile_threshold", "module"])["knn_estimate_probability"]
            .mean()
            .reset_index()
        ),
        x="df_percentile_threshold",
        y="knn_estimate_probability",
        hue="knn_estimate_paradigm" if protection_selection else "module",
        ax=ax,
        palette=[
            colour
            for _, colour in (
                paradigm_colours if protection_selection else module_colours
            )
        ],
        hue_order=[
            name
            for name, _ in (
                paradigm_colours if protection_selection else module_colours
            )
        ],
        legend=False,
        s=20,
    )
    ax.set_ylabel("Fraction of 5% nearest neighbours\n")
    ax.set_xlabel("DF threshold (%)")
    ax.set_ylim(0, None)
    ax.set_xticks([0, 10, 20, 25])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Paradigm" if protection_selection else "Module",
    )
    ax.axhline(
        1 / len(paradigm_names[protection_selection]), color="black", linestyle=":"
    )

    fig.tight_layout()
    print("protection_selection:", protection_selection)
    display(fig)
    save_dir = "notebooks/plots/24-11-28_plot_aggregated_dist_true_data"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/{'paradigm' if protection_selection else 'module'}_level_df.pdf",
    )
    plt.close(fig)

# %%
# print out filepaths for the plot for each df percentile threshold
for protection_selection in [True, False]:
    for df_percentile_threshold in knn_estimates["df_percentile_threshold"].unique():
        df_subset_string = {
            0: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            20: "mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w",
            25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp",
        }[int(df_percentile_threshold)]
        print(protection_selection, df_percentile_threshold)
        print(
            f"logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/"
            f"all_replicates/{'protection_selection' if protection_selection else 'all'}_paradigms/"
            f"df_subsets/{df_subset_string}/all_simulations/plots/aggregated/robust_range_1_normalised/"
            f"vif_threshold_20/confusion_heatmaps/clustered/top_5%.pdf"
        )
