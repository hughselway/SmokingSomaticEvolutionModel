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

# %% [markdown]
# Idea here is, at each observed data extremity threshold, to assess which values of certain parameters are associated with being closer to the true data.
#
# To get this, I'll need to read in all the parameter values (from the simulation commands), then make some basic plots.

# %%
import os
os.chdir("/Users/hughselway/Documents/ClonesModelling")

# %%
from ClonesModelling.id_test.read_data import IdentifiabilityDataset

# idt_id = "idt_2025-04-07_10-16-59"
# idt_id = "param_fit_2025-06-06"
idt_id = "param_fit_2025-07-29"
id = IdentifiabilityDataset("logs", idt_id)
id.paradigm_simulation_lists["q-ir"][0].parameter_values

# %%
# IdentifiabilityDataset("logs", "param_fit_2025-07-29").paradigm_simulation_lists[
#     "q-ir"
# ][0].replicates.data_path
# # check that this for 0 up to 9999 is a sorted list
# data_paths = [
#     IdentifiabilityDataset("logs", "param_fit_2025-07-29")
#     .paradigm_simulation_lists["q-ir"][i]
#     .replicates.data_path
#     for i in range(10000)
# ]
# sorted_data_paths = sorted(data_paths)
# assert data_paths == sorted_data_paths, "Data paths are not sorted correctly."

# %%
all_parameters = set().union(
    *[
        set(p.parameter_values.keys())
        for paradigm in id.paradigm_simulation_lists
        for p in id.paradigm_simulation_lists[paradigm]
    ]
)
all_parameters

# %%
from ClonesModelling.parameters.parameter_class import get_parameters

parameters = {p.name: p for p in get_parameters()}
full_name = {
    "".join(x[0] for x in name.split("_")): name for name in parameters
}
parameters, full_name

# %%
# first to get a feel for the data, let's plot histograms of each parameter's values
import matplotlib.pyplot as plt
from ClonesModelling.visualisation.parameter_prior import add_prior_distribution

for paradigm, simulation_list in id.paradigm_simulation_lists.items():
    posterior_samples = {parameter: [] for parameter in all_parameters}
    for simulation in simulation_list:
        for parameter_name, value in simulation.parameter_values.items():
            posterior_samples[parameter_name].append(value)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len([values for values in posterior_samples.values() if values]),
        sharey=True,
        figsize=(
            1 + len([values for values in posterior_samples.values() if values]) * 1.1,
            2,
        ),
    )
    prior_axes = [ax.twinx() for ax in axes]
    axes[0].set_ylabel("Count")
    fig.suptitle(paradigm)
    for ix, (parameter_name, values) in enumerate(
        [(key, values) for key, values in posterior_samples.items() if values]
    ):
        axes[ix].hist(values, bins=20, alpha=0.5, label=parameter_name)
        axes[ix].set_xlabel(parameter_name)
        if parameter_name not in ["fcs", "qgcc"]:
            add_prior_distribution(prior_axes[ix], parameters[full_name[parameter_name]])
    # equalise the y-axis across all subplots
    max_y = max(ax.get_ylim()[1] for ax in prior_axes)
    for ax in prior_axes[:-1]:
        ax.set_ylim(0, max_y)
        ax.set_yticks([])
    prior_axes[-1].set_ylim(0, max_y)
    prior_axes[-1].set_ylabel("Prior density")

    fig.tight_layout()
    display(fig)
    plt.close(fig)

# %%
import pandas as pd

true_data_distances = {
    distance_function: pd.read_csv(
        f"logs/{idt_id}/distance/{distance_function}/true_data_distances.csv"
    )
    for distance_function in os.listdir(f"logs/{idt_id}/distance")
    if os.path.isdir(f"logs/{idt_id}/distance/{distance_function}")
    and distance_function
    not in ["hpc", "total_branch_length", "l2_j_one", "mm_larger_weight_sq_diff"]
}

# %%
import numpy as np
from ClonesModelling.parameters.parameter_class import abbreviate_name


def get_df_subset_string(
    idt_id: str,
    df_percentile_threshold: int | None,
) -> str:
    mapping = {
        "idt_2024-10-03_19-04-47": {
            5: "blw_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            25: "blw_mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_tbl",
            None: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
        },
        "idt_2025-02-05_14-53-34": {
            5: "ajo_blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
            10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_w_zv",
            25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_zv",
            None: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms2s_msmsd_mwmbd_mwmbp_sso_w_zv",
        },
        "idt_2025-04-07_10-16-59": {
            0: "ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            10: "blw_mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w_zv",
            20: "mdmsd_mlmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp_sso_tbl_w",
            25: "mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp",
            None: "2ws_ajo_blw_ljo_mdmsd_mlmsd_mlwad_mlwsd_ms_ms2s_msmsd_mwmbd_mwmbp_rc_sso_tbl_w_zv",
        },
    }
    if idt_id not in mapping:
        assert idt_id.startswith("param_fit_")
        return mapping["idt_2025-04-07_10-16-59"][df_percentile_threshold]
    return mapping[idt_id][df_percentile_threshold]


def get_aggregated_distances(
    idt_id: str,
    df_percentile_threshold: int | None,
    protection_selection: bool,
    vif_threshold: int | None,
) -> np.ndarray:
    df_subset_string = get_df_subset_string(idt_id, df_percentile_threshold)
    if idt_id != "param_fit_2025-07-29":  # for which there aren't npz files
        return np.load(
            f"logs/{idt_id}/classifiers/all_pts/including_true_data/all_replicates"
            f"/{'protection_selection' if protection_selection else 'all'}_paradigms/"
            f"df_subsets/{df_subset_string}/all_simulations/classifier_outputs/aggregated_"
            f"distances/robust_range_1_normalised/vif_threshold_{vif_threshold}.npz"
        )["true_data_aggregated_distances"]
    # otherwise, we need to aggregate the distances manually
    # raise NotImplementedError()
    assert vif_threshold is None
    assert idt_id == "param_fit_2025-07-29"
    all_true_data_distances = (
        pd.concat(
            pd.read_csv(
                f"logs/{idt_id}/distance/{full_df_name}/true_data_distances.csv"
            )[["replicate_index", "distance"]].assign(
                full_df_name=full_df_name,
                df_name=abbreviate_name(full_df_name),
                normalised_distance=lambda df: (
                    df["distance"]
                    / (df["distance"].quantile(0.995) - df["distance"].quantile(0.005))
                ),
            )
            for full_df_name in os.listdir(f"logs/{idt_id}/distance")
            if abbreviate_name(full_df_name) in df_subset_string.split("_")
        )
        .reset_index(drop=True)
        .groupby(["replicate_index"])["normalised_distance"]
        .mean()
        .reset_index()
    )
    display(all_true_data_distances)
    # sort by replicate index, then return as numpy array
    agg_dist = all_true_data_distances.sort_values("replicate_index")[
        "normalised_distance"
    ].to_numpy()
    return [agg_dist]


# %%
# np.load(
#     "logs/idt_2025-04-07_10-16-59/classifiers/all_pts/including_true_data/all_replicates/protection_selection_paradigms/df_subsets/mdmsd_mlwad_mlwsd_msmsd_mwmbd_mwmbp/all_simulations/independent_distance_functions/vif_threshold_20/iteration_3.npz"
# )["distance_function_indices"]

# %%
paradigm_names = ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"]

# nearest_neighbour_fraction = 0.025

agg_dist = get_aggregated_distances(
    idt_id,
    df_percentile_threshold=0,
    protection_selection=False,
    # vif_threshold=20,
    vif_threshold=None,  # for the param_fit_2025-07-29 case
)[
    0
]  # [paradigm_names.index("q-ir")]
low_dist_indices = {
    nearest_neighbour_fraction: np.argsort(agg_dist)[
        : int(len(agg_dist) * nearest_neighbour_fraction)
    ]
    for nearest_neighbour_fraction in [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
}
low_dist_parameters = {
    restricted_params: {
        nearest_neighbour_fraction: pd.DataFrame(
            [
                id.paradigm_simulation_lists["q-ir"][i].parameter_values
                for i in low_dist_indices[nearest_neighbour_fraction]
            ]
        ).loc[
            :,
            (
                ["qf", "qdpy", "qgcc", "aqdpy", "idr", "sic"]
                if restricted_params
                else list(all_parameters)
            ),
        ]
        for nearest_neighbour_fraction in low_dist_indices
    }
    for restricted_params in [True, False]
}
# low_dist_parameters

# %%
agg_dist.shape

# %%
# print out mean values
mean_values = pd.concat(
    df.mean().to_frame().T.assign(nearest_neighbour_fraction=nearest_neighbour_fraction)
    for nearest_neighbour_fraction, df in low_dist_parameters[False].items()
).set_index("nearest_neighbour_fraction")
display(mean_values)
percentile_values = pd.concat(
    df.quantile([0.025, 0.5, 0.975])
    .unstack()
    .to_frame()
    .T.assign(nearest_neighbour_fraction=nearest_neighbour_fraction)
    for nearest_neighbour_fraction, df in low_dist_parameters[False].items()
).set_index("nearest_neighbour_fraction")
percentile_values

# %%
percentile_values['qdpy'][0.975]

# %%
# plot each parameter, with mean and 95% CI - separate plot for each param, x=nearest_neighbour_fraction, y=parameter value
import textwrap

for restricted_params in [False, True]:
    params_to_plot = low_dist_parameters[restricted_params][0.025].columns
    nrow = 3
    ncol = len(params_to_plot) // nrow + (1 if len(params_to_plot) % nrow else 0)
    fig, axes = plt.subplots(
        nrows=nrow, ncols=ncol, sharex=True, figsize=(ncol * 2.5, nrow * 1.9)
    )
    for ix, parameter in enumerate(params_to_plot):
        ax = axes.flatten()[ix]
        ax.set_xscale("log")
        mean_values[parameter].plot(
            ax=ax, label="Mean", marker="o", linestyle="-", markersize=3
        )
        ax.fill_between(
            percentile_values.index,
            percentile_values[parameter][0.025],
            percentile_values[parameter][0.975],
            # color="blue",
            alpha=0.2,
            label="95% CI",
        )

        # ax.set_title(parameter)
        ax.set_xlabel("Nearest neighbour fraction")
        ax.set_ylabel(
            textwrap.fill(full_name[parameter].replace("_", " ").title(), width=20)
        )
        if ix == 0:
            ax.legend()
        ax.set_xticks(percentile_values.index)
        ax.set_xticklabels(
            [f"{nnf:.0%}" for nnf in percentile_values.index],
            rotation=45,
        )
    fig.tight_layout()
    display(fig)
    plt.close(fig)

# %%
import textwrap
from ClonesModelling.parameters.parameter_class import get_parameters, abbreviate_name

parameters = {p.name: p for p in get_parameters()}
for restricted_params in low_dist_parameters:
    for nearest_neighbour_fraction in low_dist_parameters[restricted_params]:
        params_to_plot = low_dist_parameters[restricted_params][
            nearest_neighbour_fraction
        ].columns

        # histogram of the parameters in the low distance simulations
        nrow = 3
        ncol = len(params_to_plot) // nrow + (1 if len(params_to_plot) % nrow else 0)
        fig, axes = plt.subplots(
            nrows=nrow,
            ncols=ncol,
            sharey="row",
            figsize=(3 + 1 + ncol * 1.3, 2 + nrow * 0.9),
        )
        prior_axes = [ax.twinx() for ax in axes.flatten()]
        for ix, parameter_name in enumerate(params_to_plot):
            axes.flatten()[ix].hist(
                low_dist_parameters[restricted_params][nearest_neighbour_fraction][
                    parameter_name
                ].dropna(),
                bins=None if parameter_name != "qgcc" else [0, 2, 4, 6, 29, 31],
                alpha=0.4,
                label=parameter_name,
                color="C0",
            )
            axes.flatten()[ix].set_xlabel(
                textwrap.fill(
                    full_name[parameter_name].replace("_", " ").title(),
                    width=18,
                    break_long_words=False,
                    replace_whitespace=False,
                )
            )
            if parameter_name not in ["fcs", "qgcc"]:
                add_prior_distribution(
                    prior_axes[ix],
                    parameters[full_name[parameter_name]],
                    c="C1",
                    alpha=0.6,
                )
                axes.flatten()[ix].axvline(
                    low_dist_parameters[restricted_params][nearest_neighbour_fraction][
                        parameter_name
                    ].mean(),
                    color="C0",
                    linestyle="--",
                    label="Posterior Mean",
                )
                axes.flatten()[ix].axvline(
                    parameters[full_name[parameter_name]].convert_from_varying_scale(
                        parameters[full_name[parameter_name]].prior_dict["kwargs"][
                            "loc"
                        ]
                    ),
                    color="C1",
                    linestyle="--",
                    label="Prior Mean",
                )
            elif parameter_name == "qgcc":
                # scatter plot on prior ax with prior distribution: equally likely to be 1,5 or 30
                prior_axes[ix].scatter(
                    [1, 5, 30],
                    [1 / 3, 1 / 3, 1 / 3],
                    color="C1",
                    label="Prior Values",
                    s=15,
                    marker="D",
                )
                axes.flatten()[ix].set_xticks([1, 5, 30])
            else:
                prior_axes[ix].scatter(
                    [0, 0.05, 0.1],
                    [1 / 2, 1 / 4, 1 / 4],
                    color="C1",
                    label="Prior Values",
                    s=15,
                    marker="D",
                )
            # if ix == 0:
            #     axes.flatten()[ix].set_ylabel("Simulation count")
            # if ix == len(low_dist_parameters.columns) - 1:
            #     prior_axes[ix].set_ylabel("Prior density")
            # else:
            #     prior_axes[ix].set_yticklabels([])

        for row in range(nrow):
            axes[row, 0].set_ylabel("Simulation count")
            prior_axes[(row + 1) * ncol - 1].set_ylabel("Prior density")
            for col in range(ncol - 1):
                prior_axes[row * ncol + col].set_yticklabels([])

        max_y = max(ax.get_ylim()[1] for ax in prior_axes)
        for ax in prior_axes[:-1]:
            ax.set_ylim(0, max_y)
            # ax.set_yticks([])
        prior_axes[-1].set_ylim(0, max_y)
        # prior_axes[-1].set_ylabel("Prior density")

        axes[0, -1].legend(
            handles=[
                plt.Line2D([], [], color="C0", alpha=0.4),
                plt.Line2D([], [], color="C1", alpha=0.6),
                plt.Line2D([], [], color="C0", linestyle="--"),
                plt.Line2D([], [], color="C1", linestyle="--"),
                plt.Line2D([0], [0], color="C1", marker="D", linestyle=""),
            ],
            labels=[
                "Posterior Distribution",
                "Prior Distribution",
                "Posterior Mean",
                "Prior Mean",
                "Prior probability (discrete)",
            ],
            loc="center left",
            bbox_to_anchor=(1.5, 0.5),
            fontsize=8,
        )

        fig.tight_layout()
        # fig.subplots_adjust(wspace=0.2)
        print(
            f"Nearest neighbour fraction: {nearest_neighbour_fraction:.3f}, "
            f"Restricted parameters: {restricted_params}"
        )
        display(fig)
        save_dir = f"notebooks/plots/25-06-02_parameter_fit/{idt_id}/low_distance_simulation_parameters"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            f"{save_dir}/{nearest_neighbour_fraction:.3f}_"
            f"{'restricted' if restricted_params else 'all'}_params.pdf"
        )
        plt.close(fig)

# %%
# kolmogorov_smirnov test for each parameter; 1-sample test against prior distribution
from ClonesModelling.parameters.unpack_priors import get_prior_dict
from scipy.stats import kstest, chisquare
from statsmodels.stats.multitest import multipletests
import json

restricted_params = True


def kolmogorov_smirnov_test(
    posterior_samples: pd.Series, parameter_dict: dict
) -> tuple[float, float, float]:
    prior_dict = get_prior_dict(parameter_dict)
    parameter = parameters[parameter_dict["name"]]
    converted_samples = posterior_samples.apply(parameter.convert_to_varying_scale)
    test = kstest(
        converted_samples,
        "norm",
        args=(prior_dict["kwargs"]["loc"], prior_dict["kwargs"]["scale"]),
    )
    return (
        test.statistic,
        test.pvalue,
        parameter.convert_from_varying_scale(prior_dict["kwargs"]["loc"]),
    )


with open("ClonesModelling/parameters/hypothesis_module_parameters.json", "r") as f:
    parameter_dicts = json.load(f)

mean_change_p_values_dicts = []
for parameter_dict in parameter_dicts:
    if restricted_params and parameter_dict["name"] not in [
        full_name["qf"],
        full_name["qdpy"],
        full_name["qgcc"],
        full_name["aqdpy"],
        full_name["idr"],
        full_name["sic"],
    ]:
        continue
    posterior_samples = pd.Series(
        [
            id.paradigm_simulation_lists["q-ir"][i].parameter_values[
                abbreviate_name(parameter_dict["name"])
            ]
            for i in low_dist_indices[0.05]
        ]
    )
    if parameter_dict["name"] == "quiescent_gland_cell_count":
        observed_values = posterior_samples.value_counts().sort_index()
        values, expected_proportions = (
            parameter_dict["prior_dict"]["kwargs"]["values"][0],
            parameter_dict["prior_dict"]["kwargs"]["values"][1],
        )
        expected_frequencies = [
            x * len(posterior_samples) for x in expected_proportions
        ]
        chisq_result = chisquare(
            observed_values, expected_frequencies, ddof=len(values) - 1
        )
        chi_square_statistic, chi_square_pvalue = (
            chisq_result.statistic,
            chisq_result.pvalue,
        )
        print(
            f"Chi-square test for {parameter_dict['name']}: statistic={chi_square_statistic:.4f}, p-value={chi_square_pvalue:.4f}"
        )
        # mean_change_p_values_dicts.append(
        #     {
        #         "name": parameter_dict["name"],
        #         "statistic": chi_square_statistic,
        #         "pvalue": chi_square_pvalue,
        #         "restricted_params": restricted_params,
        #     }
        # )
        continue
    statistic, pvalue, prior_mean = kolmogorov_smirnov_test(
        posterior_samples, parameter_dict
    )
    # print(
    #     f"{parameter_dict['name']}: statistic={statistic:.4f}, p-value={pvalue:.4f}, "
    #     f"restricted_params={restricted_params}"
    # )
    mean_change_p_values_dicts.append(
        {
            "name": parameter_dict["name"],
            "statistic": statistic,
            "pvalue": pvalue,
            "restricted_params": restricted_params,
            "prior_mean": prior_mean,
            "posterior_mean": posterior_samples.mean(),
        }
    )

mean_change_p_values_df = pd.DataFrame(mean_change_p_values_dicts).assign(
    adjusted_pvalue=lambda df: multipletests(df["pvalue"], method="fdr_bh")[1]
)
mean_change_p_values_df

# %%
# pairplot of restricted parameters at each nearest neighbour fraction
import seaborn as sns

for nearest_neighbour_fraction in low_dist_parameters[True]:
    df = low_dist_parameters[True][nearest_neighbour_fraction]
    # remove any rows with NaN values
    df = df.dropna()
    # plot pairplot
    g = sns.pairplot(
        df,
        diag_kind="kde",
        kind="hist",
        markers="o",
        # plot_kws={"alpha": 0.5, "s": 10},
        height=1,
    )
    g.figure.suptitle(
        f"Nearest neighbour fraction: {nearest_neighbour_fraction:.3f}",
        y=1.02,
    )
    g.figure.tight_layout()
    print(
        f"Nearest neighbour fraction: {nearest_neighbour_fraction:.3f}, "
        f"Restricted parameters: True"
    )
    display(g.figure)
    save_dir = f"notebooks/plots/25-06-02_parameter_fit/{idt_id}/low_distance_simulation_parameters"
    os.makedirs(save_dir, exist_ok=True)
    g.figure.savefig(
        f"{save_dir}/{nearest_neighbour_fraction:.3f}_restricted_params_pairplot.pdf"
    )
    plt.close(g.figure)
