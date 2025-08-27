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

# %% [markdown]
# This is a notebook to print outputs from the grid search on fitness change distribution hyperparameters done in `notebooks/25-03-19_fcd_gridsearch`

# %%
import pandas as pd

data = (
    pd.read_csv("notebooks/25-03-19_fcd_gridsearch/results.csv")
    .rename(
        columns={
            "p": "fitness_change_probability",
            "s": "significance_threshold",
            "fcs": "fitness_change_scale",
        }
    )
    .assign(
        fitness_change_probability=lambda x: x.fitness_change_probability.round(4),
        significance_threshold=lambda x: x.significance_threshold.round(4),
    )
)
data

# %%
# Calculate theoretical yearly fitness change (and add to dataframe)
import numpy as np

division_rate = 33

protein_coding_fraction = 0.0174
data = data.assign(
    prob_sig_given_positive=lambda x: (
        ((1 - x.significance_threshold) / (1 + x.significance_threshold))
        ** (1 / x.fitness_change_scale)
    ),
    approx_prob_sig_given_positive=lambda x: np.exp(
        -2 * x.significance_threshold / x.fitness_change_scale
    ),
    theoretical_norm_mean_fitness_if_positive=lambda x: (
        (719 / 20_000)
        * protein_coding_fraction
        / (x.prob_sig_given_positive * x.fitness_change_probability)
    ),
    theoretical_yearly_fitness_change=lambda x: division_rate
    * np.where(
        x.theoretical_norm_mean_fitness_if_positive < 1,
        x.fitness_change_scale
        * x.fitness_change_probability
        * (2 * x.theoretical_norm_mean_fitness_if_positive - 1),
        np.nan,
    ),
    difference_from_theoretical=lambda x: x.normalisation_constant_slope
    - x.theoretical_yearly_fitness_change,
)
data.loc[data.normalisation_constant_slope.isna()]

# %%
sig_threshold_years = dict(zip(data.significance_threshold.unique(), range(40, 90, 10)))
sig_threshold_years

# %%
## heatmap of fitness change rate, fcp x st (separate plot for each fcs)
import matplotlib.pyplot as plt
import seaborn as sns

in_depth_sims = []
for unnorm_colname in [
    "normalisation_constant_slope",
    "theoretical_yearly_fitness_change",
    "difference_from_theoretical",
]:
    for fitness_change_scale in data.fitness_change_scale.unique():
        # heatmap, colour by normalisation_constant_slope, rows by fitness_change_probability, cols by significance_threshold
        this_fcs_data = data.loc[
            lambda x: x.fitness_change_scale == fitness_change_scale
        ]
        if this_fcs_data.normalisation_constant_slope.isna().all():
            print(
                f"{unnorm_colname} all NA for fitness_change_scale = {fitness_change_scale}"
            )
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        for normalised, ax in zip([False, True], axes):
            colname = ("normalised_" if normalised else "") + unnorm_colname
            sns.heatmap(
                data.assign(
                    **{
                        colname: lambda x: (
                            x[unnorm_colname]
                            / x.fitness_change_probability
                            / x.fitness_change_scale
                            if normalised
                            else x[unnorm_colname]
                        )
                    }
                )
                .loc[
                    lambda x: x.fitness_change_scale == fitness_change_scale,
                    ["fitness_change_probability", "significance_threshold", colname],
                ]
                .pivot(
                    index="fitness_change_probability",
                    columns="significance_threshold",
                    values=colname,
                ),
                # annot=True,
                # fmt=".2f",
                ax=ax,
                center=0,
                cmap="coolwarm",
                cbar_kws={
                    "label": "fitness change/year"
                    + (
                        "\n(normalised by fitness change probability)"
                        if normalised
                        else ""
                    )
                },
            )
            ax.set_title(("Un-n" if not normalised else "N") + "ormalised")

        if unnorm_colname == "normalisation_constant_slope":
            in_depth_sims.append((1.0, 0.0907, fitness_change_scale))
            # print(in_depth_sims[-1])
            for significance_threshold in data.significance_threshold.unique():
                filtered_data = data.loc[
                    lambda x: (x.fitness_change_scale == fitness_change_scale)
                    & (x.significance_threshold == significance_threshold)
                ]
                if filtered_data.loc[
                    lambda x: x.normalisation_constant_slope < 0
                ].empty:
                    continue
                min_fitness_change_probability = filtered_data.loc[
                    lambda x: x.normalisation_constant_slope < 0
                ].fitness_change_probability.min()
                in_depth_sims.append(
                    (
                        min_fitness_change_probability,
                        significance_threshold,
                        fitness_change_scale,
                    )
                )
                print(
                    in_depth_sims[-1],
                    "-> exp magnitude",
                    min_fitness_change_probability * fitness_change_scale,
                )
                # print(in_depth_sims[-1])
                # annotate a star on the heatmap for this point
                for ax in axes:
                    for st, fcp in (
                        (significance_threshold, min_fitness_change_probability),
                        (0.0907, 1.0),
                    ):
                        x_pos = [x.get_text() for x in axes[0].get_xticklabels()].index(
                            str(st)
                        ) + 0.5
                        y_pos = [x.get_text() for x in axes[0].get_yticklabels()].index(
                            str(fcp)
                        ) + 0.5
                        ax.plot(
                            x_pos,
                            y_pos,
                            marker=".",
                            color="black",
                            markersize=5,
                        )

        fig.suptitle(
            f"fitness_change_scale = {fitness_change_scale}"
            + (
                " (theoretical)"
                if unnorm_colname == "theoretical_yearly_fitness_change"
                else (
                    " (difference from theoretical)"
                    if unnorm_colname == "difference_from_theoretical"
                    else ""
                )
            )
        )
        fig.tight_layout(w_pad=2)
        display(fig)
        plt.close(fig)

print(
    "\n".join(
        [
            f"{fitness_change_probability}\t{significance_threshold}\t{fitness_change_scale}"
            f"\t(exp magnitude {fitness_change_probability*fitness_change_scale})"
            for (
                fitness_change_probability,
                significance_threshold,
                fitness_change_scale,
            ) in in_depth_sims
        ]
    )
)

# %%
in_depth_sims

# %%
# simple heatmap, with values of fitness_change_scale on x axis, fitness_change_probability on y axis and their product as the colour
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

fcp_values = [x for x in data.fitness_change_probability.unique() if 0.0735 <= x]
fcs_values = [x for x in data.fitness_change_scale.unique() if 0.03 <= x <= 0.1]

non_zero_mean = np.array([[fcp * fcs for fcs in fcs_values] for fcp in fcp_values])
fig, ax = plt.subplots(figsize=(4, 4))
sns.heatmap(
    non_zero_mean,
    ax=ax,
    norm=mcolors.LogNorm(),
    annot=True,
    cbar_kws={"label": "theoretical fitness change/year"},
)
ax.set_xticklabels(fcs_values)
ax.set_yticklabels(fcp_values)
ax.set_xlabel("fitness_change_scale")
ax.set_ylabel("fitness_change_probability")
plt.show()

# %%
## heatmap of fitness change rate, fcp x fcs (separate plot for each st)
import matplotlib.pyplot as plt
import seaborn as sns

fcs_values = [x for x in data.fitness_change_scale.unique() if 0.03 <= x <= 0.1]
fcp_values = [x for x in data.fitness_change_probability.unique()]

for significance_threshold in data.significance_threshold.unique():
    for unnorm_colname in [
        "normalisation_constant_slope",
        "theoretical_yearly_fitness_change",
        "difference_from_theoretical",
    ]:
        # heatmap, colour by normalisation_constant_slope, rows by fitness_change_probability, cols by significance_threshold
        this_fcs_data = data.loc[
            lambda x: (x.significance_threshold == significance_threshold)
            & (x.fitness_change_probability.isin(fcp_values))
            & (x.fitness_change_scale.isin(fcs_values))
        ]
        if this_fcs_data.normalisation_constant_slope.isna().all():
            print(
                f"{unnorm_colname} all NA for significance_threshold {significance_threshold}"
            )
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        for normalised, ax in zip([False, True], axes):
            colname = ("normalised_" if normalised else "") + unnorm_colname
            sns.heatmap(
                this_fcs_data.assign(
                    **{
                        colname: lambda x: (
                            x[unnorm_colname]
                            / x.fitness_change_probability
                            / x.fitness_change_scale
                            if normalised
                            else x[unnorm_colname]
                        )
                    }
                )
                .loc[:, ["fitness_change_probability", "fitness_change_scale", colname]]
                .pivot(
                    index="fitness_change_probability",
                    columns="fitness_change_scale",
                    values=colname,
                ),
                # annot=True,
                # fmt=".2f",
                ax=ax,
                center=0,
                cmap="coolwarm",
                cbar_kws={
                    "label": "fitness change/year"
                    + (
                        "(normalised by\nfitness change scale and probability)"
                        if normalised
                        else ""
                    )
                },
            )
            ax.set_title(("Un-n" if not normalised else "N") + "ormalised")

        fig.suptitle(
            f"significance threshold at {sig_threshold_years[significance_threshold]} years"
            + (
                " (theoretical)"
                if unnorm_colname == "theoretical_yearly_fitness_change"
                else (
                    " (difference from theoretical)"
                    if unnorm_colname == "difference_from_theoretical"
                    else ""
                )
            )
        )
        fig.tight_layout(w_pad=2)
        display(fig)
        save_dir = (
            f"notebooks/25-03-19_fcd_gridsearch/heatmaps/st_{significance_threshold}"
        )
        plt.close(fig)

# %%
## same heatmaps, but with norm and unnorm on separate plots, with column by significance_threshold

filtered_data = data.loc[
    lambda x: (x.fitness_change_probability.isin(fcp_values))
    & (x.fitness_change_scale.isin(fcs_values))
]

bounds = {
    normalised: {
        unnorm_colname: {
            "min": (
                filtered_data[unnorm_colname]
                if not normalised
                else filtered_data[unnorm_colname]
                / filtered_data.fitness_change_probability
                / filtered_data.fitness_change_scale
            ).min(),
            "max": (
                filtered_data[unnorm_colname]
                if not normalised
                else filtered_data[unnorm_colname]
                / filtered_data.fitness_change_probability
                / filtered_data.fitness_change_scale
            ).max(),
        }
        for unnorm_colname in [
            "normalisation_constant_slope",
            "theoretical_yearly_fitness_change",
            "difference_from_theoretical",
        ]
    }
    for normalised in [False, True]
}

for normalised in [False, True]:
    for unnorm_colname in [
        "normalisation_constant_slope",
        # "theoretical_yearly_fitness_change",
        # "difference_from_theoretical",
    ]:
        fig, axes = plt.subplots(
            1,
            len(data.significance_threshold.unique()),
            figsize=(2 * len(data.significance_threshold.unique()), 4),
            sharex=True,
            sharey=True,
            width_ratios=[1] * (len(data.significance_threshold.unique()) - 1) + [1.2],
        )
        for st_ix, (ax, significance_threshold) in enumerate(
            zip(axes, data.significance_threshold.unique())
        ):
            # heatmap, colour by normalisation_constant_slope, rows by fitness_change_probability, cols by fitness_change_scale
            this_fcs_data = data.loc[
                lambda x: (x.significance_threshold == significance_threshold)
                & (x.fitness_change_probability.isin(fcp_values))
                & (x.fitness_change_scale.isin(fcs_values))
            ]
            colname = ("normalised_" if normalised else "") + unnorm_colname
            sns.heatmap(
                this_fcs_data.assign(
                    **{
                        colname: lambda x: (
                            x[unnorm_colname]
                            / x.fitness_change_probability
                            / x.fitness_change_scale
                            if normalised
                            else x[unnorm_colname]
                        )
                    }
                )
                .loc[:, ["fitness_change_probability", "fitness_change_scale", colname]]
                .pivot(
                    index="fitness_change_probability",
                    columns="fitness_change_scale",
                    values=colname,
                ),
                ax=ax,
                center=0,
                cmap="coolwarm",
                cbar=st_ix == len(data.significance_threshold.unique()) - 1,
                cbar_kws={
                    "label": "fitness change/year"
                    + (
                        " (normalised by\nfitness change scale and probability)"
                        if normalised
                        else ""
                    )
                },
                vmin=bounds[normalised][unnorm_colname]["min"],
                vmax=bounds[normalised][unnorm_colname]["max"],
            )
            ax.set_title(
                f"{sig_threshold_years[significance_threshold]} year threshold"
            )
            ax.set_ylabel("Fitness change probability" if st_ix == 0 else "")
            ax.set_xlabel("Fitness change scale" if st_ix == len(axes) // 2 else "")
        fig.suptitle("Fitness change rate by significance threshold")
        fig.tight_layout(w_pad=2)
        print(f"normalised={normalised}, unnorm_colname={unnorm_colname}")
        display(fig)
        save_dir = f"notebooks/plots/25-03-19_fcd_gridsearch/heatmaps/{unnorm_colname}"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            os.path.join(
                save_dir,
                f"{'normalised' if normalised else 'unnormalised'}.png",
            ),
        )
        plt.close(fig)

# %% [markdown]
# These simulations have been run in `notebooks/25-03-19_fcd_gridsearch/all_statuses_simulation.jl` (as changing significance threshold requires a bit of hackery). Now to plot the outputs! The aim here is to see if the problem has been solved: are we getting
# a) quiescent division after smoking cessation
# b) mutations net deleterious (as measured by normalisation constant slope)
# c) maybe smoking_driver should push mutations to be net helpful?
#
# Independent variables: fcs, fcp, st, patient, paradigm, replicate, (year, cell)
# Dependent variables: mutational burden, mean_fitness
#
# I think I can just import all the data as a single df/few dfs
#
# compartment-wise, year-wise
# fcs,fcp,st,patient,paradigm,replicate_index,year,compartment,mean_mutational_burden,mean_fitness,empirical_division_rate,empirical_differentiation_rate,empirical_immune_death_rate,driver_fraction
#
# year-wise
# fcs,fcp,st,patient,paradigm,replicate_index,year,normalisation_constant
#
# status mb gradients
# fcs,fcp,st,patient,paradigm,replicate_index,compartment,mutational_burden_gradient,
#
# Best plots to interrogate this:
# * Normalisation constant over time, column as pt

# %%
in_depth_sims = [
    (0.1, 0.11, 0.05),
    # (0.87, 0.11, 0.05),
    # (0.696, 0.11, 0.05),
]

# %%
for (
    fitness_change_probability,
    significance_threshold_,
    fitness_change_scale,
) in in_depth_sims:
    for spatial in [False, True]:
        significance_threshold = significance_threshold_ if spatial else 0.03
        prob_sig_given_positive = (
            (1 - significance_threshold) / (1 + significance_threshold)
        ) ** (1 / fitness_change_scale)
        driver_prob_estimate = 2.36e-4
        prob_positive_given_non_zero = driver_prob_estimate / (
            fitness_change_probability * prob_sig_given_positive
        )
        print(
            f"fcp = {fitness_change_probability}, st = {significance_threshold_}, fcs = {fitness_change_scale}; "
            f"{' spatial' if spatial else 'non-spatial'}"
            f" -> prob_sig_given_positive = {prob_sig_given_positive}, prob_positive_given_non_zero = {prob_positive_given_non_zero*100:.2f}%"
        )

# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ClonesModelling.parameters.hypothetical_paradigm_class import MODULE_ORDERING

qgcc = 5


@dataclass
class InDepthSim:
    fitness_change_probability: float
    significance_threshold: float
    fitness_change_scale: float

    @property
    def paradigms(self):
        return os.listdir(
            f"notebooks/data/25-03-19_fcd_gridsearch/qgcc_{qgcc}/fcs_{self.fitness_change_scale}/"
            f"fcp_{self.fitness_change_probability}/st_{self.significance_threshold}"
        )

    @property
    def n_replicates(self):
        return len(
            os.listdir(
                f"notebooks/data/25-03-19_fcd_gridsearch/qgcc_{qgcc}/"
                f"fcs_{self.fitness_change_scale}/"
                f"fcp_{self.fitness_change_probability}/"
                f"st_{self.significance_threshold}/{self.paradigms[0]}"
            )
        )


patients = ["test_never_smoker", "test_smoker", "test_ex_smoker"]


@dataclass
class PtSimData:
    fitness_change_probability: float
    significance_threshold: float
    fitness_change_scale: float
    paradigm: str
    replicate_index: int
    patient: str

    @property
    def replicate_dir(self):
        return (
            f"notebooks/data/25-03-19_fcd_gridsearch/qgcc_{qgcc}/fcs_{self.fitness_change_scale}/"
            f"fcp_{self.fitness_change_probability}/st_{self.significance_threshold}/"
            f"{self.paradigm}/replicate_{self.replicate_index}"
        )

    @property
    def csv_log_file(self):
        return f"{self.replicate_dir}/{self.patient}.csv"

    @property
    def fitness_summaries_file(self):
        return f"{self.replicate_dir}/cell_records/fitness_summaries/{self.patient}.csv"

    @property
    def mutational_burden_file(self):
        return f"{self.replicate_dir}/cell_records/mutational_burden/{self.patient}.csv"


in_depth_sims_df = pd.concat(
    [
        pd.read_csv(pt_sim_dta.csv_log_file)
        .assign(
            division_rate=lambda x: x.new_cell_count / x.cell_count,
            differentiation_rate=lambda x: x.differentiated_cell_count / x.cell_count,
            immune_death_rate=lambda x: x.immune_death_count / x.cell_count,
            fitness_change_probability=pt_sim_dta.fitness_change_probability,
            significance_threshold=pt_sim_dta.significance_threshold,
            fitness_change_scale=pt_sim_dta.fitness_change_scale,
            # paradigm=pt_sim_dta.paradigm,
            paradigm="-".join(
                sorted(
                    pt_sim_dta.paradigm.split("-"),
                    key=lambda x: MODULE_ORDERING.index(x),
                )
            ),
            replicate_index=pt_sim_dta.replicate_index,
            patient=pt_sim_dta.patient,
            on_fitness_change_positivity_boundary=lambda x: ~(
                (x.significance_threshold == 0.09)
                & (x.fitness_change_probability == 1.0)
            ),
        )
        .loc[
            :,
            [
                "year",
                "compartment",
                "division_rate",
                "differentiation_rate",
                "immune_death_rate",
                "fitness_change_probability",
                "significance_threshold",
                "fitness_change_scale",
                "paradigm",
                "replicate_index",
                "patient",
                "on_fitness_change_positivity_boundary",
            ],
        ]
        .merge(
            pd.read_csv(pt_sim_dta.fitness_summaries_file)
            .rename(columns={"step_number": "year"})
            .assign(
                relevant_fitness=lambda x: np.where(
                    ((pt_sim_dta.patient == "test_smoker") & (x.year >= 20))
                    | (
                        (pt_sim_dta.patient == "test_ex_smoker")
                        & (x.year >= 20)
                        & (x.year <= 60)
                    ),
                    x.sm_mean_fitness,
                    x.ns_mean_fitness,
                )
            )
            .loc[
                :,
                [
                    "year",
                    "compartment",
                    "sm_mean_fitness",
                    "ns_mean_fitness",
                    "relevant_fitness",
                    "normalisation_constant",
                ],
            ],
            on=["year", "compartment"],
        )
        .merge(
            pd.read_csv(pt_sim_dta.mutational_burden_file)
            .rename(columns={"record_number": "year"})
            .assign(
                total_mutations=lambda x: x.driver_non_smoking_signature_mutations
                + x.driver_smoking_signature_mutations
                + x.passenger_smoking_signature_mutations
                + x.passenger_non_smoking_signature_mutations,
                driver_fraction=lambda x: (
                    (
                        x.driver_non_smoking_signature_mutations
                        + x.driver_smoking_signature_mutations
                    )
                    / x.total_mutations
                ),
                smoking_signature_fraction=lambda x: (
                    x.driver_smoking_signature_mutations
                    + x.passenger_smoking_signature_mutations
                )
                / x.total_mutations,
            )
            .loc[
                :,
                [
                    "year",
                    "compartment",
                    "total_mutations",
                    "driver_fraction",
                    "smoking_signature_fraction",
                    "divisions",
                ],
            ]
            .groupby(["year", "compartment"])
            .mean()
            .reset_index(),
            on=["year", "compartment"],
        )
        # then add in mean_mutational_burden, driver_fraction from mutational burden df
        for in_depth_sim in [
            InDepthSim(*in_depth_sim)
            for in_depth_sim in in_depth_sims
            if os.path.isdir(
                f"notebooks/data/25-03-19_fcd_gridsearch/qgcc_{qgcc}/fcs_{in_depth_sim[2]}/"
                f"fcp_{in_depth_sim[0]}/st_{in_depth_sim[1]}"
            )
        ]
        for pt_sim_dta in [
            PtSimData(
                in_depth_sim.fitness_change_probability,
                in_depth_sim.significance_threshold,
                in_depth_sim.fitness_change_scale,
                paradigm,
                replicate_index,
                patient,
            )
            for paradigm in in_depth_sim.paradigms
            for replicate_index in range(in_depth_sim.n_replicates)
            for patient in patients
        ]
    ]
)
in_depth_sims_df

# %%
## Plotting setup
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors


titles = {
    "test_never_smoker": "80-year-old\nNever smoker",
    "test_ex_smoker": "80-year-old\nSmoked 20-60",
    "test_smoker": "80-year-old\nSmoked 20-80",
}
plots_dir = "notebooks/plots/25-03-19_fcd_gridsearch"
os.makedirs(plots_dir, exist_ok=True)


def add_smoking_history(ax, patient):
    if patient == "test_ex_smoker":
        ax.axvspan(20, 60, color="grey", alpha=0.2)
    elif patient == "test_smoker":
        ax.axvspan(20, 80, color="grey", alpha=0.2)
    return ax


# %%
## x=year, y=division_rate, hue=fitness_change_scale, column=patient
# separate plot for paradigm in ['q','q-ir','q-sd']
# subset to compartment=='quiescent'
div_rate_year_bin_size = 10
fitness_year_bin_size = 1

fcs_values_to_plot = [
    x for x in in_depth_sims_df.fitness_change_scale.unique() if 0.025 < x < 10.0
]


def get_relevant_binned_data(
    data, year_bin_size, colour_by, on_boundary, paradigm, patient
):
    return (
        data.loc[
            lambda x: (x.paradigm == paradigm)
            & (x.compartment == "quiescent")
            & (x.patient == patient)
            & (x.on_fitness_change_positivity_boundary == on_boundary)
            & (x.fitness_change_scale.isin(fcs_values_to_plot))
        ]
        .assign(
            year=lambda x: year_bin_size * (x.year // year_bin_size),
            normalised_relative_fitness=lambda x: x.relevant_fitness
            / x.fitness_change_scale
            / x.fitness_change_probability,
            projected_relative_fitness=lambda x: (
                (1 - np.exp(-x.relevant_fitness)) / (1 + np.exp(-x.relevant_fitness))
            ),
        )
        .groupby(["year", colour_by])[
            [
                "division_rate",
                "relevant_fitness",
                "normalised_relative_fitness",
                "projected_relative_fitness",
            ]
        ]
        .mean()
        .reset_index()
    )


for colour_by in [
    "fitness_change_scale",
    "significance_threshold",
    "fitness_change_probability",
]:
    for on_boundary in [True]:#, False] if colour_by == "fitness_change_scale" else [True]:
        for paradigm in ["q", "q-ir", "q-sd"]:
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(6 + (colour_by != "fitness_change_scale"), 4),
                sharex=True,
                sharey="row",
            )
            for col_ix, patient in enumerate(patients):
                div_rate_ax = axes[0, col_ix]
                div_rate_data = get_relevant_binned_data(
                    in_depth_sims_df,
                    div_rate_year_bin_size,
                    colour_by,
                    on_boundary,
                    paradigm,
                    patient,
                )
                sns.lineplot(
                    data=div_rate_data,
                    x="year",
                    y="division_rate",
                    hue=colour_by,
                    palette=sns.color_palette(
                        "viridis",
                        n_colors=len(div_rate_data[colour_by].unique()),
                    ),
                    ax=add_smoking_history(div_rate_ax, patient),
                    legend=col_ix == 2,
                )
                div_rate_ax.set_title(titles[patient])
                div_rate_ax.set_ylabel("Division rate")
                div_rate_ax.set_xlabel("Year")
                if col_ix == 2:
                    div_rate_ax.legend(
                        title=colour_by.replace(
                            "_", "\n" if colour_by == "fitness_change_scale" else " "
                        ).title(),
                        loc="center left",
                        bbox_to_anchor=(1, 0.5),
                        ncol=1 + (colour_by != "fitness_change_scale"),
                    )

                ax = axes[1, col_ix]
                fitness_data = get_relevant_binned_data(
                    in_depth_sims_df,
                    fitness_year_bin_size,
                    colour_by,
                    on_boundary,
                    paradigm,
                    patient,
                )
                sns.lineplot(
                    data=fitness_data,
                    x="year",
                    # y="normalised_relative_fitness",
                    y="projected_relative_fitness",
                    hue=colour_by,
                    palette=sns.color_palette(
                        "viridis",
                        n_colors=len(fitness_data[colour_by].unique()),
                    ),
                    ax=add_smoking_history(ax, patient),
                    legend=False,
                )
                ax.set_ylabel(
                    # "Quiescent Mean Fitness\n(normalised by scale\nand probability)"
                    "Quiescent mean\nprojected fitness"
                )
                ax.set_xlabel("Year")
            fig.suptitle(f"Paradigm: {paradigm}; on_boundary={on_boundary}")
            fig.tight_layout()
            save_dir = f"{plots_dir}/quiescent_division_rate/{'on_boundary' if on_boundary else 'min_fc'}"
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(f"{save_dir}/{paradigm}.png")
            display(fig)
            plt.close(fig)

# %% [markdown]
# This lets us narrow down to a few good candidates, with low enough fcs and fcp that there's not too much saturation, while also keeping mutations deleterious and keeping within the constraints of 'reasonableness' of the fcp and st parameters:
#

# %%
# print out unique values of fitness_change_scale, fitness_change_probability, significance_threshold (as triplets)
print(
    "\n".join(
        [
            f"fcs {fitness_change_scale}\tfcp {fitness_change_probability}\tst {significance_threshold}"
            for (
                fitness_change_scale,
                fitness_change_probability,
                significance_threshold,
            ) in in_depth_sims_df.loc[
                lambda x: x.on_fitness_change_positivity_boundary
            ][
                ["fitness_change_scale", "fitness_change_probability", "significance_threshold"]
            ]
            .drop_duplicates()
            .itertuples(index=False)
        ]
    )
)

# %%
abbrev_paradigm_colours = {
    par: f"C{ix}"
    for ix, par in enumerate(
        ["base", "q", "p", "ir", "sd", "q-ir", "q-sd", "p-ir", "p-sd"]
    )
}
module_full_names = {
    "base": "base",
    "q": "quiescent",
    "p": "protected",
    "ir": "immune_response",
    "sd": "smoking_driver",
}
paradigm_full_names = {
    "-".join(
        sorted(
            par.split("-"),
            key=lambda x: MODULE_ORDERING.index(x),
        )
    ): "-".join(
        sorted(
            [module_full_names[x] for x in par.split("-")],
            key=lambda x: MODULE_ORDERING.index(x),
        )
    )
    for par in abbrev_paradigm_colours.keys()
}
paradigm_colours = {
    paradigm_full_names[par]: colour
    for par, colour in abbrev_paradigm_colours.items()
}
paradigm_full_names


# %%
def plot_by_paradigm(
    paradigms: list[str],
    fcp: float,
    st: float,
    fcs: float,
    patients: list[str],
    y_col: str,
    row_two_y_col: str | None = None,
    year_bin_size: int = 1,
    row_two_year_bin_size: int | None = None,
    compartment_restriction: str | None = None,
) -> plt.Figure:
    n_rows = 2 if row_two_y_col else 1
    percentage_colnames = ["driver_fraction", "smoking_signature_fraction"]
    zero_line_colnames = ["projected_relative_fitness"]
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(7.5, 4 if row_two_y_col else 3), sharex=True, sharey="row"
    )

    for col_ix, patient in enumerate(patients):
        data = in_depth_sims_df.loc[
            lambda x: (x.paradigm.isin(paradigms))
            & (
                (x.compartment == compartment_restriction)
                if compartment_restriction is not None
                else True
            )
            & (x.patient == patient)
            & (x.fitness_change_scale == fcs)
            & (x.fitness_change_probability == fcp)
            & (x.significance_threshold == st)
        ].assign(
            paradigm=lambda x: x.paradigm.replace(paradigm_full_names),
            projected_relative_fitness=lambda x: (
                (1 - np.exp(-x.relevant_fitness)) / (1 + np.exp(-x.relevant_fitness))
            ),
        )

        # First row plot
        ax = axes[0, col_ix] if row_two_y_col else axes[col_ix]
        sns.lineplot(
            data=data.assign(year=lambda x: year_bin_size * (x.year // year_bin_size))
            .groupby(
                ["year", "paradigm", "replicate_index"]
                + (["compartment"] if compartment_restriction is None else [])
            )[[y_col]]
            .mean()
            .reset_index(),
            x="year",
            y=y_col,
            hue="paradigm",
            palette=paradigm_colours,
            hue_order=map(lambda x: paradigm_full_names[x], paradigms),
            style=(
                "compartment"
                if compartment_restriction is None
                and any(
                    "p" in parad.split("-") or "q" in parad.split("-")
                    for parad in paradigms
                )
                else None
            ),
            ax=add_smoking_history(ax, patient),
            legend=col_ix == 2,
        )
        ax.set_title(titles[patient])
        ax.set_ylabel(f"{y_col.replace('_', ' ').title()}")
        ax.set_xlabel("Year")
        if y_col in zero_line_colnames:
            ax.axhline(0, color="black", linestyle="--")
        if col_ix == 2:
            ax.legend(title="Paradigm", loc="center left", bbox_to_anchor=(1, 0.5))
        if y_col in percentage_colnames:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
            )

        # Second row plot (if applicable)
        if row_two_y_col:
            ax = axes[1, col_ix]
            row_two_year_bin_size = (
                row_two_year_bin_size
                if row_two_year_bin_size is not None
                else year_bin_size
            )
            sns.lineplot(
                data=data.assign(
                    year=lambda x: row_two_year_bin_size
                    * (x.year // row_two_year_bin_size)
                )
                .groupby(["year", "paradigm", "replicate_index"])[[row_two_y_col]]
                .mean()
                .reset_index(),
                x="year",
                y=row_two_y_col,
                hue="paradigm",
                palette=paradigm_colours,
                hue_order=map(lambda x: paradigm_full_names[x], paradigms),
                ax=add_smoking_history(ax, patient),
                legend=False,
            )
            ax.set_ylabel(f"{row_two_y_col.replace('_', ' ').title()}")
            ax.set_xlabel("Year")
            if row_two_y_col in zero_line_colnames:
                ax.axhline(0, color="black", linestyle="--")
            if row_two_y_col in percentage_colnames:
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
                )

    fig.suptitle(f"fcs={fcs}, fcp={fcp}, st={st}")
    fig.tight_layout()
    return fig


# %%
for fcp, st, fcs in in_depth_sims:
    fig=plot_by_paradigm(
        paradigms=["q", "q-ir", "q-sd"],
        fcp=fcp,
        st=st,
        fcs=fcs,
        patients=patients,
        y_col="projected_relative_fitness",
        row_two_y_col="division_rate",
        row_two_year_bin_size=5,
        compartment_restriction="quiescent",
    )
    save_dir = f"{plots_dir}/hypothesis_demonstrations/fcp_{fcp}_st_{st}_fcs_{fcs}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/quiescent.png"
    )
    display(fig)
    plt.close(fig)

# %%
for fcp, st, fcs in in_depth_sims:
    fig = plot_by_paradigm(
        paradigms=["ir", "q-ir", "p-ir"],
        fcp=fcp,
        st=st,
        fcs=fcs,
        patients=patients,
        y_col="immune_death_rate",
        year_bin_size=5,
        compartment_restriction="main",
    )
    save_dir = f"{plots_dir}/hypothesis_demonstrations/fcp_{fcp}_st_{st}_fcs_{fcs}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/immune_response.png")
    display(fig)
    plt.close(fig)

# %%
for fcp, st, fcs in in_depth_sims:
    fig = plot_by_paradigm(
        paradigms=["base", "sd"],  # , "q-sd"],
        fcp=fcp,
        st=st,
        fcs=fcs,
        patients=patients,
        y_col="driver_fraction",
        year_bin_size=5,
    )
    save_dir = f"{plots_dir}/hypothesis_demonstrations/fcp_{fcp}_st_{st}_fcs_{fcs}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(f"{save_dir}/smoking_driver.png")
    display(fig)
    plt.close(fig)

# %%
in_depth_sims_df

# %%
# at in_depth_sims[0], print out the mean quiescent division rate over lifetime
in_depth_sims_df.loc[
    lambda x: (x.fitness_change_probability == in_depth_sims[0][0])
    & (x.significance_threshold == in_depth_sims[0][1])
    & (x.fitness_change_scale == in_depth_sims[0][2])
    & (x.compartment == "quiescent")
    & (x.paradigm == "q")
].groupby(["patient", "replicate_index"]).division_rate.mean().groupby("patient").agg(
    mean="mean",
    median="median",
    p2_5=lambda x: x.quantile(0.025),
    p97_5=lambda x: x.quantile(0.975),
).reset_index()

# %%
# at in_depth_sims[0], print out the mean immune death rate at 80
in_depth_sims_df.loc[
    lambda x: (x.year == 80)
    & (x.paradigm == "ir")
    & (x.fitness_change_probability == in_depth_sims[0][0])
    & (x.significance_threshold == in_depth_sims[0][1])
    & (x.fitness_change_scale == in_depth_sims[0][2])
].groupby("patient")["immune_death_rate"].mean()

# %%
# at in_depth_sims[0], print out the mean mutationl burden in the protected compartment vs main compartment (protected paradigm)
print(
    in_depth_sims_df.loc[
        lambda x: (x.fitness_change_probability == in_depth_sims[0][0])
        & (x.significance_threshold == in_depth_sims[0][1])
        & (x.fitness_change_scale == in_depth_sims[0][2])
        & (x.patient == "test_smoker")
        & (x.year.isin([20, 80]))
        & (x.paradigm.isin(["base", "p"]))
    ]
    .groupby(["paradigm", "compartment", "year"])["total_mutations"]
    .mean()
)

# %%
1-(4160.861357-478.144543)/(6272.739583-469.901042), 1-(5820.192558-473.899574)/(6272.739583-469.901042)

# %%
for fcp, st, fcs in in_depth_sims:
    fig = plot_by_paradigm(
        paradigms=["base", "p"],
        fcp=fcp,
        st=st,
        fcs=fcs,
        patients=patients,
        y_col="total_mutations",
        year_bin_size=1,
    )
    save_dir = f"{plots_dir}/hypothesis_demonstrations/fcp_{fcp}_st_{st}_fcs_{fcs}"
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(
        f"{save_dir}/protected.png"
    )
    display(fig)
    plt.close(fig)

# %%
in_depth_sims_df

# %%
paradigms = ["q-ir"]
year_bin_size = 2
fcp, st, fcs = 0.1, 0.11, 0.05

percentage_colnames = ["driver_fraction", "smoking_signature_fraction"]
zero_line_colnames = ["projected_relative_fitness"]
fig, axes = plt.subplots(1, 3, figsize=(8, 2.3), sharex=True, sharey="row")

styles = [
    "-",
    "--",
    ":",
]

for i, (y_col, compartment_restriction) in enumerate(
    [
        ("division_rate", "main"),
        ("immune_death_rate", "main"),
        ("division_rate", "quiescent"),
    ]
):
    for col_ix, patient in enumerate(patients):
        data = in_depth_sims_df.loc[
            lambda x: (x.paradigm.isin(paradigms))
            & (
                (x.compartment == compartment_restriction)
                if compartment_restriction is not None
                else True
            )
            & (x.patient == patient)
            & (x.fitness_change_scale == fcs)
            & (x.fitness_change_probability == fcp)
            & (x.significance_threshold == st)
        ].assign(
            paradigm=lambda x: x.paradigm.replace(paradigm_full_names),
            projected_relative_fitness=lambda x: (
                (1 - np.exp(-x.relevant_fitness)) / (1 + np.exp(-x.relevant_fitness))
            ),
        )

        # First row plot
        ax = axes[col_ix]
        sns.lineplot(
            data=data.assign(year=lambda x: year_bin_size * (x.year // year_bin_size))
            .groupby(
                ["year", "paradigm", "replicate_index"]
                + (["compartment"] if compartment_restriction is None else [])
            )[[y_col]]
            .mean()
            .reset_index(),
            x="year",
            y=y_col,
            color="black",
            linestyle=styles[i],
            label=f"{y_col.replace('_', ' ').title()} ({compartment_restriction})",
            # hue="paradigm",
            # palette=paradigm_colours,
            # hue_order=map(lambda x: paradigm_full_names[x], paradigms),
            # style=(
            #     "compartment"
            #     if compartment_restriction is None
            #     and any(
            #         "p" in parad.split("-") or "q" in parad.split("-")
            #         for parad in paradigms
            #     )
            #     else None
            # ),
            ax=add_smoking_history(ax, patient),
            legend=col_ix == 2,
        )
        ax.set_title(titles[patient])
        ax.set_ylabel(f"{y_col.replace('_', ' ').title()}")
        ax.set_xlabel("Year")
        if y_col in zero_line_colnames:
            ax.axhline(0, color="black", linestyle="--")
        if col_ix == 2:
            ax.legend(title="Paradigm", loc="center left", bbox_to_anchor=(1, 0.5))
        if y_col in percentage_colnames:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y*100:.1f}%")
            )
axes[0].set_ylabel("Events per year")

# fig.suptitle(f"fcs={fcs}, fcp={fcp}, st={st}")
fig.tight_layout()
