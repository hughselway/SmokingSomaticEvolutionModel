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
from ClonesModelling.data.smoking_records import get_smoking_records
from ClonesModelling.data.smoking_record_class import (
    SmokerRecord,
    ExSmokerRecord,
    NonSmokerRecord,
)

smoking_records: dict[str, SmokerRecord | ExSmokerRecord | NonSmokerRecord] = {
    # sr.patient: sr for sr in get_smoking_records()
    "test_smoker": SmokerRecord(
        patient="test_smoker", age=80, start_smoking_age=20, pack_years=60
    ),
    "test_ex_smoker": ExSmokerRecord(
        patient="test_ex_smoker",
        age=80,
        start_smoking_age=20,
        stop_smoking_age=60,
        pack_years=40,
    ),
    "test_never_smoker": NonSmokerRecord(patient="test_never_smoker", age=80),
}
smoking_status_colours = {
    # "non-smoker": "green",
    # "ex-smoker": "orange",
    # "smoker": "red",
    # "non-smoker": "#648FFF",
    # "ex-smoker": "#FFB000",
    # "smoker": "#DC267F",
    "non-smoker": "#5d4b98",
    "ex-smoker": "#80bb51",
    "smoker": "#ab5c9f",
    "child": "#cbaf8a",
}

# %%
import dataclasses
import json

# import numpy as np
import pandas as pd

from ClonesModelling.simulation.run_fast_simulation import (
    PatientSimulationOutput,
    SpatialPatientSimulationOutput,
)
from ClonesModelling.simulation.output_class import read_simulation_output_from_json

output_directory = "notebooks/24-10-09-default-simulation_status_rep_pts"


@dataclasses.dataclass
class Simulation:
    paradigm_name: str
    spatial: bool

    @property
    def logs_directory(self) -> str:
        return (
            f"{output_directory}/"
            f"{'spatial' if self.spatial else 'non-spatial'}/logs/{self.paradigm_name}"
        )

    @property
    def simulation_output_json(self) -> str:
        return (
            f"{output_directory}/{'spatial' if self.spatial else 'non-spatial'}"
            f"/simulation_outputs/{self.paradigm_name}.json"
        )

    @property
    def n_replicates(self) -> int:
        return len(json.load(open(self.simulation_output_json)))
        # return len(
        #     # [x for x in os.listdir(self.logs_directory) if x.startswith("replicate")]

        # )

    @property
    def patients(self) -> list[str]:
        return list(json.load(open(self.simulation_output_json))[0].keys())
        # return [
        #     filename.split(".")[0]
        #     for filename in os.listdir(
        #         f"{self.logs_directory}/replicate_0/cell_records/fitness_summaries"
        #     )
        # ]

    def get_mutational_burden(self, replicate_index: int, patient: str) -> pd.DataFrame:
        return pd.read_csv(
            f"{self.logs_directory}/replicate_{replicate_index}/cell_records/mutational_burden/{patient}.csv"
        ).assign(
            total_mutations=lambda df: (
                df["driver_non_smoking_signature_mutations"]
                + df["driver_smoking_signature_mutations"]
                + df["passenger_smoking_signature_mutations"]
                + df["passenger_non_smoking_signature_mutations"]
            )
        )

    def get_simulation_output(
        self,
    ) -> (
        list[dict[str, PatientSimulationOutput]]
        | list[dict[str, SpatialPatientSimulationOutput]]
    ):
        simulation_outputs = read_simulation_output_from_json(
            self.simulation_output_json, self.spatial
        )
        assert len(simulation_outputs) == self.n_replicates, (
            len(simulation_outputs),
            self.n_replicates,
        )
        assert set(simulation_outputs[0].keys()) == set(self.patients), (
            set(simulation_outputs[0].keys()),
            set(self.patients),
        )
        return simulation_outputs

    # def get_final_timepoint_mutational_burden(
    #     self, replicate_index: int, patient: str
    # ) -> list[int]:
    #     # all_timepoints_mb = self.get_mutational_burden(replicate_index, patient)
    #     # time_colname = "step_number" if not self.spatial else "record_number"
    #     # return all_timepoints_mb[
    #     #     all_timepoints_mb[time_colname] == all_timepoints_mb[time_colname].max()
    #     # ]
    #     return self.get_simulation_output()[replicate_index][patient].mutational_burden


def read_simulation(paradigm_name: str, spatial: bool) -> Simulation:
    return Simulation(paradigm_name, spatial)


# %%
read_simulation("q-ir", False).logs_directory


# %%
def get_mb_gradient(relevant_mb: pd.Series, records_per_year: float) -> float:
    mutations_per_record = relevant_mb.diff().mean()
    return mutations_per_record * records_per_year


def get_mutational_burden_gradients(
    simulation: Simulation, replicate_index: int, patient: str
) -> tuple[float | None, float | None, float | None]:
    mb = simulation.get_mutational_burden(replicate_index, patient)
    if len(mb) == 0:
        assert patient == "PD37455", f"{simulation} {replicate_index} {patient}"
        return None, None, None
    time_colname = "step_number" if not simulation.spatial else "record_number"
    smoking_record = smoking_records[patient]
    mean_mb = (
        mb.groupby(time_colname)["total_mutations"]
        .mean()
        .reset_index()
        .assign(
            record_number=lambda df: (df[time_colname] / df[time_colname][0]).astype(
                int
            )
        )[["record_number", "total_mutations"]]
    )
    records_per_year: float = (
        mean_mb["record_number"].max() / smoking_records[patient].age
    )
    # for each of 1 - smoking_record.start_smoking_age, smoking_record.start_smoking_age - smoking_record.stop_smoking_age, smoking_record.stop_smoking_age - smoking_record.age
    # calculate the gradient of the mutational burden with respect to time in years in that range
    if smoking_record.status == "non-smoker":
        assert isinstance(smoking_record, NonSmokerRecord)
        return (get_mb_gradient(mean_mb.total_mutations, records_per_year), None, None)
    elif smoking_record.status == "ex-smoker":
        assert isinstance(smoking_record, ExSmokerRecord)
        return (
            get_mb_gradient(
                mean_mb.loc[
                    mean_mb["record_number"] / records_per_year
                    <= smoking_record.start_smoking_age,
                    "total_mutations",
                ],
                records_per_year,
            ),
            get_mb_gradient(
                mean_mb.loc[
                    (
                        mean_mb["record_number"] / records_per_year
                        > smoking_record.start_smoking_age
                    )
                    & (
                        mean_mb["record_number"] / records_per_year
                        <= smoking_record.stop_smoking_age
                    ),
                    "total_mutations",
                ],
                records_per_year,
            ),
            get_mb_gradient(
                mean_mb.loc[
                    mean_mb["record_number"] / records_per_year
                    > smoking_record.stop_smoking_age,
                    "total_mutations",
                ],
                records_per_year,
            ),
        )
    assert isinstance(smoking_record, SmokerRecord)
    return (
        get_mb_gradient(
            mean_mb.loc[
                mean_mb["record_number"] / records_per_year
                <= smoking_record.start_smoking_age,
                "total_mutations",
            ],
            records_per_year,
        ),
        get_mb_gradient(
            mean_mb.loc[
                mean_mb["record_number"] / records_per_year
                > smoking_record.start_smoking_age,
                "total_mutations",
            ],
            records_per_year,
        ),
        None,
    )


# %%
# make mb_gradients with columns paradigm,spatial,patient,smoking_status,pre_smoking_muts_per_year,smoking_muts_per_year,post_smoking_muts_per_year
if os.path.exists(f"{output_directory}/mb_gradients.csv"):
    print(f"Reading existing mb_gradients from {output_directory}/mb_gradients.csv")
    mb_gradients = pd.read_csv(f"{output_directory}/mb_gradients.csv")
else:
    data = []
    for paradigm_name in sorted(
        os.listdir(f"{output_directory}/spatial/logs"),
        # os.listdir(f"{output_directory}/non-spatial/logs"),
        key=lambda x: (len(x.split("-")), x),
    ):
        print(paradigm_name, end=" ", flush=True)
        for spatial in [False, True]:
            simulation = read_simulation(paradigm_name, spatial)
            for replicate_index in range(simulation.n_replicates):
                for patient in simulation.patients:
                    smoking_record = smoking_records[patient]
                    (
                        pre_smoking_muts_per_year,
                        smoking_muts_per_year,
                        post_smoking_muts_per_year,
                    ) = get_mutational_burden_gradients(
                        simulation, replicate_index, patient
                    )
                    if pre_smoking_muts_per_year is None:
                        continue
                    data.append(
                        {
                            "paradigm": paradigm_name,
                            "spatial": spatial,
                            "patient": patient,
                            "smoking_status": smoking_record.status,
                            "pre_smoking_muts_per_year": pre_smoking_muts_per_year,
                            "smoking_muts_per_year": smoking_muts_per_year,
                            "post_smoking_muts_per_year": post_smoking_muts_per_year,
                        }
                    )
        print()
    mb_gradients = pd.DataFrame(data)
    mb_gradients.to_csv(f"{output_directory}/mb_gradients.csv", index=False)
mb_gradients

# %%
from ClonesModelling.parameters.hypothetical_paradigm_class import MODULE_ORDERING


def is_protection_selection_paradigm(paradigm: str) -> bool:
    protection_modules = [
        "q",
        "quiescent",
        "p",
        "protected",
        "qp",
        "quiescent_protected",
    ]
    selection_modules = ["ir", "immune_response", "sd", "smoking_driver"]
    return (
        len([module for module in paradigm.split("-") if module in protection_modules])
        <= 1
    ) and (
        len([module for module in paradigm.split("-") if module in selection_modules])
        <= 1
    )


def filter_df_to_protection_selection_paradigms(
    df: pd.DataFrame, exclude_infants: bool = False
) -> pd.DataFrame:
    if exclude_infants:
        assert "patient" in df.columns, df.columns
        # prev_length = len(df)
        df = df[~df["patient"].isin(["PD37453", "PD37455", "PD37456"])]
        # assert len(df) < prev_length, (len(df), prev_length)
    assert "paradigm" in df.columns
    # filter for paradigm.split('-') having at most one protection module and at most one selection module
    return (
        df[
            df["paradigm"].apply(
                lambda paradigm: is_protection_selection_paradigm(paradigm)
            )
        ]
        .assign(
            module_count=lambda df: df["paradigm"].apply(
                lambda paradigm: len(paradigm.split("-"))
            ),
            module_indices=lambda df: df["paradigm"].apply(
                lambda paradigm: sum(
                    MODULE_ORDERING.index(module) * 10**i
                    for i, module in enumerate(paradigm.split("-"))
                )
            ),
        )
        .sort_values(["module_count", "module_indices"], ascending=True)
        .drop(["module_count", "module_indices"], axis=1)
    )


# %%
## Plot final-timepoint mutational burden distributions for representative patients
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

representative_patient_colours = {
    # "PD34209": smoking_status_colours["non-smoker"],
    # "PD34206": smoking_status_colours["ex-smoker"],
    # "PD34204": smoking_status_colours["smoker"],
    "test_never_smoker": smoking_status_colours["non-smoker"],
    "test_ex_smoker": smoking_status_colours["ex-smoker"],
    "test_smoker": smoking_status_colours["smoker"],
}
titles = {
    # "test_never_smoker": "80-year-old\nNever smoker",
    # "test_ex_smoker": "80-year-old\nSmoked 20-60",
    # "test_smoker": "80-year-old\nSmoked 20-80",
    pt: f"{int(smoking_records[pt].age)}-year-old\n{smoking_records[pt].status.replace('_', ' ').title()}"
    for pt in representative_patient_colours
}

save_dir = f"{output_directory}/plots"

os.makedirs(f"{save_dir}/final_mb/spatial", exist_ok=True)
os.makedirs(f"{save_dir}/final_mb/non-spatial", exist_ok=True)
for paradigm in sorted(
    os.listdir(f"{output_directory}/spatial/logs"),
    key=lambda x: (len(x.split("-")), x),
):
    if not os.path.exists(f"{output_directory}/spatial/logs/{paradigm}"):
        print(f"Skipping {paradigm}; no spatial logs")
        continue
    # if (
    #     os.path.exists(f"{save_dir}/final_mb/{paradigm}.pdf")
    #     and os.path.exists(f"{save_dir}/final_mb/non-spatial/{paradigm}.pdf")
    #     and os.path.exists(f"{save_dir}/final_mb/spatial/{paradigm}.pdf")
    # ):
    #     print(f"Skipping {paradigm}; already plotted")
    #     continue
    if len(paradigm.split("-")) > 2:
        print(f"Skipping {paradigm}; too many modules")
        continue
    fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharey="row")
    for row, spatial in enumerate([False, True]):
        spatial_specific_fig, spatial_specific_axes = plt.subplots(
            1, 3, figsize=(7, 2), sharey=True
        )
        simulation = read_simulation(paradigm, spatial)
        simulation_outputs = simulation.get_simulation_output()
        for col, (patient, colour) in enumerate(representative_patient_colours.items()):
            any_plotted = False
            for replicate_index in range(simulation.n_replicates):
                # mb = sim.get_final_timepoint_mutational_burden(replicate_index, patient)
                mb = simulation_outputs[replicate_index][patient].mutational_burden
                if len(mb) == 0 or np.var(mb) == 0:
                    continue
                any_plotted = True
                for ax in [axes[row, col], spatial_specific_axes[col]]:
                    sns.kdeplot(
                        # data=mb,
                        x=mb,
                        ax=ax,
                        color=colour,
                        alpha=0.5,
                        label=patient,
                    )
            # axes[row, col].set_title(f"{patient}; {(not spatial) * 'non-'}spatial")
            axes[row, col].set_title(titles[patient])
            # spatial_specific_axes[col].set_title(patient)
            spatial_specific_axes[col].set_title(titles[patient])
            # for ax in [axes[row, col], spatial_specific_axes[col]]:
            #     ax.set_ylabel("")
            #     ax.set_yticks([])
            if col == 1:
                for ax in [axes[row, col], spatial_specific_axes[col]]:
                    ax.set_xlabel("Mutational burden")

            ylim = ax.get_ylim()
            # add kde of 1st replicate of base paradigm if spatial
            if paradigm != "base":
                base_simulation = read_simulation("base", spatial)
                base_simulation_outputs = base_simulation.get_simulation_output()
                base_mb = base_simulation_outputs[0][patient].mutational_burden
                sns.kdeplot(
                    x=base_mb,
                    ax=spatial_specific_axes[col],
                    color="black",
                    alpha=0.7,
                    label="base",
                    linestyle="--",
                )
            if any_plotted:
                ax.set_ylim(ylim)

        # # add legend to the right of the rightmost plot
        # spatial_specific_axes[-1].legend(
        #     handles=(
        #         [
        #             plt.Line2D([0], [0], color=colour, label=patient)
        #             for patient, colour in representative_patient_colours.items()
        #         ]
        #         + (
        #             [
        #                 plt.Line2D(
        #                     [0],
        #                     [0],
        #                     color="black",
        #                     label="base paradigm",
        #                     linestyle="--",
        #                     alpha=0.7,
        #                 )
        #             ]
        #             if paradigm != "base"
        #             else []
        #         )
        #     ),
        #     loc="center left",
        #     bbox_to_anchor=(1, 0.5),
        # )

        spatial_specific_fig.tight_layout()
        spatial_specific_fig.savefig(
            f"{save_dir}/final_mb/{'non-' * (not spatial)}spatial/{paradigm}.pdf"
        )
        print(f"{paradigm} {'non-' * (not spatial)}spatial")
        display(spatial_specific_fig)
        plt.close(spatial_specific_fig)

    # for each column, set the same x limits (ie min of the lowers, max of the uppers)
    for col in range(3):
        lower = min(ax.get_xlim()[0] for ax in axes[:, col])
        upper = max(ax.get_xlim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_xlim(lower, upper)

    # fig.suptitle(f"Final mb; {paradigm}")
    fig.tight_layout()
    fig.savefig(f"{save_dir}/final_mb/{paradigm}.pdf")
    # display(fig)
    plt.close(fig)


legend_fig, legend_ax = plt.subplots(figsize=(1, 1))
legend_ax.axis("off")
legend_ax.legend(
    handles=[
        plt.Line2D(
            [0], [0], color=smoking_status_colours["non-smoker"], label="never smoker"
        ),
        plt.Line2D(
            [0], [0], color=smoking_status_colours["ex-smoker"], label="ex-smoker"
        ),
        plt.Line2D(
            [0], [0], color=smoking_status_colours["smoker"], label="current smoker"
        ),
    ]
)
legend_fig.tight_layout()

# %%
## Read in/calculate mixture model data
import numpy as np
from ClonesModelling.data.mixture_model_data import fit_mixture_model, MixtureModelData


def get_mixture_model_df(sim: Simulation, log_transform: bool = False) -> pd.DataFrame:
    print(sim.paradigm_name, sim.spatial)
    data = []
    simulation_outputs = sim.get_simulation_output()
    for patient in sim.patients:
        for replicate_index in range(sim.n_replicates):
            mb = simulation_outputs[replicate_index][patient].mutational_burden
            if len(mb) < 10:
                continue
            mixture_model = fit_mixture_model(
                np.array(mb), log_transform=log_transform
            )
            data.append(
                {
                    "patient": patient,
                    "replicate_index": replicate_index,
                    "larger_weight": mixture_model.larger_weight,
                    "dominant_mean": mixture_model.dominant_mean,
                    "other_mean": mixture_model.other_mean,
                    "larger_mean_weight": mixture_model.larger_mean_weight,
                    "larger_mean": mixture_model.larger_mean,
                    "smaller_mean": mixture_model.smaller_mean,
                }
            )
    return pd.DataFrame(data).assign(
        dominant_mean=lambda df: (
            np.exp(df["dominant_mean"]) if log_transform else df["dominant_mean"]
        ),
        other_mean=lambda df: (
            np.exp(df["other_mean"]) if log_transform else df["other_mean"]
        ),
        larger_mean=lambda df: (
            np.exp(df["larger_mean"]) if log_transform else df["larger_mean"]
        ),
        smaller_mean=lambda df: (
            np.exp(df["smaller_mean"]) if log_transform else df["smaller_mean"]
        ),
    )


if not os.path.exists(f"{output_directory}/mixture_model_data.csv"):
    mixture_model_data = pd.concat(
        [
            get_mixture_model_df(
                read_simulation(paradigm, spatial), log_transform=True
            ).assign(
                paradigm=paradigm,
                paradigm_index=lambda df: df["paradigm"].map(
                    lambda x: sorted(
                        os.listdir(
                            f"{output_directory}/"
                            f"{'non-' if not spatial else ''}spatial/logs"
                        ),
                        key=lambda x: (len(x.split("-")), x),
                    ).index(x)
                ),
                spatial=spatial,
                smoking_status=lambda df: df["patient"].map(
                    lambda x: smoking_records[x].status
                ),
                smoking_status_index=lambda df: df["smoking_status"].map(
                    lambda x: ["non-smoker", "ex-smoker", "smoker"].index(x)
                ),
            )
            for spatial in [False, True]
            for paradigm in sorted(
                os.listdir(
                    f"{output_directory}/{'non-' if not spatial else ''}spatial/logs"
                ),
                key=lambda x: (len(x.split("-")), x),
            )
        ]
    )
    mixture_model_data.to_csv(
        f"{output_directory}/mixture_model_data.csv"
    )
else:
    mixture_model_data = pd.read_csv(
        f"{output_directory}/mixture_model_data.csv"
    )

# %%
mixture_model_data.larger_weight.hist()
plt.xlabel("Larger weight")
plt.ylabel("Frequency")

# %%
full_paradigm_names = {
    "base": "base",
    "q": "quiescent",
    "ir": "immune_response",
    "p": "protected",
    "sd": "smoking_driver",
    "q-ir": "quiescent-immune_response",
    "p-ir": "protected-immune_response",
    "q-sd": "quiescent-smoking_driver",
    "p-sd": "protected-smoking_driver",
}

# %%
fig, ax = plt.subplots()
sns.boxplot(
    filter_df_to_protection_selection_paradigms(
        mixture_model_data, exclude_infants=True
    ).assign(paradigm=lambda df: df["paradigm"].map(lambda x: full_paradigm_names[x])),
    x="paradigm",
    y="larger_weight",
    hue="spatial",
)
plt.xticks(rotation=15, ha="right")
print()

# %%
## boxplot of larger_weight by paradigm, hue as smoking status
for spatial in [False, True]:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=filter_df_to_protection_selection_paradigms(
            mixture_model_data[mixture_model_data.spatial == spatial],
            exclude_infants=True,
        ).assign(
            paradigm=lambda df: df["paradigm"].map(lambda x: full_paradigm_names[x])
        ),
        x="paradigm",
        y="larger_weight",
        hue="smoking_status",
        hue_order=["non-smoker", "smoker", "ex-smoker"],
        palette=smoking_status_colours,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Gaussian mixture model\nlarger weight")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    print("spatial" if spatial else "non-spatial")
    display(fig)
    os.makedirs(f"{save_dir}/mixture_model", exist_ok=True)
    fig.savefig(
        f"{save_dir}/mixture_model/"
        f"larger_weight_by_paradigm_{'' if spatial else 'non'}spatial.pdf"
    )
    plt.close(fig)

# %%
# bar plot of the proportion of patients with 1 component (ie larger_weight == 1) by paradigm, hue as smoking_status, spatial as facet
for col, spatial in enumerate([False, True]):
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(
        data=filter_df_to_protection_selection_paradigms(
            mixture_model_data[mixture_model_data.spatial == spatial]
            .sort_values(["smoking_status_index", "paradigm_index"])
            .groupby(["paradigm", "smoking_status"])["larger_weight"]
            .apply(lambda x: (x == 1).mean())
            .reset_index()
        ).assign(
            paradigm=lambda df: df["paradigm"].map(lambda x: full_paradigm_names[x])
        ),
        x="paradigm",
        y="larger_weight",
        hue="smoking_status",
        palette=smoking_status_colours,
        ax=ax,
        legend=False,
    )
    ax.set_title("Single-component mixture model")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of\nsimulated patients")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    ax.set_yticklabels([f"{float(y):.0%}" for y in ax.get_yticks()])
    fig.tight_layout()
    print("spatial" if spatial else "non-spatial")
    fig.savefig(
        f"{save_dir}/mixture_model/1-component_proportion_"
        f"{'spatial' if spatial else 'non-spatial'}.pdf"
    )
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                color=smoking_status_colours["non-smoker"],
                label="never smoker",
                lw=5,
            ),
            plt.Line2D(
                [0],
                [0],
                color=smoking_status_colours["ex-smoker"],
                label="ex-smoker",
                lw=5,
            ),
            plt.Line2D(
                [0],
                [0],
                color=smoking_status_colours["smoker"],
                label="current smoker",
                lw=5,
            ),
        ],
        bbox_to_anchor=(1, 1),
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(
        f"{save_dir}/mixture_model/1-component_proportion_"
        f"{'spatial' if spatial else 'non-spatial'}_with_legend.pdf"
    )
    display(fig)
    plt.close(fig)


legend_fig, legend_ax = plt.subplots(figsize=(1, 1))
legend_ax.axis("off")
legend_ax.legend(
    handles=[
        plt.Line2D(
            [0],
            [0],
            color=smoking_status_colours["non-smoker"],
            label="never smoker",
            lw=5,
        ),
        plt.Line2D(
            [0], [0], color=smoking_status_colours["ex-smoker"], label="ex-smoker", lw=5
        ),
        plt.Line2D(
            [0],
            [0],
            color=smoking_status_colours["smoker"],
            label="current smoker",
            lw=5,
        ),
    ]
)
legend_fig.tight_layout()
display(legend_fig)
plt.close(legend_fig)

# %%
## Collect tree_balance_indices for each patient and paradigm
import numpy as np

data = []
for paradigm in sorted(
    os.listdir(f"{output_directory}/non-spatial/logs"),
    key=lambda x: (len(x.split("-")), x),
):
    if not os.path.exists(
        f"{output_directory}/spatial/logs/{paradigm}"
    ):
        continue
    for spatial in [False, True]:
        sim = read_simulation(paradigm, spatial=spatial)
        simulation_outputs = sim.get_simulation_output()
        for patient in sorted(
            sim.patients, key=lambda x: (smoking_records[x].status, x)
        ):
            tree_balance_indices = [
                simulation_output[patient].tree_balance_indices
                for simulation_output in simulation_outputs
                if simulation_output[patient].tree_balance_indices is not None
            ]
            if not tree_balance_indices:
                continue
            data.append(
                pd.DataFrame(
                    {
                        "paradigm": paradigm,
                        "paradigm_index": sorted(
                            os.listdir(
                                f"{output_directory}/"
                                f"{'non-' if not spatial else ''}spatial/logs"
                            ),
                            key=lambda x: (len(x.split("-")), x),
                        ).index(paradigm),
                        "tree_balance_index": np.concatenate(
                            tree_balance_indices, axis=None
                        ),
                        "patient": patient,
                        "smoking_status": smoking_records[patient].status,
                        "smoking_status_index": [
                            "non-smoker",
                            "ex-smoker",
                            "smoker",
                        ].index(smoking_records[patient].status),
                        "spatial": spatial,
                    }
                )
            )

# Convert to DataFrame
df_tree_balance = pd.concat(data)
df_tree_balance

# %%
## Plot boxplot of tree_balance_indices by paradigm, hue as smoking_status, spatial as facet
for spatial in [False, True]:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.boxplot(
        # data=df_tree_balance[df_tree_balance.spatial == spatial].sort_values(
        #     ["smoking_status_index", "paradigm_index"]
        # ),
        data=filter_df_to_protection_selection_paradigms(
            df_tree_balance[df_tree_balance.spatial == spatial],
            exclude_infants=True,
        )
        .sort_values(["smoking_status_index", "paradigm_index"])
        .assign(
            paradigm=lambda df: df["paradigm"].map(lambda x: full_paradigm_names[x])
        ),
        x="paradigm",
        order=full_paradigm_names.values(),
        y="tree_balance_index",
        hue="smoking_status",
        palette=smoking_status_colours,
        ax=ax,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_ylabel("Tree balance index")
    plt.xticks(rotation=25, ha="right")
    ax.set_ylim(-0.02, None)
    fig.tight_layout()
    print("spatial" if spatial else "non-spatial")
    display(fig)
    fig.savefig(
        f"{save_dir}/tree_balance_index_by_paradigm_"
        f"{'spatial' if spatial else 'non-spatial'}.pdf"
    )
    plt.close(fig)

# %%
# # now separate boxplot for each paradigm with x=patient
# for spatial in [False, True]:
#     tree_balance_save_directory = (
#         f"{save_dir}/tree_balance_indices/{'spatial' if spatial else 'non-spatial'}"
#     )
#     os.makedirs(tree_balance_save_directory, exist_ok=True)
#     for paradigm in sorted(df_tree_balance["paradigm"].unique()):
#         if not is_protection_selection_paradigm(paradigm):
#             continue
#         fig, ax = plt.subplots(figsize=(5, 3))
#         sns.boxplot(
#             data=df_tree_balance[
#                 (df_tree_balance["paradigm"] == paradigm)
#                 & (df_tree_balance["spatial"] == spatial)
#             ].sort_values(
#                 "smoking_status",
#                 key=lambda x: x.apply(
#                     lambda x: list(smoking_status_colours.keys()).index(x)
#                 ),
#             ),
#             x="patient",
#             y="tree_balance_index",
#             hue="smoking_status",
#             palette=smoking_status_colours,
#             legend=False,
#             ax=ax,
#         )
#         ax.set_ylabel("Tree Balance Index")
#         plt.xticks(rotation=45, ha="right")
#         ax.set_ylim(0, 1)
#         fig.tight_layout()
#         fig.savefig(f"{tree_balance_save_directory}/{paradigm}.pdf")
#         print(paradigm, "spatial" if spatial else "non-spatial")
#         display(fig)
#         plt.close(fig)


# legend_fig, legend_ax = plt.subplots(figsize=(1, 1))
# legend_ax.axis("off")
# legend_ax.legend(
#     handles=[
#         plt.Line2D([0], [0], color="green", label="never smoker"),
#         plt.Line2D([0], [0], color="orange", label="ex-smoker"),
#         plt.Line2D([0], [0], color="red", label="current smoker"),
#     ]
# )
# legend_fig.tight_layout()
# display(legend_fig)
# legend_fig.savefig(f"{save_dir}/tree_balance_indices/legend.pdf")
# plt.close(legend_fig)

# %%
# plot branch_lengths, kde for each representative patient

for log_scale in [True, False]:
    branch_lengths_save_dir = (
        f"{save_dir}/branch_lengths/{'log' if log_scale else 'linear'}_scale"
    )
    for paradigm in sorted(
        os.listdir(f"{output_directory}/non-spatial/logs"),
        key=lambda x: (len(x.split("-")), x),
    ):
        if not is_protection_selection_paradigm(paradigm):
            continue
        if not os.path.exists(f"{output_directory}/spatial/logs/{paradigm}"):
            print("skipping", paradigm)
            continue
        os.makedirs(branch_lengths_save_dir + "/spatial", exist_ok=True)
        os.makedirs(branch_lengths_save_dir + "/non-spatial", exist_ok=True)
        for spatial in [False, True]:
            save_filepath = f"{branch_lengths_save_dir}/{'spatial' if spatial else 'non-spatial'}/{paradigm}.pdf"
            if os.path.exists(save_filepath):
                print(paradigm, spatial, "already plotted at ", save_filepath)
                continue
            fig, axes = plt.subplots(1, 3, figsize=(7, 2.5), sharex=True, sharey=True)
            sim = read_simulation(paradigm, spatial=spatial)
            for col, (patient, colour, ax) in enumerate(
                zip(
                    representative_patient_colours.keys(),
                    representative_patient_colours.values(),
                    axes,
                )
            ):
                for replicate_index in range(sim.n_replicates):
                    patient_simulation_output = sim.get_simulation_output()[
                        replicate_index
                    ][patient]
                    if patient_simulation_output.phylogeny_branch_lengths is None:
                        continue
                    # for subsample_index in range(
                    #     patient_simulation_output.phylogeny_branch_lengths.shape[0]
                    # ):
                    for subsample_index in range(10):
                        sns.kdeplot(
                            data=patient_simulation_output.phylogeny_branch_lengths[
                                subsample_index
                            ]
                            + (1 if log_scale else 0),
                            ax=ax,
                            color=colour,
                            alpha=0.3,
                            label=patient,
                            log_scale=log_scale,
                        )
                ax.set_title(titles[patient])
                ax.set_xlabel("Branch length")

            # add legend to the right of the rightmost plot
            axes[-1].legend(
                handles=[
                    plt.Line2D([0], [0], color=colour, label=patient)
                    for patient, colour in representative_patient_colours.items()
                ],
                loc="center left",
                bbox_to_anchor=(1, 0.5),
            )

            fig.tight_layout()
            fig.savefig(save_filepath)
            print(paradigm, "spatial" if spatial else "non-spatial")
            display(fig)
            plt.close(fig)
        # # for each column, set the same x limits (ie min of the lowers, max of the uppers)
        # for col in range(3):
        #     lower = min(ax.get_xlim()[0] for ax in axes[:, col])
        #     upper = max(ax.get_xlim()[1] for ax in axes[:, col])
        #     for ax in axes[:, col]:
        #         ax.set_xlim(lower, upper)

# %%
full_paradigm_names = {
    "base": "base",
    "q": "quiescent",
    "ir": "immune_response",
    "p": "protected",
    "sd": "smoking_driver",
    "q-ir": "quiescent-immune_response",
    "p-ir": "protected-immune_response",
    "q-sd": "quiescent-smoking_driver",
    "p-sd": "protected-smoking_driver",
}

# %%
outlier_paradigms = ["q", "q-sd", "q-ir"]

for spatial in [True, False]:
    for exclude_outliers in [True, False]:
        fig, axes = plt.subplots(
            1, 3, figsize=(7, 3.6), sharey=True
        )
        for colname, ax in zip(
            [
                "pre_smoking_muts_per_year",
                "smoking_muts_per_year",
                "post_smoking_muts_per_year",
            ],
            axes,
        ):
            sns.boxplot(
                data=(
                    filter_df_to_protection_selection_paradigms(
                        mb_gradients[mb_gradients.spatial == spatial]
                    ).loc[~mb_gradients[colname].isna()]
                    if not exclude_outliers
                    else filter_df_to_protection_selection_paradigms(
                        mb_gradients[mb_gradients.spatial == spatial]
                    ).loc[
                        ~mb_gradients[colname].isna()
                        & ~mb_gradients["paradigm"].isin(outlier_paradigms)
                    ]
                ).assign(paradigm=lambda df: df["paradigm"].map(full_paradigm_names)),
                x=colname,
                y="paradigm",
                hue="smoking_status",
                order=[
                    x
                    for x in full_paradigm_names.values()
                    if not exclude_outliers
                    or x
                    not in [
                        full_paradigm_names[paradigm] for paradigm in outlier_paradigms
                    ]
                ],
                palette=smoking_status_colours,
                ax=ax,
                legend=False,
                showfliers=False,
            )
            ax.set_title("-".join(colname.split("_")[:-3]).title(), fontsize=10)
            if colname == "smoking_muts_per_year":
                ax.set_xlabel("Yearly change in Mutational Burden")
            else:
                ax.set_xlabel("")
            # ax.set_yticks([])
        # hlines at 25,100,25 (for axes 0,1,2)
        for ax, x in zip(axes, [25, 100, 25]):
            ax.axvline(x=x, color="black", linestyle="--", alpha=0.5)
        axes[0].legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    color=smoking_status_colours[status],
                    label=status,
                    linewidth=5,
                )
                for status in ["non-smoker", "ex-smoker", "smoker"]
            ],
            # loc="center left",
            bbox_to_anchor=(-0.6, 1.07),
        )
        axes[0].set_ylabel("Paradigm")
        fig.tight_layout()
        fig.subplots_adjust(left=0.34)
        print(
            "spatial" if spatial else "non-spatial",
            "excluding" if exclude_outliers else "including",
            ", ".join(outlier_paradigms),
        )
        display(fig)
        os.makedirs(save_dir + "/mb_gradients", exist_ok=True)
        fig.savefig(
            f"{save_dir}/mb_gradients/{'spatial' if spatial else 'non-spatial'}"
            f"{'_exclude_q_ir_p_ir' if exclude_outliers else ''}.pdf"
        )
        plt.close(fig)

# %%
mb_gradients

# %%
# print out mb_gradients spatial=True 25 - post_smoking_muts_per_year mean in paradigm q,ir,q-ir, patient=='test_ex_smoker'
{
    paradigm: (25 - x)
    for paradigm, x in (
        mb_gradients.loc[
            (mb_gradients["spatial"] == True)
            & (mb_gradients["paradigm"].isin(["q", "ir", "q-ir"]))
            & (mb_gradients["patient"] == "test_ex_smoker"),
        ]
        .groupby("paradigm")["post_smoking_muts_per_year"]
        .mean()
        .to_dict()
        .items()
    )
}

# %%
# new plot to show non-spatial->spatial->true has an increase in the std among non-smokers (to be a one-peak situation)
# base paradigm only
from ClonesModelling.data.mutations_data import get_total_mutations_data_per_patient


if os.path.exists(f"{output_directory}/mb_std_df.csv"):
    mb_std_df = pd.read_csv(f"{output_directory}/mb_std_df.csv")
else:
    total_mutations_data = get_total_mutations_data_per_patient(
        include_smoking_signatures=True
    )

    mb_std_dfs = [
        pd.DataFrame(
            {
                "patient": list(total_mutations_data.keys()),
                "setting": "true_data",
                "std": [
                    np.std(total_mutations_data[patient][:, 0])
                    for patient in total_mutations_data.keys()
                ],
                "mean": [
                    np.mean(total_mutations_data[patient][:, 0])
                    for patient in total_mutations_data.keys()
                ],
                "age": [80 for _ in total_mutations_data.keys()],
                # smoking_records[patient].age for patient in total_mutations_data.keys()
                "status": list(
                    map(
                        lambda patient: patient.replace("test_", "").replace(
                            "never", "non"
                        ),
                        total_mutations_data.keys(),
                    )
                ),
                # smoking_records[patient].status for patient in total_mutations_data.keys()
            }
        )
        # ).loc[
        #     # lambda df: df.patient.isin(
        #     #     [sr.patient for sr in smoking_records.values() if sr.status == "non-smoker"]
        #     # )
        # ]
    ]
    for spatial in [False, True]:
        for paradigm in sorted(
            os.listdir(f"{output_directory}/spatial/logs"),
            key=lambda x: (
                len(x.split("-")),
                *(
                    MODULE_ORDERING.index(x.split("-")[i])
                    for i in range(len(x.split("-")))
                ),
            ),
        ):
            # for paradigm in ["base"]:
            simulation = read_simulation(paradigm, spatial)
            simulation_outputs = simulation.get_simulation_output()
            for patient in total_mutations_data.keys():
                if (
                    patient
                    not in simulation.patients
                    # or smoking_records[patient].status != "non-smoker"
                ):
                    continue
                for replicate_index in range(simulation.n_replicates):
                    mb = simulation_outputs[replicate_index][patient].mutational_burden
                    if len(mb) == 0:
                        continue
                    mb_std_dfs.append(
                        pd.DataFrame(
                            {
                                "paradigm": [paradigm],
                                "patient": [patient],
                                "setting": (
                                    "non-spatial_simulation"
                                    if not spatial
                                    else "spatial_simulation"
                                ),
                                "std": [np.std(mb)],
                                "mean": [np.mean(mb)],
                                "age": [smoking_records[patient].age],
                                "status": [smoking_records[patient].status],
                            }
                        )
                    )
    mb_std_df = pd.concat(mb_std_dfs)
    mb_std_df.to_csv(f"{output_directory}/mb_std_df.csv", index=False)
mb_std_df

# %%
# plot the std of the mutational burden for each smoking_status=non-smoker, hue as non-spatial/spatial/true
fig, ax = plt.subplots(figsize=(5, 4))

sns.boxplot(
    data=mb_std_df.sort_values("setting").loc[
        lambda df: (df.paradigm == "base") | df.paradigm.isna()
    ],
    hue="status",
    y="std",
    x="setting",
    palette="tab10",
    ax=ax,
    showfliers=False,
)
ax.set_ylabel("Standard deviation of mutational burden")
ax.set_xlabel("")
ax.set_yscale("log")
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
display(fig)
fig.savefig(f"{save_dir}/mb_std.pdf")
plt.close(fig)

# %%
# make new mb_std_fraction, where the simulation stds are divided by the true data std for that patient
mb_std_fraction = (
    mb_std_df.loc[
        lambda df: df["setting"].isin(["non-spatial_simulation", "spatial_simulation"])
        & (df["paradigm"] == "base")
    ]
    .merge(
        mb_std_df.loc[lambda df: df["setting"] == "true_data"],
        on="patient",
        suffixes=["_simulation", "_true_data"],
    )
    .assign(
        std_fraction=lambda df: df["std_simulation"] / df["std_true_data"],
        spatial=lambda df: df["setting_simulation"] == "spatial_simulation",
    )
    .drop(["setting_simulation", "setting_true_data"], axis=1)
)
fig, ax = plt.subplots(figsize=(5, 4))
sns.boxplot(
    data=mb_std_fraction.loc[lambda df: df.patient.str.contains("PD")],
    x="patient",
    y="std_fraction",
    hue="spatial",
    showfliers=False,
    ax=ax,
)
ax.set_ylabel("Simulated:True data ratio of\nmutational burden standard deviation")
ax.set_xlabel("")
ax.set_ylim(0, None)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# %%
# plot standard deviation / sqrt(mean mb) for each patient, true, non-spatial, spatial
fig, ax = plt.subplots(figsize=(6, 4))

sns.boxplot(
    data=mb_std_df.assign(
        std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"])
    ).sort_values("setting"),
    x="setting",
    y="std_over_sqrt_mean",
    hue="setting",
    ax=ax,
    showfliers=False,
)

# %%
mean_std_over_sqrt_mean = (
    mb_std_df[
        (mb_std_df["setting"] == "true_data")
        & ((mb_std_df["paradigm"] == "base") | (mb_std_df["paradigm"].isna()))
    ]
    .assign(
        std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"]),
        status=lambda df: df["patient"].map(lambda x: smoking_records[x].status),
    )
    .groupby("status")["std_over_sqrt_mean"]
    .mean()
    .to_dict()
)
print(mean_std_over_sqrt_mean)

# %%
# same but restrict to each setting
# Plot for setting == 'true_data'
import math


fig, ax = plt.subplots(figsize=(6, 2))
sns.stripplot(
    data=mb_std_df[
        (mb_std_df["setting"] == "true_data")
        & ((mb_std_df["paradigm"] == "base") | (mb_std_df["paradigm"].isna()))
    ]
    .assign(
        std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"]),
        age=lambda df: df["patient"].map(lambda x: smoking_records[x].age),
        status=lambda df: df["patient"].map(lambda x: smoking_records[x].status),
    )
    .sort_values(["age"]),
    x="std_over_sqrt_mean",
    y="status",
    hue="status",
    order=["non-smoker", "ex-smoker", "smoker"],
    palette=smoking_status_colours,
    ax=ax,
    # showfliers=False,
    legend=False,
    log_scale=True,
)
ax.axvline(1.33, linestyle="--", color="black", alpha=0.5)
ax.axvline(1.64, linestyle="--", color="black", alpha=0.5)
ax.text(1.33, 1, "Non-smoking", ha="right", va="center", rotation=90)
ax.text(1.64, 1, "Smoking", ha="right", va="center", rotation=90)
ax.set_xlabel(
    "Standard deviation of mutational burden\nover square root of mean mutational burden"
)
fig.tight_layout()
print("true_data")
os.makedirs(f"{save_dir}/mb_std_over_sqrt_mean", exist_ok=True)
fig.tight_layout()
fig.savefig(f"{save_dir}/mb_std_over_sqrt_mean/true_data_with_legend.pdf")
display(fig)
plt.close(fig)

# get mean of std_over_sqrt_mean in true_data setting for each smoking status as dict
mean_std_over_sqrt_mean = (
    mb_std_df[
        (mb_std_df["setting"] == "true_data")
        & ((mb_std_df["paradigm"] == "base") | (mb_std_df["paradigm"].isna()))
    ]
    .assign(
        std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"]),
        status=lambda df: df["patient"].map(lambda x: smoking_records[x].status),
    )
    .groupby("status")["std_over_sqrt_mean"]
    .mean()
    .to_dict()
)
print(mean_std_over_sqrt_mean)
# Plot for other settings
for setting in mb_std_df["setting"].unique():
    if setting == "true_data":
        continue
    for by_patient in [True, False]:
        # first, single plot for 'base' paradigm
        fig, ax = plt.subplots(figsize=(7 if by_patient else 3, 4))
        sns.boxplot(
            data=mb_std_df[
                (mb_std_df["setting"] == setting)
                & ((mb_std_df["paradigm"] == "base") | (mb_std_df["paradigm"].isna()))
            ]
            .assign(
                std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"]),
                age=lambda df: df["patient"].map(lambda x: smoking_records[x].age),
                status=lambda df: df["patient"].map(
                    lambda x: smoking_records[x].status
                ),
            )
            .sort_values(["age"]),
            x="patient" if by_patient else "status",
            order=(["non-smoker", "ex-smoker", "smoker"] if not by_patient else None),
            y="std_over_sqrt_mean",
            hue="status",
            palette=smoking_status_colours,
            ax=ax,
            showfliers=False,
            legend=False,
            log_scale=True,
        )
        ax.set_xlabel("Patient" if by_patient else "Smoking status")
        ax.set_ylabel("Dispersion ratio")
        plt.xticks(rotation=90)
        ax.axhline(1.33, linestyle="--", color="black", alpha=0.5)
        ax.axhline(1.64, linestyle="--", color="black", alpha=0.5)
        xmax = ax.get_xlim()[1]
        ax.text(
            1 if not by_patient else 34 if setting == "non-spatial_simulation" else 10,
            1.33,
            "Non-smoking" if setting == "non-spatial_simulation" else "Non-\nsmoking",
            ha="center",
            va="top",
        )
        ax.text(
            1 if not by_patient else 34 if setting == "non-spatial_simulation" else 3,
            1.64,
            "Smoking",
            ha="center",
            va="bottom",
        )
        os.makedirs(
            f"{save_dir}/mb_std_over_sqrt_mean/{'by_patient' if by_patient else 'by_status'}",
            exist_ok=True,
        )
        fig.tight_layout()
        fig.savefig(
            f"{save_dir}/mb_std_over_sqrt_mean/{'by_patient' if by_patient else 'by_status'}/{setting}_base.pdf"  # {'_with_legend' * legend}/{paradigm}.pdf"
        )
        display(fig)
        plt.close(fig)

        # then one with the other single-hypothesis paradigms
        fig, axes = plt.subplots(
            4 if by_patient else 1,
            1 if by_patient else 4,
            figsize=(7, 8) if by_patient else (7, 3.4),
            sharex=True,
            sharey=not by_patient,
        )
        for paradigm, ax in zip(
            [
                p
                for p in mb_std_df["paradigm"].unique()
                if p == p and p != "base" and len(p.split("-")) == 1
            ],
            axes,
        ):
            # ax.set_xscale("log")
            sns.boxplot(
                data=mb_std_df[
                    (mb_std_df["setting"] == setting)
                    & ((mb_std_df["paradigm"] == paradigm))
                ]
                .assign(
                    std_over_sqrt_mean=lambda df: df["std"] / np.sqrt(df["mean"]),
                    age=lambda df: df["patient"].map(lambda x: smoking_records[x].age),
                    status=lambda df: df["patient"].map(
                        lambda x: smoking_records[x].status
                    ),
                )
                .sort_values(["age"]),
                x="patient" if by_patient else "status",
                order=(
                    ["non-smoker", "ex-smoker", "smoker"] if not by_patient else None
                ),
                y="std_over_sqrt_mean",
                hue="status",
                palette=smoking_status_colours,
                ax=ax,
                showfliers=False,
                legend=False,
                log_scale=True,
            )
            ax.set_title(
                {
                    "p": "Protected",
                    "q": "Quiescent",
                    "ir": "Immune Response",
                    "sd": "Smoking Driver",
                }[
                    paradigm
                ],  # .replace(" ", " " if by_patient else "\n"),
                fontsize=10,
            )
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.axhline(1.33, linestyle="--", color="black", alpha=0.5)
            ax.axhline(1.64, linestyle="--", color="black", alpha=0.5)
            ax.set_ylabel("Dispersion ratio")
            if by_patient:
                ax.set_xlabel("Patient (ordered by increasing age)")
                plt.xticks(rotation=90)
            else:
                ax.set_xlabel("")
                ax.set_xticks(ax.get_xticks())
                ax.set_xticklabels(
                    [
                        label.get_text().replace("-", " ").title().replace(" ", "-")
                        for label in ax.get_xticklabels()
                    ],
                    rotation=30,
                    ha="right",
                )
                if paradigm == "q":
                    ax.text(1, 1.33, "Non-smoking", ha="center", va="top")
                    ax.text(1, 1.64, "Smoking", ha="center", va="bottom")
                # add lines for mean of true data
                xmin, xmax = ax.get_xlim()
                for status, x_value in zip(
                    ["non-smoker", "ex-smoker", "smoker"], ax.get_xticks()
                ):
                    width = 0.25
                    projected_x_value = (x_value - xmin) / (xmax - xmin)
                    ax.axhline(
                        mean_std_over_sqrt_mean[status],
                        xmin=projected_x_value - width / 2,
                        xmax=projected_x_value + width / 2,
                        # xmin=x_value / x_range - width / 2,
                        # xmax=x_value / x_range + width / 2,
                        linestyle="--",
                        color=smoking_status_colours[status],
                        alpha=0.5,
                    )

                # if paradigm == "sd":
                #     ax.legend(
                #         handles=[
                #             plt.Line2D(
                #                 [0],
                #                 [0],
                #                 color=smoking_status_colours[status],
                #                 label=status.replace("-", " ")
                #                 .title()
                #                 .replace(" ", "-"),
                #                 lw=5,
                #             )
                #             for status in ["non-smoker", "ex-smoker", "smoker"]
                #         ]
                #         + [
                #             plt.Line2D(
                #                 [0],
                #                 [0],
                #                 color="black",
                #                 linestyle="--",
                #                 lw=1,
                #                 alpha=0.5,
                #                 label="Smoking/\nNon-smoking\nExpected\nRatios",
                #             )
                #         ],
                #         loc="center left",
                #         bbox_to_anchor=(1, 0.5),
                #     )
            # else:
            #     ax.axvline(1.33, linestyle="--", color="black", alpha=0.5)
            #     ax.axvline(1.64, linestyle="--", color="black", alpha=0.5)
            #     ax.set_xlabel("Dispersion ratio")
            #     ax.set_ylabel("")
            # fig.tight_layout()
            # print(setting, paradigm)
            # if paradigm in ["base", "ir", "sd"]:
            #     ax.set_xlim(
            #         10 ** math.floor(np.log10(ax.get_xlim()[0])),
            #         10 ** math.ceil(np.log10(ax.get_xlim()[1])),
            #     )

        fig.tight_layout()
        if not by_patient:
            # adjust wspace
            fig.subplots_adjust(wspace=0.25)
        fig.savefig(
            f"{save_dir}/mb_std_over_sqrt_mean/{'by_patient' if by_patient else 'by_status'}/{setting}.pdf"  # {'_with_legend' * legend}/{paradigm}.pdf"
        )
        print(setting)
        display(fig)
        plt.close(fig)
