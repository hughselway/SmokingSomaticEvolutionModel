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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

mutation_colnames = [
    "driver_non_smoking_signature_mutations",
    "driver_smoking_signature_mutations",
    "passenger_non_smoking_signature_mutations",
    "passenger_smoking_signature_mutations",
]


def is_smoking(test_patient: str, age: int) -> bool:
    if test_patient == "test_smoker":
        return 20 < age
    if test_patient == "test_ex_smoker":
        return 20 < age < 60
    if test_patient == "test_never_smoker":
        return False


status_colours = {
    # "test_smoker": "red",
    # "test_ex_smoker": "orange",
    # "test_never_smoker": "green",
    "test_never_smoker": "#5d4b98",
    "test_smoker": "#ab5c9f",
    "test_ex_smoker": "#80bb51",
}
titles = {
    "test_never_smoker": "80-year-old\nNever smoker",
    "test_smoker": "80-year-old\nSmoked 20-80",
    "test_ex_smoker": "80-year-old\nSmoked 20-60",
}

# # set all fontsizes to 12
# plt.rcParams.update({"font.size": 12})


def plot_animated_spatial_simulation(
    mb_csv_filepaths: dict[str, str], plot_filepath: str, records_per_year: int = 1
) -> None:
    patients = sorted(
        list(mb_csv_filepaths.keys()),
        key=lambda x: list(status_colours.keys()).index(x),
    )

    # fig = plt.figure(figsize=(2.5 * len(patients), 6))
    # axes = fig.subplots(2, len(patients), sharey="row")
    # instead, have third row for colourbar (single column in that row, via gridspec)
    fig = plt.figure(figsize=(2.5 * len(patients), 6.2))
    gs = fig.add_gridspec(3, len(patients), height_ratios=[0.3, 0.58, 0.035])
    axes = [
        [fig.add_subplot(gs[0, i]) for i in range(len(patients))],
        [fig.add_subplot(gs[1, i]) for i in range(len(patients))],
    ]
    # axes = gs.subplots(sharey="row")
    for i in range(1, len(patients)):
        axes[0][i].sharey(axes[0][0])
        plt.setp(axes[0][i].get_yticklabels(), visible=False)

    mb = {
        patient: pd.read_csv(mb_csv_filepaths[patient])
        .assign(total_mutations=lambda df: df[mutation_colnames].sum(axis=1))
        .groupby("record_number")
        for patient in patients
    }

    step_numbers = sorted(map(int, mb[patients[0]].groups.keys()))
    # for debugging: step_numbers = list(range(1, 10))

    max_mutations = max(
        max(df["total_mutations"].max() for _, df in mb[patient])
        for patient in patients
    )
    cmap, norm = plt.cm.viridis, plt.Normalize(vmin=0, vmax=max_mutations)

    def animate(timestep):
        if timestep % 30 == 0:
            print(int(timestep / records_per_year), end=" ")

        for patient_index, (ax0, ax1, patient) in enumerate(
            zip(axes[0], axes[1], patients)
        ):
            mb_patient = mb[patient].get_group(timestep)

            ax0.clear()
            if mb_patient.total_mutations.var() > 0:
                sns.kdeplot(
                    mb_patient.total_mutations,
                    ax=ax0,
                    fill=True,
                    legend=False,
                    color=status_colours[patient],
                )
            ax0.set_xlim(0, max_mutations * 1.1)
            ax0.set_xlabel("Mutational burden", fontsize=10)
            # ax0.set_ylim(0, 0.05)
            if patient_index != 0:
                # ax0.set_yticklabels([])
                plt.setp(ax0.get_yticklabels(), visible=False)
                ax0.set_ylabel("")
            ax0.set_title(titles[patient])

            mb_patient = mb_patient[mb_patient["compartment"] != "quiescent"]
            ax1.clear()
            heatmap_data = mb_patient.pivot(
                index="y", columns="x", values="total_mutations"
            )
            sns.heatmap(
                heatmap_data,
                ax=ax1,
                cmap=cmap,
                norm=norm,
                cbar=False,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            ax1.set_aspect("equal")
            ax1.annotate(
                (
                    "smoking"
                    if is_smoking(patient, int(timestep / records_per_year))
                    else "not smoking"
                ),
                xy=(0.5, -0.075),
                xycoords="axes fraction",
                fontsize=12,
                horizontalalignment="center",
                verticalalignment="top",
            )

        axes[1][1].set_title("Spatial distribution of mutational burden")
        axes[1][1].annotate(
            f"year: {int(timestep/records_per_year)} / {int(max(step_numbers)/records_per_year)}",
            xy=(0.5, -0.225),
            xycoords="axes fraction",
            fontsize=12,
            horizontalalignment="center",
            verticalalignment="top",
        )
        fig.savefig(
            plot_filepath.replace(".gif", f"/{timestep}.pdf"),
        )

    ani = animation.FuncAnimation(fig, animate, interval=200, frames=step_numbers)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_subplot(gs[2, :])
    cbar_ax.set_frame_on(False)
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])
    fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        label="Mutational burden",
    )
    cbar_ax.tick_params(labelsize=12)
    cbar_ax.set_xlabel("Mutational burden", fontsize=12)

    os.makedirs(os.path.dirname(plot_filepath), exist_ok=True)
    ani.save(plot_filepath, fps=4)
    display(fig)
    plt.close(fig)


sim_directory = "notebooks/24-10-09-default-simulation_status_rep_pts/spatial/logs"
# sim_directory = "notebooks/data/25-03-19_fcd_gridsearch/fcs_0.05/fcp_0.25/st_0.115"
# sim_directory = "notebooks/data/25-03-19_fcd_gridsearch/fcs_0.05/fcp_0.25/st_0.11"
for paradigm in os.listdir(sim_directory):  # ['ir-p']:
    print(paradigm)
    cell_records_dir = (
        f"{sim_directory}/{paradigm}/replicate_0/cell_records/mutational_burden"
    )
    os.makedirs(
        f"notebooks/plots/25-02-17_animated_default_param_sim_plot/{paradigm}",
        exist_ok=True,
    )
    plot_animated_spatial_simulation(
        {
            "test_smoker": f"{cell_records_dir}/test_smoker.csv",
            "test_ex_smoker": f"{cell_records_dir}/test_ex_smoker.csv",
            "test_never_smoker": f"{cell_records_dir}/test_never_smoker.csv",
        },
        f"notebooks/plots/25-02-17_animated_default_param_sim_plot/{paradigm}.gif",
        # f"notebooks/plots/25-03-19_fcd_gridsearch/anim_fcs_0.05-fcp_0.25-st_0.11/{paradigm}.gif",
    )
