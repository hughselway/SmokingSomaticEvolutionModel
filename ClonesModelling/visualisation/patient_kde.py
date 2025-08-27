import os
import math
from scipy.stats import gaussian_kde  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from ..data.mutations_data import get_patient_data
from ..data.smoking_records import get_smoking_records


def plot_patient_kdes(
    kde_plot_folder: str, same_x: bool = False, log_scale_x: bool = False
):
    figures: dict[str, dict[str, plt.Figure]] = {}
    axes: dict[str, dict[str, plt.Axes]] = {}

    smoking_records = {
        smoking_record.patient: smoking_record
        for smoking_record in get_smoking_records()
    }
    patient_data = {
        patient: get_patient_data(patient)["total_mutations"]
        for patient in smoking_records.keys()
    }
    colours = {
        "smoker": "red",
        "ex-smoker": "green",
        "non-smoker": "blue",
        None: "black",
    }
    if same_x:
        # find upper and lower x limits by finding max and min of all patients' mutational burdens
        highest_x_limit = max(max(patient_data[patient]) for patient in patient_data)
        lowest_x_limit = min(min(patient_data[patient]) for patient in patient_data)
    for plot_type in ["kde", "hist", "kde_hist"]:
        figures[plot_type] = {}
        axes[plot_type] = {}
        for i, smoking_record in enumerate(smoking_records.values()):
            fig, axis = plt.subplots()
            figures[plot_type][smoking_record.patient] = fig
            axes[plot_type][smoking_record.patient] = axis
            if "hist" in plot_type:
                axis.hist(
                    patient_data[smoking_record.patient],
                    color=colours[smoking_record.status],
                )
            if "kde" in plot_type:
                kde_axis = axis.twinx()
                fill_in_interval = (
                    (
                        np.linspace(
                            min(patient_data[smoking_record.patient]) - 5,
                            max(patient_data[smoking_record.patient]) + 5,
                            1000,
                        )
                        if not same_x
                        else np.linspace(lowest_x_limit, highest_x_limit, 1000)
                    )
                    if not log_scale_x
                    else (
                        np.logspace(
                            np.log10(min(patient_data[smoking_record.patient]) - 5),
                            np.log10(max(patient_data[smoking_record.patient]) + 5),
                            1000,
                            base=10,
                        )
                        if not same_x
                        else np.logspace(
                            np.log10(lowest_x_limit),
                            np.log10(highest_x_limit),
                            1000,
                            base=10,
                        )
                    )
                )
                kde = gaussian_kde(
                    patient_data[smoking_record.patient]
                    if not log_scale_x
                    else np.log10(patient_data[smoking_record.patient])
                )

                kde_axis.plot(
                    fill_in_interval,
                    (
                        kde(fill_in_interval)
                        if not log_scale_x
                        else kde(np.log10(fill_in_interval))
                    ),
                    color=colours[smoking_record.status],
                )
                # shade underneath the curve
                kde_axis.fill_between(
                    fill_in_interval,
                    (
                        kde(fill_in_interval)
                        if not log_scale_x
                        else kde(np.log10(fill_in_interval))
                    ),
                    alpha=0.2,
                    color=colours[smoking_record.status],
                )
                kde_axis.set_yticks([])
                kde_axis.set_ylabel("")
            if log_scale_x:
                axis.set_xscale("log")
            axis.set_xlabel("mutational burden", fontsize=15)
            axis.set_ylabel("Frequency in data", fontsize=15)
            axis.set_title(f"{smoking_record.patient}", fontsize=20)
            fig.set_facecolor("white")
            os.makedirs(f"{kde_plot_folder}/{plot_type}", exist_ok=True)
            fig.savefig(f"{kde_plot_folder}/{plot_type}/{smoking_record.patient}.png")
            plt.close(fig)

        # plot all axes in a grid
        n_cols = 6
        n_rows = math.ceil(len(figures[plot_type]) // n_cols)
        fig, axis = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(5 * n_cols, 4.25 * n_rows),
        )

        # reorder plots to be in order of smoking status
        # (smokers, ex-smokers, never-smokers)
        figures[plot_type] = {
            patient: figures[plot_type][patient]
            for patient in sorted(
                figures[plot_type],
                key=lambda patient: (
                    0
                    if smoking_records[patient].status == "smoker"
                    else 1 if smoking_records[patient].status == "ex-smoker" else 2
                ),
            )
        }
        for i, patient in enumerate(figures[plot_type]):
            row = i // n_cols
            col = i % n_cols
            axis[row, col].imshow(figures[plot_type][patient].canvas.buffer_rgba())
            axis[row, col].set_axis_off()
            if row != len(figures[plot_type]) // n_cols:
                axis[row, col].set_xticks([])
            if col != 0:
                axis[row, col].set_yticks([])

        fig.subplots_adjust(hspace=0.00, wspace=0.00, top=0.95, bottom=0.05)
        fig.tight_layout()
        fig.set_facecolor("white")

        # add a legend to the figure for the colours
        fig.legend(
            handles=[
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Smoker",
                    markerfacecolor="red",
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Ex-smoker",
                    markerfacecolor="green",
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Never-smoker",
                    markerfacecolor="blue",
                    markersize=10,
                ),
            ],
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, 0),
            prop={"size": 20},
        )

        fig.suptitle(
            "Mutational burden distributions for each patient", fontsize=20, y=0.99
        )
        fig.savefig(
            f"{kde_plot_folder}/{plot_type}_all{'_same_x' if same_x else ''}"
            f"{'_log_scale_x' if log_scale_x else ''}.png"
        )

        plt.close(fig)


if __name__ == "__main__":
    for same_x_ in [False, True]:
        for log_scale_x_ in [False, True]:
            plot_patient_kdes("logs/kde_plots", same_x_, log_scale_x_)
