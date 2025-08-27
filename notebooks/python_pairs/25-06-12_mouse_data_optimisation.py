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
initial_state = [33, 12]  # 2.12]
temperature = 100.0
cooling_rate = 0.95
max_iterations = 200
repeat_count = 8
test = False  # use dummy optimisation function

# %%
import os
from multiprocessing import Pool
import json
from annealing import simulated_annealing
import random


def save_results_to_json(results_dir, optimal_states, optimal_values, records):
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(
            {
                "optimal_states": optimal_states,
                "optimal_values": optimal_values,
                "records": records,
            },
            f,
            indent=4,
        )


def read_json(results_dir):
    with open(f"{results_dir}/results.json", "r") as f:
        json_data = json.load(f)
    optimal_states = json_data["optimal_states"]
    optimal_values = json_data["optimal_values"]
    records = json_data["records"]
    return optimal_states, optimal_values, records


os.chdir("/Users/hughselway/Documents/ClonesModelling")

randomiser = random.Random(2025)

annealing_results_dir = "notebooks/data/25-06-12_mouse_data_optimisation/annealing/" + (
    "test_function" if test else f"mi_{max_iterations}_rc_{repeat_count}"
)
annealing_plot_dir = "notebooks/plots/25-06-12_mouse_data_optimisation/annealing/" + (
    "test_function" if test else f"mi_{max_iterations}_rc_{repeat_count}"
)

if not os.path.exists(annealing_results_dir + "/results.json") or test:
    with Pool(8) as pool:
        gs_results = pool.starmap(
            simulated_annealing,
            [
                (
                    initial_state,
                    temperature,
                    cooling_rate,
                    max_iterations,
                    randomiser.randint(0, 10**10),
                    test,
                )
                for _ in range(repeat_count)
            ],
        )
    optimal_states, optimal_values, records = zip(*gs_results)
    if not test:
        save_results_to_json(
            annealing_results_dir, optimal_states, optimal_values, records
        )
else:
    optimal_states, optimal_values, records = read_json(annealing_results_dir)

# %%
## plot optimal states (x = first element, y = second element) coloured by optimal value
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

fig, ax = plt.subplots(figsize=(4, 3))

scatter = ax.scatter(
    [state[0] for state in optimal_states],
    [state[1] for state in optimal_states],
    c=optimal_values,
    cmap="viridis_r",
    s=20,
    norm=Normalize(vmin=0),
)
cbar = fig.colorbar(scatter, ax=ax, label="Optimal distance from observation")
ax.set_xlabel("non_smoking_divisions_per_year")
ax.set_ylabel("annual_turnover")
ax.set_title("Optimal States from Simulated Annealing")
fig.tight_layout()
display(fig)
os.makedirs(annealing_plot_dir, exist_ok=True)
plt.savefig(f"{annealing_plot_dir}/optimal_states.pdf")
plt.close(fig)

# %%
## for each record, lineplot the states over iterations
fig, ax = plt.subplots(figsize=(6, 6))

for i, record in enumerate(records):
    ax.plot(
        [state[1][0] for state in record],
        [state[1][1] for state in record],
        alpha=0.3,
    )
    # scatter of first and last state
    ax.scatter(
        record[0][1][0],
        record[0][1][1],
        color="blue",
        s=10,
    )
    ax.scatter(
        record[-1][1][0],
        record[-1][1][1],
        color="red",
        s=10,
    )
ax.set_xlabel("non_smoking_divisions_per_year")
ax.set_ylabel("annual_turnover")
ax.set_title("Simulated Annealing Path for Each Run")
display(fig)
fig.savefig(f"{annealing_plot_dir}/annealing_paths.pdf")
plt.close(fig)

# %%
## plot value over iterations
fig, ax = plt.subplots(figsize=(6, 3))
for i, record in enumerate(records):
    ax.plot(
        [state[2] for state in record],
        alpha=0.3,
    )
ax.set_xlabel("Iteration")
ax.set_ylabel("Distance from observation")
fig.tight_layout()
display(fig)
fig.savefig(f"{annealing_plot_dir}/annealing_values.pdf")
plt.close(fig)

# %%
# plot jump size over iterations
window_size = 20
fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
for i, record in enumerate(records):
    jump_sizes = [
        (
            (record[j][1][0] - record[j - 1][1][0]) ** 2
            + (record[j][1][1] - record[j - 1][1][1]) ** 2
        )
        ** 0.5
        for j in range(1, len(record))
    ]
    # rolling average
    rolling_average_jump_sizes = [
        sum(jump_sizes[max(0, j - window_size) : j + 1])
        / (j - max(0, j - window_size) + 1)
        for j in range(len(jump_sizes))
    ]
    axes[0].plot(rolling_average_jump_sizes, alpha=0.3, label=f"Run {i + 1}")

    # plot rolling fraction of non-zero jump sizes
    non_zero_jump_sizes = [size for size in jump_sizes if size > 0]
    rolling_fraction_non_zero = [
        sum(1 for size in jump_sizes[max(0, j - window_size) : j + 1] if size > 0)
        / (j - max(0, j - window_size) + 1)
        for j in range(len(jump_sizes))
    ]
    axes[1].plot(rolling_fraction_non_zero, alpha=0.3, label=f"Run {i + 1}")
axes[1].set_xlabel("Iteration")
axes[0].set_ylabel(f"Jump Size (Rolling\nAverage over {window_size})")
axes[1].set_ylabel("Fraction of Non-Zero\nJump Sizes")
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
display(fig)
fig.tight_layout()
fig.savefig(f"{annealing_plot_dir}/annealing_jump_sizes.pdf")
plt.close(fig)

# %%
# grid search parameters
non_smoking_divisions_per_year_bounds = (20, 50)
annual_turnover_bounds = (1.5, 6)
non_smoking_divisions_per_year_value_count = 10
annual_turnover_value_count = 10
test=False

# %%
# quick grid search for comparison
import numpy as np


os.chdir("/Users/hughselway/Documents/ClonesModelling/notebooks")
from annealing import get_mouse_simulation_distance, get_dummy_loss

os.chdir("/Users/hughselway/Documents/ClonesModelling")


def grid_search(
    non_smoking_divisions_per_year,
    annual_turnover,
    test=test,
):
    results = np.zeros((len(non_smoking_divisions_per_year), len(annual_turnover)))
    with Pool(8) as pool:
        results = pool.starmap(
            get_mouse_simulation_distance if not test else get_dummy_loss,
            [
                (n, a, randomiser.randint(0, 10**10))
                for n in non_smoking_divisions_per_year
                for a in annual_turnover
            ],
        )
        results = np.array(results).reshape(
            len(non_smoking_divisions_per_year), len(annual_turnover)
        )
    best_value = np.min(results)
    best_state_index = np.unravel_index(np.argmin(results), results.shape)
    best_state = [
        non_smoking_divisions_per_year[best_state_index[0]],
        annual_turnover[best_state_index[1]],
    ]
    return best_state, best_value, results


# save grid search results to npz
def save_grid_search_results(grid_search_results_dir, best_state, best_value, results):
    os.makedirs(grid_search_results_dir, exist_ok=True)
    np.savez(
        f"{grid_search_results_dir}/grid_search_results.npz",
        best_state=best_state,
        best_value=best_value,
        results=results,
    )


def read_grid_search_results(grid_search_results_dir):
    with np.load(f"{grid_search_results_dir}/grid_search_results.npz") as data:
        best_state = data["best_state"]
        best_value = data["best_value"]
        results = data["results"]
    return best_state, best_value, results


non_smoking_divisions_per_year = np.linspace(
    non_smoking_divisions_per_year_bounds[0],
    non_smoking_divisions_per_year_bounds[1],
    non_smoking_divisions_per_year_value_count,
)
annual_turnover = np.linspace(
    annual_turnover_bounds[0],
    annual_turnover_bounds[1],
    annual_turnover_value_count,
)

grid_search_results_dir = "notebooks/data/25-06-12_mouse_data_optimisation/grid_search/" + (
    "test_function"
    if test
    else f"nsdpy_{non_smoking_divisions_per_year_value_count}_at_{annual_turnover_value_count}"
)
grid_search_plot_dir = "notebooks/plots/25-06-12_mouse_data_optimisation/grid_search/" + (
    "test_function"
    if test
    else f"nsdpy_{non_smoking_divisions_per_year_value_count}_at_{annual_turnover_value_count}"
)
if not os.path.exists(grid_search_results_dir + "/grid_search_results.npz"):
    gs_best_state, gs_best_value, gs_results = grid_search(
        non_smoking_divisions_per_year,
        annual_turnover,
        test=test,
    )
    save_grid_search_results(
        grid_search_results_dir, gs_best_state, gs_best_value, gs_results
    )
else:
    gs_best_state, gs_best_value, gs_results = read_grid_search_results(
        grid_search_results_dir
    )

# %%
# heatmap of grid search results
import seaborn as sns

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    # np.exp(-(gs_results.T - (np.sum(gs_results) / np.prod(gs_results.shape)))),
    -gs_results.T,
    cmap="viridis",
    ax=ax,
    xticklabels=np.round(non_smoking_divisions_per_year, 1),
    yticklabels=np.round(annual_turnover, 1),
    cbar_kws={"label": "Unnormalised loglikelihood"},
    # vmin=0,
)
ax.set_xlabel("Non Smoking Divisions per Year")
ax.set_ylabel("Annual Turnover")
ax.set_title("Grid Search Results for Mouse Data Optimisation")
ax.invert_yaxis()  # Flip the y-axis
# add star on optimal point
optimal_point = gs_best_state
print(gs_best_state)
ax.scatter(
    np.argmin(np.abs(non_smoking_divisions_per_year - optimal_point[0])) + 0.5,
    np.argmin(np.abs(annual_turnover - optimal_point[1])) + 0.5,
    color="red",
    s=100,
    marker="*",
    label="Optimal Point",
)

# optimal_x = np.argmin(gs_results, axis=1)
# optimal_y = np.argmin(gs_results, axis=0)
# ax.scatter(
#     optimal_y,
#     optimal_x,
#     color="red",
#     s=100,
#     marker="*",
#     label="Optimal Point",
# )
fig.tight_layout()
display(fig)
os.makedirs(grid_search_plot_dir, exist_ok=True)
fig.savefig(
    f"{grid_search_plot_dir}/grid_search_results.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# %%
# run new mouse simulation with optimal parameters, from grid search and annealing
os.chdir("/Users/hughselway/Documents/ClonesModelling/notebooks")
from annealing import get_mouse_simulation_distance

os.chdir("/Users/hughselway/Documents/ClonesModelling")

replicate_count = 8
simulations_run = False

annealing_best_state = optimal_states[optimal_values.index(min(optimal_values))]
with Pool(8) as pool:
    # print(f"Annealing Best State: {annealing_best_state}, Value: {min(optimal_values)}")
    # annealing_results = pool.starmap(
    #     get_mouse_simulation_distance,
    #     [
    #         (
    #             annealing_best_state[0],
    #             annealing_best_state[1],
    #             randomiser.randint(0, 10**10),
    #             f"{annealing_results_dir}/optimal_simulation_clone_sizes/r_{ix}.csv",
    #         )
    #         for ix in range(replicate_count)
    #     ],
    # )

    print(f"Grid Search Best State: {gs_best_state}, Value: {gs_best_value}")
    grid_search_results = pool.starmap(
        get_mouse_simulation_distance,
        [
            (
                gs_best_state[0],
                gs_best_state[1],
                randomiser.randint(0, 10**10),
                f"{grid_search_results_dir}/optimal_simulation_clone_sizes/r_{ix}.csv",
            )
            for ix in range(replicate_count)
        ],
    )

fig, ax = plt.subplots(figsize=(6, 4))

sns.stripplot(
    x=["Grid Search"] * replicate_count,# + ["Annealing"] * replicate_count,
    y=grid_search_results, # + annealing_results,
    jitter=True,
    color="blue",
)
ax.set_xlabel("Optimisation Method")
ax.set_ylabel("Distance from Observation at Optimal Parameters")
# ax.set_ylim(0, None)

# %%
grid_search_results_dir
