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
# os.environ["PATH"] += ":/Users/hughselway/.juliaup/bin"

# %%
from multiprocessing import Pool

from ClonesModelling.parameters.hypothetical_paradigm_class import (
    get_hypothetical_paradigm_for_each_subset,
)
from notebooks.run_default_simulation import run_default_simulation

for spatial in [False, True]:
    print(f"Running {'spatial' if spatial else 'non-spatial'} simulations")
    output_dir = "notebooks/24-10-09-default-simulation/" + (
        "spatial/" if spatial else "non-spatial/"
    )
    os.makedirs(output_dir, exist_ok=True)
    hps = get_hypothetical_paradigm_for_each_subset(
        hypothesis_module_names=[
            "quiescent",
            "quiescent_protected",
            "protected",
            "immune_response",
            "smoking_driver",
        ],
        spatial=spatial,
        skipped_parameters=["mutation_rate_multiplier_shape"],
    )
    with Pool(4) as pool:
        pool.starmap(
            run_default_simulation,
            [
                (spatial, hp, output_dir, False, 10000, None, 0.05, None, 1)  # fcs
                for hp in hps
                if len(hp.get_module_names()) <= 2
            ],
        )
