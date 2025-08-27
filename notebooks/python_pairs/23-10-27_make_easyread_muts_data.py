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
from ClonesModelling.data.mutations_data import get_single_cell_datasets

single_cell_datasets = get_single_cell_datasets()
with open(
    "ClonesModelling/data/patient_data/cell_mutation_counts_yoshida_huang.csv",
    "w",
    encoding="utf-8",
) as lmer_data_file:
    lmer_data_file.write(
        "patient,age,pack_years,smoking_status,total_mutations,years_smoked,"
        "years_since_smoking,gender,dataset,sensitivity\n"
    )
    for dataset, dataset_name in zip(single_cell_datasets, ["Yoshida", "Huang"]):
        for _, row in dataset.iterrows():
            years_smoked = (
                0
                if (row["start_smoking_age"] != row["start_smoking_age"])  # non-smoker
                else row["age"] - row["start_smoking_age"]
                if row["stop_smoking_age"] != row["stop_smoking_age"]  # smoker
                else row["stop_smoking_age"] - row["start_smoking_age"]  # ex-smoker
                # row["stop_smoking_age"] - row["start_smoking_age"]
                # if not (
                #     row["stop_smoking_age"] != row["stop_smoking_age"]
                #     and row["start_smoking_age"] != row["start_smoking_age"]
                # )
                # else 0
            )
            years_since_smoking = (
                0
                if row["stop_smoking_age"] != row["stop_smoking_age"]
                else row["age"] - row["stop_smoking_age"]
            )
            lmer_data_file.write(
                f"{row['patient']},{row['age']},{row['pack_years']},"
                f"{row['smoking_status']},{row['total_mutations']},{years_smoked},"
                f"{years_since_smoking},{row['gender']},{dataset_name},"
                f"{None if 'sensitivity' not in row else row['sensitivity']}\n"
            )
