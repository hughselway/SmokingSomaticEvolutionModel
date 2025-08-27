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
import numpy as np

smoking_records_df = (
    pd.read_csv("ClonesModelling/data/patient_data/smoking_records.csv")
    # .assign(weight=lambda x: np.log(x["n_cells"]))
    .assign(weight=lambda x: np.log(1 + x["n_cells"]))
    # .set_index("patient")[["weight"]]
    # .to_dict()["weight"]
)
pt_weights = smoking_records_df.set_index("patient")[["weight"]].to_dict()["weight"]
pt_statuses = smoking_records_df.set_index("patient")[["smoking_status"]].to_dict()[
    "smoking_status"
]
smoking_records_df

# %%
# idt_id = "idt_2024-10-03_19-04-47"
# idt_id = "idt_2025-02-05_14-53-34"
# idt_id = "idt_2025-04-04_14-30-05"
idt_id = "idt_2025-04-07_10-16-59"

for df_name in os.listdir(f"logs/{idt_id}/distance"):
    if not os.path.isdir(f"logs/{idt_id}/distance/{df_name}") or df_name == "hpc":
        continue
    print(df_name, end=" ")
    # if os.path.exists(f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled_new.csv"):
    #     print("Already recalculated")
    #     continue
    if os.path.exists(
        f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled.csv"
    ):
        continue
        # assert not os.path.exists(
        #     f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled_old.csv"
        # )
        # os.rename(
        #     f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled.csv",
        #     f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled_old.csv",
        # )

    # start_time = pd.Timestamp.now()
    # if not os.path.exists(
    #     f"logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv"
    # ):
    #     exit_code = os.system(
    #         f"tar -xzf logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv.tar.gz"
    #     )
    #     if exit_code != 0:
    #         print("\nFailed to untar all_patients")
    #     after_unzip_time = pd.Timestamp.now()
    #     print(f"; unzipped in {after_unzip_time - start_time}; ", end="")
    # else:
    #     after_unzip_time = start_time
    start_time = pd.Timestamp.now()
    after_unzip_time = pd.Timestamp.now()

    is_first_chunk = True
    for chunk in pd.read_csv(
        f"logs/{idt_id}/distance/{df_name}/pairwise_distances.csv",
        chunksize=1_000_000,
    ):
        patients: list[str] = chunk.columns[5:]
        assert set(patients) == set(pt_weights.keys())
        augmented_chunk = (
            chunk.assign(
                **{f"{pt}_weighted": chunk[pt] * pt_weights[pt] for pt in patients},
                **{
                    f"{name}_distance": lambda x, is_included_=is_included: x[
                        [
                            f"{pt}_weighted"
                            for pt in patients
                            if is_included_(pt, pt_statuses)
                        ]
                    ].sum(axis=1)
                    for name, is_included in [
                        ("nature_patients", lambda pt, _: pt.startswith("PD")),
                        (
                            "nature_genetics_patients",
                            lambda pt, _=pt_statuses: not pt.startswith("PD"),
                        ),
                        (
                            "status_representatives",
                            lambda pt, _=pt_statuses: pt
                            in ["PD26988", "PD34204", "PD34209"],
                        ),
                        (
                            "smokers_only",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            == "smoker",
                        ),
                        (
                            "non_smokers_only",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            == "non-smoker",
                        ),
                        (
                            "ex_smokers_only",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            == "ex-smoker",
                        ),
                        (
                            "without_smokers",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            != "smoker",
                        ),
                        (
                            "without_non_smokers",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            != "non-smoker",
                        ),
                        (
                            "without_ex_smokers",
                            lambda pt, pt_statuses_=pt_statuses: pt_statuses_[pt]
                            != "ex-smoker",
                        ),
                        ("total", lambda _, __: True),
                    ]
                },
            )
            .drop(columns=[f"{pt}_weighted" for pt in patients])
            .drop(columns=patients)
            .drop(columns=["distance"])
        )
        # print(augmented_chunk)
        augmented_chunk.to_csv(
            f"logs/{idt_id}/distance/{df_name}/pairwise_distances_subsampled.csv",
            mode="a",
            index=False,
            header=is_first_chunk,
        )
        is_first_chunk = False
        print(".", end="")

    after_augmentation_time = pd.Timestamp.now()
    print(f"; augmented in {after_augmentation_time - after_unzip_time}; ", end="")

    os.remove(f"logs/{idt_id}/distance/{df_name}/pairwise_distances.csv")
    print(f"removed in {pd.Timestamp.now() - after_augmentation_time}")

# %%
# tar the pairwise_distances_all_patients.csv files
for df_name in os.listdir(f"logs/{idt_id}/distance"):
    if not os.path.isdir(f"logs/{idt_id}/distance/{df_name}") or df_name == "hpc":
        continue
    print(df_name, end=" ")
    previous_size = os.path.getsize(
        f"logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv"
    )
    exit_code = os.system(
        f"tar -czf logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv.tar.gz logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv"
    )
    if exit_code != 0:
        print("Failed to tar")
        continue
    new_size = os.path.getsize(
        f"logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv.tar.gz"
    )
    print(f"{previous_size} -> {new_size}")
    os.remove(f"logs/{idt_id}/distance/{df_name}/pairwise_distances_all_patients.csv")
    print()
