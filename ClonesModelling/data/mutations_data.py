import numpy as np
import pandas as pd  # type: ignore


def get_clones_data() -> pd.DataFrame:
    return (
        pd.read_csv(
            "ClonesModelling/data/patient_data/"
            "Lung_organoids_telomeres_with_contamination_20190628.txt",
            sep="\t",
        )
        .merge(
            pd.read_excel(
                "ClonesModelling/data/patient_data/yoshida_supplementary_tables.xlsx",
                sheet_name="Sup_Table1",
                header=1,
            )
            .rename({"ID": "Patient", "Sex": "gender"}, axis=1)
            .loc[
                lambda x: ~x["Patient"].str.contains("Abbreviations"),
                ["Patient", "gender", "Duration of smoking cessation (year)"],
            ],
            on="Patient",
            how="left",
        )
        .rename(
            columns={
                "Patient": "patient",
                "Age": "age",
                "Smoking": "smoking_status",
                "PackYear": "pack_years",
                "n_drivers": "number_of_drivers",
            }
        )
        .merge(
            pd.read_csv(
                "ClonesModelling/data/patient_data/smoking_signature_mutations.csv"
            )
            .rename(columns={"Samples": "Sample"})
            .loc[:, ["Sample", "smoking_signature_mutations"]],
            on="Sample",
            how="left",
        )
        .assign(
            total_mutations=lambda x: x["n_muts"] + x["n_indels"],
            years_since_quitting=(
                lambda x: x["Duration of smoking cessation (year)"].apply(
                    lambda x: (
                        x
                        if (isinstance(x, int))
                        else (
                            float(x.split(" ")[0]) / 12
                            if isinstance(x, str)
                            else np.nan
                        )
                    )  # months
                )
            ),
            start_smoking_age=lambda x: np.where(
                x["smoking_status"] == "ex-smoker",
                x["age"] - x["years_since_quitting"] - x["Duration"],
                np.where(
                    x["smoking_status"] == "smoker",
                    x["age"] - x["Duration"],
                    np.nan,
                ),
            ),
            stop_smoking_age=lambda x: np.where(
                x["smoking_status"] == "ex-smoker",
                x["age"] - x["years_since_quitting"],
                np.nan,
            ),
        )
        .loc[
            :,
            [
                "patient",
                "age",
                "smoking_status",
                "gender",
                "pack_years",
                "total_mutations",
                "number_of_drivers",
                "smoking_signature_mutations",
                "start_smoking_age",
                "stop_smoking_age",
            ],
        ]
    )


def get_nature_genetics_data() -> pd.DataFrame:
    sensitivity_threshold = 0.5
    return (
        pd.read_excel(
            "ClonesModelling/data/patient_data/nature_genetics_data.xlsx",
            sheet_name="Sup_Table2",
            header=1,
        )
        .merge(
            pd.read_excel(
                "ClonesModelling/data/patient_data/nature_genetics_data.xlsx",
                sheet_name="Sup_Table5",  # driver genes per cell
                header=1,
            )
            .groupby("Cell")
            .count()["Region"]
            .rename("number_of_drivers"),
            left_on="Cell ID",
            right_on="Cell",
            how="left",
        )
        .merge(
            pd.read_csv(
                "ClonesModelling/data/patient_data/smoking_signature_mutations.csv"
            )
            .rename(columns={"Samples": "Cell ID"})
            .loc[
                :,
                [
                    "Cell ID",
                    "smoking_signature_substitutions",
                    "smoking_signature_indels",
                ],
            ],
            on="Cell ID",
            how="left",
        )
        .rename(
            columns={
                "Smoking_pack_year": "pack_years",
                "Age": "age",
                "Quit": "time_since_quitting",
                "Gender": "gender",
                "SNV_sen": "sensitivity",
            }
        )
        .fillna({"number_of_drivers": 0})
        .assign(
            patient=lambda x: x["Subject ID"].astype(str),
            total_mutations=lambda x: x["SNV_perCell"] + x["INDEL_perCell"],
            snv_sensitivity_correction_multiplier=lambda x: (
                x["SNV_perCell"] / x["SNV_raw"]
            ),
            indel_sensitivity_correction_multiplier=lambda x: np.where(
                x["INDEL_raw"] > 0, x["INDEL_perCell"] / x["INDEL_raw"], 0
            ),
            smoking_signature_mutations=lambda x: (
                x["smoking_signature_substitutions"]
                * x["snv_sensitivity_correction_multiplier"]
                + x["smoking_signature_indels"]
                * x["indel_sensitivity_correction_multiplier"]
            ),
            smoking_status=lambda x: x["time_since_quitting"].apply(
                lambda x: (
                    "smoker" if x == 0.0 else ("ex-smoker" if x > 0.0 else "non-smoker")
                )
            ),
            start_smoking_age=(
                lambda x: x["age"] - x["time_since_quitting"] - x["SmokingYears"]
            ),
            stop_smoking_age=lambda x: x["age"] - x["time_since_quitting"],
        )
        .loc[
            lambda x: x["sensitivity"] > sensitivity_threshold,
            [
                "patient",
                "age",
                "smoking_status",
                "gender",
                "pack_years",
                "total_mutations",
                "number_of_drivers",
                "smoking_signature_mutations",
                "start_smoking_age",
                "stop_smoking_age",
                "sensitivity",
            ],
        ]
    )


def get_sanger_data() -> pd.DataFrame:
    return (
        pd.read_csv(
            "ClonesModelling/data/patient_data/"
            "COPD_per_clone_sensitivity_adjusted_block.csv"
        )
        .rename(columns={"tree_mut_count": "total_mutations"})
        .assign(
            patient=lambda x: x["patient_ID"].apply(lambda x: x.split("_")[0]),
            block=lambda x: x["patient_ID"].apply(lambda x: x.split("_")[1]),
        )
        .loc[lambda x: x["clone_detection_sensitivity"] > 0.75]
    )


def get_single_cell_datasets() -> list[pd.DataFrame]:
    return [get_clones_data(), get_nature_genetics_data()]


def get_patient_data(patient: str) -> pd.DataFrame:
    patient_found = False
    patient_data: pd.DataFrame | None = None
    for data in get_single_cell_datasets():
        if any(data["patient"] == patient):
            if patient_found:
                raise ValueError(f"Patient {patient} found in two datasets!")
            patient_found = True
            patient_data = data[data["patient"] == patient]
    if not patient_found:
        raise ValueError(f"Patient {patient} not found")
    assert patient_data is not None
    return patient_data


def get_total_mutations_data_per_patient(
    include_smoking_signatures: bool,
) -> dict[str, np.ndarray]:
    datasets = get_single_cell_datasets()
    patients: list[str] = sum((list(dataset["patient"]) for dataset in datasets), [])
    total_mutations_data: dict[str, np.ndarray] = {}
    for dataset in datasets:
        for patient in dataset["patient"].unique():
            patient_data = dataset[dataset["patient"] == patient][
                (
                    ["total_mutations", "smoking_signature_mutations"]
                    if include_smoking_signatures
                    else "total_mutations"
                )
            ].to_numpy()
            if patient in total_mutations_data:
                raise ValueError(f"Patient {patient} found in two datasets!")
            total_mutations_data[patient] = patient_data
    assert all(len(total_mutations_data[patient]) > 0 for patient in patients), (
        total_mutations_data,
        patients,
    )
    return total_mutations_data
