import pandas as pd  # type: ignore

from .mutations_data import (
    get_clones_data,
    get_nature_genetics_data,
    get_total_mutations_data_per_patient,
)
from .smoking_record_class import (
    SmokerRecord,
    NonSmokerRecord,
    ExSmokerRecord,
)


def get_smoking_records(
    include_infants: bool = True,
    first_patient_test: bool = False,
    exclude_nature_genetics: bool = False,
) -> list[SmokerRecord | NonSmokerRecord | ExSmokerRecord]:
    return (
        get_clones_smoking_records(include_infants, first_patient_test)
        if exclude_nature_genetics or first_patient_test
        else (
            get_clones_smoking_records(include_infants, first_patient_test)
            + get_nature_genetics_smoking_records()
        )
    )


def get_smoking_record(patient) -> SmokerRecord | NonSmokerRecord | ExSmokerRecord:
    for smoking_record in get_smoking_records():
        if smoking_record.patient == patient:
            return smoking_record
    raise ValueError(f"No smoking record matches patient ID {patient}")


def get_clones_smoking_records(
    include_infants: bool, first_patient_test: bool
) -> list[SmokerRecord | NonSmokerRecord | ExSmokerRecord]:
    return list(
        get_clones_data()
        .drop_duplicates(subset=["patient"])
        .loc[lambda x: x["age"] > (5 if not include_infants else 0)]
        .apply(extract_smoking_record, axis=1)
    )[: (1 if first_patient_test else None)]


def get_nature_genetics_smoking_records() -> (
    list[SmokerRecord | NonSmokerRecord | ExSmokerRecord]
):
    return list(
        get_nature_genetics_data()
        .drop_duplicates(subset="patient")
        .apply(extract_smoking_record, axis=1)
    )


def extract_smoking_record(
    clones_data_row: pd.Series,
) -> SmokerRecord | NonSmokerRecord | ExSmokerRecord:
    assert clones_data_row["smoking_status"] in [
        "smoker",
        "non-smoker",
        "ex-smoker",
    ], f"Unexpected smoking status {clones_data_row['smoking_status']}"
    return (
        NonSmokerRecord(patient=clones_data_row["patient"], age=clones_data_row["age"])
        if clones_data_row["smoking_status"] == "non-smoker"
        else (
            SmokerRecord(
                patient=clones_data_row["patient"],
                age=clones_data_row["age"],
                start_smoking_age=clones_data_row["start_smoking_age"],
                pack_years=clones_data_row["pack_years"],
            )
            if clones_data_row["smoking_status"] == "smoker"
            else ExSmokerRecord(
                patient=clones_data_row["patient"],
                age=clones_data_row["age"],
                start_smoking_age=clones_data_row["start_smoking_age"],
                stop_smoking_age=clones_data_row["stop_smoking_age"],
                pack_years=clones_data_row["pack_years"],
            )
        )
    )


def update_smoking_records_csv() -> None:
    smoking_records = get_smoking_records()
    smoking_records_df = pd.DataFrame(
        [
            {
                "patient": record.patient,
                "age": record.age,
                "smoking_status": record.status,
                "start_smoking_age": record.start_smoking_age,
                "stop_smoking_age": getattr(record, "stop_smoking_age", None),
                "pack_years": getattr(record, "pack_years", None),
            }
            for record in smoking_records
        ]
    )
    # add column for if it's in nature_genetics
    smoking_records_df["nature_genetics"] = smoking_records_df["patient"].isin(
        get_nature_genetics_data()["patient"]
    )
    smoking_records_df["n_cells"] = [
        len(get_total_mutations_data_per_patient(False)[patient])
        for patient in smoking_records_df["patient"]
    ]
    smoking_records_df.to_csv(
        "ClonesModelling/data/patient_data/smoking_records.csv", index=False
    )
