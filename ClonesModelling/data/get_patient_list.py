from .mutations_data import get_clones_data, get_nature_genetics_data


def get_patient_list(
    include_infants: bool = True,
    first_patient_test: bool = False,
    exclude_nature_genetics: bool = False,
) -> list[str]:
    return (
        get_clones_patient_list(include_infants, first_patient_test)
        if exclude_nature_genetics or first_patient_test
        else get_clones_patient_list(include_infants, first_patient_test)
        + get_nature_genetics_patient_list()
    )


def get_clones_patient_list(
    include_infants: bool = True, first_patient_test: bool = False
) -> list[str]:
    return list(
        get_clones_data()
        .drop_duplicates(subset=["patient"])
        .loc[lambda x: x["age"] > (5 if not include_infants else 0)]
        .loc[:, "patient"]
    )[: (1 if first_patient_test else None)]


def get_nature_genetics_patient_list() -> list[str]:
    return list(get_nature_genetics_data().drop_duplicates(subset="patient")["patient"])
