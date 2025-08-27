from dataclasses import dataclass
import json
from typing import Any


PENALTY_ABC_VALUE = 1e12


@dataclass
class SubsamplePercentileRecord:
    fifth: float
    median: float
    ninety_fifth: float


@dataclass
class DistanceFunctionResult:
    weighted_sum: float | None
    patient_subsample_percentiles: dict[str, list[SubsamplePercentileRecord | None]]

    def json_dict(self) -> dict[str, Any]:
        return {
            "weighted_sum": self.weighted_sum,
            "patient_subsample_percentiles": {
                patient: [
                    x.__dict__ if x is not None else None
                    for x in self.patient_subsample_percentiles[patient]
                ]
                for patient in self.patient_subsample_percentiles
            },
        }

    @property
    def patients(self) -> list[str]:
        return list(self.patient_subsample_percentiles.keys())

    @property
    def replicate_count(self) -> int:
        return len(self.patient_subsample_percentiles[self.patients[0]])

    def fifth_percentiles(self, patient: str) -> list[float]:
        return [x.fifth for x in self.patient_subsample_percentiles[patient] if x]

    def medians(self, patient: str) -> list[float]:
        return [x.median for x in self.patient_subsample_percentiles[patient] if x]

    def ninety_fifth_percentiles(self, patient: str) -> list[float]:
        return [
            x.ninety_fifth for x in self.patient_subsample_percentiles[patient] if x
        ]


def read_df_result_from_json_dict(json_dict: dict[str, Any]) -> DistanceFunctionResult:
    return DistanceFunctionResult(
        json_dict["weighted_sum"],
        {
            patient: [
                (
                    SubsamplePercentileRecord(
                        x["fifth"], x["median"], x["ninety_fifth"]
                    )
                    if x is not None
                    else None
                )
                for x in json_dict["patient_subsample_percentiles"][patient]
            ]
            for patient in json_dict["patient_subsample_percentiles"]
        },
    )


@dataclass
class DistanceResult:
    wasserstein_mb: DistanceFunctionResult
    smoking_sig_mb: DistanceFunctionResult
    mixture_model_weight: DistanceFunctionResult
    mixture_model_larger_mean: DistanceFunctionResult
    mixture_model_smaller_mean: DistanceFunctionResult
    branch_length_wasserstein: DistanceFunctionResult
    tree_balance: DistanceFunctionResult

    def items(self) -> list[tuple[str, DistanceFunctionResult]]:
        return [
            ("wasserstein_mb", self.wasserstein_mb),
            ("smoking_sig_mb", self.smoking_sig_mb),
            ("mixture_model_weight", self.mixture_model_weight),
            ("mixture_model_larger_mean", self.mixture_model_larger_mean),
            ("mixture_model_smaller_mean", self.mixture_model_smaller_mean),
            ("branch_length_wasserstein", self.branch_length_wasserstein),
            ("tree_balance", self.tree_balance),
        ]

    def keys(self) -> list[str]:
        return [df_name for df_name, _ in self.items()]

    def json_dict(self) -> dict[str, Any]:
        return {df_name: df_result.json_dict() for df_name, df_result in self.items()}

    def get_abc_registration_dict(self) -> dict[str, float]:
        return {
            df_name: (df_result.weighted_sum or 20.0)
            for df_name, df_result in self.items()
        }


def get_empty_distance_result(distance_value: float | None = None) -> DistanceResult:
    return DistanceResult(
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
        DistanceFunctionResult(distance_value, {}),
    )


def write_distances_to_json(
    distances: dict[str, dict[str, DistanceResult | None]], json_file: str
) -> None:
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                paradigm_name: (
                    {
                        sample_name: (
                            distance.json_dict() if distance is not None else None
                        )
                        for sample_name, distance in distance_by_sample.items()
                    }
                )
                for paradigm_name, distance_by_sample in distances.items()
            },
            file,
            indent=4,
        )


def read_distance_result_from_json_dict(json_dict: dict[str, Any]) -> DistanceResult:
    df_names = get_empty_distance_result().keys()
    assert sorted(json_dict.keys()) == sorted(df_names)
    return DistanceResult(
        *(read_df_result_from_json_dict(json_dict[df_name]) for df_name in df_names)
    )


def read_distances_from_json(
    json_file: str,
) -> dict[str, dict[str, DistanceResult | None]]:
    with open(json_file, "r", encoding="utf-8") as file:
        json_dict = json.load(file)
    return {
        paradigm_name: {
            sample_name: (
                read_distance_result_from_json_dict(distance)
                if distance is not None
                else None
            )
            for sample_name, distance in distance_by_sample.items()
        }
        for paradigm_name, distance_by_sample in json_dict.items()
    }


def get_zero_registration_dict() -> dict[str, float]:
    return get_empty_distance_result(distance_value=0.0).get_abc_registration_dict()


def get_penalty_registration_dict() -> dict[str, float]:
    return get_empty_distance_result(
        distance_value=PENALTY_ABC_VALUE
    ).get_abc_registration_dict()
