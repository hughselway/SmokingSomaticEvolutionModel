import json
import os
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture  # type: ignore

from .mutations_data import get_total_mutations_data_per_patient


@dataclass
class MixtureModelData:
    larger_weight: float
    dominant_mean: float  # with weight larger_weight
    other_mean: float | None

    def __post_init__(self):
        assert 0.5 <= self.larger_weight <= 1, str(self)
        assert self.other_mean is None or self.dominant_mean != self.other_mean, str(
            self
        )

    @property
    def larger_mean(self) -> float:
        if self.other_mean is None:
            return self.dominant_mean
        return max(self.dominant_mean, self.other_mean)

    @property
    def larger_mean_weight(self) -> float:
        if self.other_mean is None:
            return 1
        return (
            self.larger_weight
            if self.dominant_mean > self.other_mean
            else 1 - self.larger_weight
        )

    @property
    def smaller_mean(self) -> float:
        if self.other_mean is None:
            return self.dominant_mean
        return min(self.dominant_mean, self.other_mean)

    @property
    def smaller_mean_weight(self) -> float:
        if self.other_mean is None:
            return 1
        return (
            1 - self.larger_weight
            if self.dominant_mean > self.other_mean
            else self.larger_weight
        )

    @property
    def only_mean(self) -> float:
        assert self.other_mean is None
        return self.dominant_mean

    @property
    def n_components(self) -> int:
        return 1 if self.other_mean is None else 2

    def write_to_string(self) -> str:
        return f"MM: {self.larger_weight} {self.dominant_mean} {self.other_mean}"


def read_mixture_model_from_string(s: str) -> MixtureModelData:
    assert s.startswith("MM: "), s
    try:
        larger_weight, dominant_mean, other_mean = map(float, s[len("MM: ") :].split())
    except ValueError:
        try:
            larger_weight, dominant_mean = map(float, s[len("MM: ") :].split()[:2])
            other_mean = None
        except ValueError as exc:
            raise ValueError(f"Invalid mixture model string: {s}") from exc
    return MixtureModelData(larger_weight, dominant_mean, other_mean)


def read_mixture_model(patient: str) -> MixtureModelData:
    if not os.path.exists("ClonesModelling/data/patient_data/mixture_models.json"):
        with open(
            "ClonesModelling/data/patient_data/mixture_models.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({}, f)
        raise KeyError()
    with open(
        "ClonesModelling/data/patient_data/mixture_models.json", "r", encoding="utf-8"
    ) as f:
        mixture_models = json.load(f)
    return MixtureModelData(**mixture_models[patient])


def fit_mixture_model(
    mutations_data: np.ndarray, log_transform: bool = False
) -> MixtureModelData:
    """
    Fit a 2-component Gaussian Mixture Model to the data, and return the larger weight
    and the means of the two components.
    """
    if mutations_data.ndim == 1:
        mutations_data = mutations_data.reshape(-1, 1)
    elif mutations_data.shape[1] == 2:
        # remove the smoking signature mutation count
        mutations_data = mutations_data[:, 0].reshape(-1, 1)
    if log_transform:
        mutations_data = np.log(mutations_data + 1)

    if len(set(mutations_data.flatten())) == 1:
        return MixtureModelData(1.0, mutations_data[0, 0], None)
    gmms = [GaussianMixture(n_components=n).fit(mutations_data) for n in [1, 2]]
    if gmms[0].bic(mutations_data) < gmms[1].bic(mutations_data):
        # 1-component model has lower BIC
        return MixtureModelData(1.0, gmms[0].means_[0][0], None)
    weights = gmms[1].weights_
    means = gmms[1].means_
    assert len(weights) == 2
    assert len(means) == 2
    return (
        MixtureModelData(weights[0], means[0][0], means[1][0])
        if weights[0] > weights[1]
        else MixtureModelData(weights[1], means[1][0], means[0][0])
    )


def get_patient_mixture_models() -> dict[str, MixtureModelData]:
    total_mutations_data = get_total_mutations_data_per_patient(False)
    patient_mixture_models: dict[str, MixtureModelData] = {}
    for patient, patient_mutations in total_mutations_data.items():
        try:
            patient_mixture_models[patient] = read_mixture_model(patient)
        except KeyError:
            patient_mixture_models[patient] = fit_mixture_model(patient_mutations)
            save_mixture_model(patient, patient_mixture_models[patient])
    return patient_mixture_models


def get_mixture_model(patient: str) -> MixtureModelData:
    try:
        return read_mixture_model(patient)
    except KeyError:
        patient_mutations = get_total_mutations_data_per_patient(False)[patient]
        mixture_model = fit_mixture_model(patient_mutations, log_transform=True)
        save_mixture_model(patient, mixture_model)
        return mixture_model


def save_mixture_model(patient: str, mixture_model: MixtureModelData) -> None:
    try:
        with open(
            "ClonesModelling/data/patient_data/mixture_models.json",
            "r",
            encoding="utf-8",
        ) as f:
            mixture_models = json.load(f)
    except FileNotFoundError:
        mixture_models = {}
    assert patient not in mixture_models
    mixture_models[patient] = mixture_model.__dict__
    with open(
        "ClonesModelling/data/patient_data/mixture_models.json", "w", encoding="utf-8"
    ) as f:
        json.dump(mixture_models, f)
