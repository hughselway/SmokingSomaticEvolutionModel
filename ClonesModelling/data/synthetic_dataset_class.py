from dataclasses import dataclass
import numpy as np

from .mixture_model_data import MixtureModelData


@dataclass
class PatientSyntheticData:
    mutational_burden: np.ndarray
    patient_mixture_models: MixtureModelData
    phylogeny_branch_lengths: np.ndarray | None
    tree_balance: float | None


class SyntheticDataset:
    patient_synthetic_data: dict[str, PatientSyntheticData]

    def __init__(
        self,
        patient_mb: dict[str, np.ndarray],
        patient_mixture_models: dict[str, MixtureModelData],
        patient_bl: dict[str, np.ndarray] | None,
        patient_tree_balance: dict[str, float] | None,
    ) -> None:
        assert set(patient_mb.keys()) == set(patient_mixture_models.keys())
        if patient_bl is not None:
            assert patient_tree_balance is not None
            assert set(patient_tree_balance.keys()).issubset(set(patient_bl.keys()))
            assert set(patient_bl.keys()).issubset(patient_mb.keys())
            self.patient_synthetic_data = {
                patient: PatientSyntheticData(
                    patient_mb[patient],
                    patient_mixture_models[patient],
                    patient_bl[patient] if patient in patient_bl else None,
                    (
                        patient_tree_balance[patient]
                        if patient in patient_tree_balance
                        else None
                    ),
                )
                for patient in patient_mb
            }
        else:
            self.patient_synthetic_data = {
                patient: PatientSyntheticData(
                    patient_mb[patient], patient_mixture_models[patient], None, None
                )
                for patient in patient_mb
            }

    def __getitem__(self, patient: str) -> PatientSyntheticData:
        return self.patient_synthetic_data[patient]

    def __iter__(self):
        return iter(self.patient_synthetic_data)

    @property
    def patient_mutational_burden(self) -> dict[str, np.ndarray]:
        return {patient: self[patient].mutational_burden for patient in self}

    @property
    def patient_mixture_models(self) -> dict[str, MixtureModelData]:
        return {patient: self[patient].patient_mixture_models for patient in self}

    @property
    def patient_phylogeny_branch_lengths(self) -> dict[str, np.ndarray]:
        return {
            patient: self.patient_synthetic_data[
                patient
            ].phylogeny_branch_lengths  # type: ignore
            for patient in self
            if self.patient_synthetic_data[patient].phylogeny_branch_lengths is not None
        }

    @property
    def patient_tree_balance_indices(self) -> dict[str, float]:
        return {
            patient: self.patient_synthetic_data[patient].tree_balance  # type: ignore
            for patient in self
            if self.patient_synthetic_data[patient].tree_balance is not None
        }
