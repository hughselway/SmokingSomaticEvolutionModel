from dataclasses import dataclass
import json
import os
import shutil
from typing import Generator
from Bio import Phylo
import numpy as np
import pandas as pd  # type: ignore

from .abc_distance_result import DistanceResult, read_distances_from_json
from .smoking_record_class import ExSmokerRecord, NonSmokerRecord, SmokerRecord


@dataclass
class ABCPosteriorData:
    paradigm_names: list[str]
    samples_per_paradigm: int
    replicates_per_sample: int
    posterior_simulations_directory: str
    record_frequency: int
    distance_results: dict[str, dict[str, DistanceResult | None]]

    def __init__(
        self,
        logging_directory: str,
        run_id: str,
        record_frequency: int = 3,
        paradigm_names: list[str] | None = None,
        samples_per_paradigm: int | None = None,
        replicates_per_sample: int | None = None,
        distance_results: dict[str, dict[str, DistanceResult | None]] | None = None,
    ) -> None:
        self.posterior_simulations_directory = (
            f"{logging_directory}/{run_id}/posterior_simulations"
        )
        self.record_frequency = record_frequency
        self.paradigm_names = paradigm_names or [
            x
            for x in os.listdir(self.posterior_simulations_directory)
            if os.path.isdir(f"{self.posterior_simulations_directory}/{x}")
        ]
        samples_directory = (
            f"{self.posterior_simulations_directory}/{self.paradigm_names[0]}/"
            "posterior_samples"
        )
        self.samples_per_paradigm = samples_per_paradigm or len(
            os.listdir(samples_directory)
        )
        self.replicates_per_sample = replicates_per_sample or len(
            [
                x
                for x in os.listdir(f"{samples_directory}/sample_0")
                if os.path.isdir(f"{samples_directory}/sample_0/{x}")
                and x.startswith("replicate_")
            ]
        )
        self.distance_results = distance_results or read_distances_from_json(
            os.path.join(
                *(
                    ["/"] * self.posterior_simulations_directory.startswith("/")
                    + self.posterior_simulations_directory.split("/")[:-1]
                ),
                "posterior_distances.json",
            )
        )
        assert set(self.paradigm_names) == set(self.distance_results.keys())

        with open(
            f"{self.posterior_simulations_directory}/abc_posterior_data.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "logging_directory": logging_directory,
                    "run_id": run_id,
                    "record_frequency": record_frequency,
                    "paradigm_names": paradigm_names,
                    "samples_per_paradigm": samples_per_paradigm,
                    "replicates_per_sample": replicates_per_sample,
                },
                file,
                indent=4,
            )

    def __iter__(self):
        for paradigm_name in self.paradigm_names:
            for sample_name in self.sample_names:
                for replicate_number in range(self.replicates_per_sample):
                    yield paradigm_name, sample_name, replicate_number

    @property
    def sample_names(self):
        return ["posterior_mean", "posterior_mode"] + [
            f"posterior_samples/sample_{i}" for i in range(self.samples_per_paradigm)
        ]

    @property
    def samples(self) -> list[tuple[str, str]]:
        return [
            (paradigm_name, sample_name)
            for paradigm_name in self.paradigm_names
            for sample_name in self.sample_names
        ]

    @property
    def patients(self) -> list[str]:
        return [
            x.split(".")[0]
            for x in os.listdir(
                f"{self.posterior_simulations_directory}/{self.paradigm_names[0]}/"
                f"posterior_samples/sample_0/replicate_0/cell_records/mutational_burden"
            )
        ]

    @property
    def json_file(self):
        return f"{self.posterior_simulations_directory}/abc_posterior_data.json"

    def get_distance_result(
        self, paradigm_name: str, sample_name: str
    ) -> DistanceResult | None:
        return self.distance_results[paradigm_name][sample_name]

    def read_fitness_summary(
        self,
        paradigm_name: str,
        sample_name: str,
        replicate_number: int,
        smoking_record: ExSmokerRecord | NonSmokerRecord | SmokerRecord | None,
        patient: str,
    ) -> pd.DataFrame:
        if smoking_record is not None:
            assert smoking_record.patient == patient
        return pd.read_csv(
            f"{self.posterior_simulations_directory}/{paradigm_name}/{sample_name}/"
            f"replicate_{replicate_number}/cell_records/fitness_summaries/"
            f"{patient}.csv"
        ).assign(
            year=lambda x: x["step_number"] / self.record_frequency,
            normalised_sm_mean_fitness=lambda x: x["sm_mean_fitness"]
            - x.groupby("year")["sm_mean_fitness"].transform("first"),
            normalised_ns_mean_fitness=lambda x: x["ns_mean_fitness"]
            - x.groupby("year")["ns_mean_fitness"].transform("first"),
            smoking=lambda x, _smoking_record=smoking_record: x["year"].apply(
                lambda year: (
                    _smoking_record.smoking_at_age(year)
                    if _smoking_record is not None
                    else False
                )
            ),
            relevant_normalised_mean_fitness=lambda x: np.where(
                x["smoking"],
                x["normalised_sm_mean_fitness"],
                x["normalised_ns_mean_fitness"],
            ),
            relevant_std_fitness=lambda x: np.where(
                x["smoking"], x["sm_std_fitness"], x["ns_std_fitness"]
            ),
            relevant_mean_fitness=lambda x: np.where(
                x["smoking"], x["sm_mean_fitness"], x["ns_mean_fitness"]
            ),
        )

    def read_csv_log(
        self,
        paradigm_name: str,
        sample_name: str,
        replicate_number: int,
        patient: str,
    ) -> pd.DataFrame:
        return pd.read_csv(
            f"{self.posterior_simulations_directory}/{paradigm_name}/{sample_name}/"
            f"replicate_{replicate_number}/{patient}.csv"
        ).assign(
            effective_division_rate=lambda x: x["new_cell_count"] / x["cell_count"],
            effective_differentiation_rate=lambda x: x["differentiated_cell_count"]
            / x["cell_count"],
            effective_immune_death_rate=lambda x: x["immune_death_count"]
            / x["cell_count"],
        )

    def read_phylogenies(
        self,
        paradigm_name: str,
        sample_name: str,
        replicate_number: int,
        patient: str,
    ) -> Generator[Phylo.Newick.Tree, None, None]:
        return Phylo.parse(  # type: ignore
            f"{self.posterior_simulations_directory}/{paradigm_name}/{sample_name}/"
            f"replicate_{replicate_number}/{patient}.nwk",
            "newick",
        )

    def read_cell_records(
        self,
        paradigm_name: str,
        sample_name: str,
        replicate_number: int,
        patient: str,
        smoking_record: ExSmokerRecord | NonSmokerRecord | SmokerRecord | None,
    ) -> pd.DataFrame:
        if smoking_record is not None:
            assert smoking_record.patient == patient
        cell_records = pd.read_csv(
            f"{self.posterior_simulations_directory}/{paradigm_name}/{sample_name}/"
            f"replicate_{replicate_number}/cell_records/mutational_burden/{patient}.csv"
        ).assign(
            total_mutations=lambda df: (
                df["driver_non_smoking_signature_mutations"]
                + df["driver_smoking_signature_mutations"]
                + df["passenger_smoking_signature_mutations"]
                + df["passenger_non_smoking_signature_mutations"]
            ),
            smoking_signature_mutations=lambda df: (
                df["driver_smoking_signature_mutations"]
                + df["passenger_smoking_signature_mutations"]
            ),
            non_smoking_signature_mutations=lambda df: (
                df["driver_non_smoking_signature_mutations"]
                + df["passenger_non_smoking_signature_mutations"]
            ),
            age=lambda df: (
                (df["record_number"] / self.record_frequency)
                if "record_number" in df
                else (
                    df["step_number"]
                    / df["step_number"].unique()[self.record_frequency - 1]
                    # this is steps_per_year
                )
            ),
        )
        if smoking_record is not None and smoking_record.status == "non-smoker":
            assert cell_records[
                cell_records["smoking_signature_mutations"]
                != cell_records["smoking_signature_mutations"].iloc[0]
            ].empty
        return cell_records

    def read_final_timepoint_mutational_burden(
        self,
        paradigm_name: str,
        sample_name: str,
        replicate_number: int,
        patient: str,
        smoking_record: ExSmokerRecord | NonSmokerRecord | SmokerRecord | None,
    ) -> pd.DataFrame:
        cell_records = self.read_cell_records(
            paradigm_name, sample_name, replicate_number, patient, smoking_record
        )
        if "record_number" in cell_records.columns:
            return cell_records[
                cell_records["record_number"] == cell_records["record_number"].max()
            ]
        return cell_records[
            cell_records["step_number"] == cell_records["step_number"].max()
        ]

    def read_sample_cell_records(
        self,
        paradigm_name: str,
        sample_name: str,
        patient: str,
        smoking_record: ExSmokerRecord | NonSmokerRecord | SmokerRecord | None,
        final_timepoint: bool = False,
    ) -> list[pd.DataFrame]:
        return [
            (
                self.read_final_timepoint_mutational_burden(
                    paradigm_name, sample_name, i, patient, smoking_record
                )
                if final_timepoint
                else self.read_cell_records(
                    paradigm_name, sample_name, i, patient, smoking_record
                )
            )
            for i in range(self.replicates_per_sample)
        ]

    def delete_cell_records(self) -> None:
        for paradigm_name, sample_name, replicate_number in self:
            try:
                shutil.rmtree(
                    f"{self.posterior_simulations_directory}/{paradigm_name}/"
                    f"{sample_name}/replicate_{replicate_number}/cell_records"
                )
            except FileNotFoundError:
                print(
                    f"Cell records for {paradigm_name}, {sample_name}, "
                    f"{replicate_number}  in {self.posterior_simulations_directory} "
                    "not found; not deleting"
                )


def read_from_json(json_file: str) -> ABCPosteriorData:
    with open(json_file, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as err:
            raise ValueError(f"Could not decode JSON file {json_file}") from err
    return ABCPosteriorData(**data)
