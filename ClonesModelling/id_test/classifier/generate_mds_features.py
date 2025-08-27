from dataclasses import dataclass
import json
import os
import numpy as np
from sklearn.manifold import MDS  # type: ignore
import matplotlib.pyplot as plt

from .generate_distance_matrix import DistanceMatrix
from .indexer import Indexer

from ...parameters.hypothetical_paradigm_class import is_protection_selection_subset


@dataclass
class MDSFeatures:
    features: np.ndarray
    # shape: (n_distance_functions, n_datapoints, n_features_per_dist_fn)
    stress: np.ndarray
    # shape: (n_distance_functions,)
    indexer: Indexer

    def __init__(
        self,
        n_features_per_dist_fn: int,
        distance_matrix: DistanceMatrix | None = None,
        mds_verbose: int = 0,
        seed: int = 0,
        features: np.ndarray | None = None,
        stress: np.ndarray | None = None,
        indexer: Indexer | None = None,
    ):
        if features is not None:
            assert stress is not None
            assert indexer is not None

            assert distance_matrix is None

            assert (
                features.shape[0] == indexer.n_distance_functions
            ), f"{features.shape[0]} != {indexer.n_distance_functions}; {indexer.distance_function_names}"
            assert (
                features.shape[1] == indexer.n_datapoints
            ), f"{features.shape[1]} != {indexer.n_datapoints}\n{indexer.n_paradigms},{indexer.simulation_replicates_per_paradigm}"
            assert (
                features.shape[2] == n_features_per_dist_fn
            ), f"{features.shape[2]} != {n_features_per_dist_fn}"

            assert (
                stress.shape[0] == indexer.n_distance_functions
            ), f"{stress.shape[0]} != {indexer.n_distance_functions}"

            self.features = features
            self.stress = stress
            self.indexer = indexer
            return
        assert distance_matrix is not None
        self.indexer = distance_matrix.indexer
        self.features = np.empty(
            (
                distance_matrix.indexer.n_distance_functions,
                distance_matrix.indexer.n_datapoints,
                n_features_per_dist_fn,
            )
        )
        self.stress = np.empty((distance_matrix.indexer.n_distance_functions))

        for distance_function_index, distance_function_name in enumerate(
            distance_matrix.indexer.distance_function_names
        ):
            if mds_verbose:
                print(f"Distance function: {distance_function_name}")
            mds = MDS(
                n_components=n_features_per_dist_fn,
                dissimilarity="precomputed",
                verbose=mds_verbose,
                n_jobs=-1,
                random_state=seed,
            )

            self.features[distance_function_index] = mds.fit_transform(
                distance_matrix.get_normalised_paradigm_agnostic_distances(
                    distance_function_index
                )
            )
            self.stress[distance_function_index] = mds.stress_

    @property
    def n_features_per_dist_fn(self) -> int:
        return self.features.shape[-1]

    @property
    def features_without_true_data(self) -> np.ndarray:
        if self.indexer.include_true_data:
            return self.features[:, :-1]
        return self.features

    def plot_all_options(self, mds_replicate_dir: str) -> None:
        for include_legend in [False, True]:
            for df_index, _ in enumerate(self.indexer.distance_function_names):
                for plot_true_data in [False] + (
                    [True] * self.indexer.include_true_data
                ):
                    self.plot(
                        save_dir=(
                            mds_replicate_dir
                            + f"/{'in' if plot_true_data else 'ex'}cluding_true_data/"
                            f"with{'out'*(not include_legend)}_legend/"
                        ),
                        include_true_data=plot_true_data,
                        include_legend=include_legend,
                        df_index=df_index,
                    )

    def plot(
        self,
        save_dir: str,
        include_true_data: bool,
        include_legend: bool,
        df_index: int,
    ):
        fig, ax = plt.subplots(figsize=(3 + 3 * include_legend, 3))
        for paradigm_index, paradigm in enumerate(self.indexer.hypothetical_paradigms):
            paradigm_mask = self.indexer.paradigm_index_labels == paradigm_index
            ax.scatter(
                self.features_without_true_data[df_index, paradigm_mask, 0],
                self.features_without_true_data[df_index, paradigm_mask, 1],
                label=paradigm.get_modules_string(),
                alpha=0.3,
                s=20,
            )
        if include_true_data:
            ax.scatter(
                self.features[df_index, -1, 0],
                self.features[df_index, -1, 1],
                label="true data",
                color="red",
                s=50,
                alpha=0.6,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("MDS 1")
        ax.set_ylabel("MDS 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if include_legend:
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                title="Paradigm",
                fontsize=12,
                title_fontsize=12,
            )
        fig.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(
            f"{save_dir}/{self.indexer.distance_function_names[df_index]}.pdf",
        )
        plt.close(fig)

    def concatenate_df_features(self) -> np.ndarray:
        assert self.features_without_true_data.shape == (
            self.indexer.n_distance_functions,
            self.indexer.n_datapoints_excluding_true_data,
            self.n_features_per_dist_fn,
        )
        return self.features.transpose(1, 0, 2).reshape(
            self.indexer.n_datapoints,
            self.indexer.n_distance_functions * self.n_features_per_dist_fn,
        )

    def write(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/features.npy", self.features)
        np.save(f"{save_dir}/stress.npy", self.stress)
        with open(f"{save_dir}/indexer.json", "w", encoding="utf-8") as f:
            json.dump(self.indexer.json_dict(), f)

    def subset_distance_functions(self, indexer: Indexer) -> "MDSFeatures":
        assert indexer.is_df_subset(self.indexer), f"{self.indexer}\n\n{indexer}"
        distance_function_indices = [
            self.indexer.distance_function_names.index(df_name)
            for df_name in indexer.distance_function_names
        ]
        return MDSFeatures(
            self.n_features_per_dist_fn,
            features=self.features[distance_function_indices],
            stress=self.stress[distance_function_indices],
            indexer=indexer,
        )

    def subset_to_first_replicate(self, n_replicates: int) -> "MDSFeatures":
        """
        Subset the datapoints to be only the first replicate of each simulation
        """
        assert self.indexer.simulation_replicates_per_paradigm % n_replicates == 0
        n_simulations_per_paradigm = (
            self.indexer.simulation_replicates_per_paradigm // n_replicates
        )
        assert self.features.shape[1] == (
            self.indexer.n_paradigms * n_simulations_per_paradigm * n_replicates
            + int(self.indexer.include_true_data)
        )
        if self.indexer.include_true_data:
            features = np.concatenate(
                [self.features[:, ::n_replicates], self.features[:, -1][:, np.newaxis]],
                axis=1,
            )
        else:
            features = self.features[:, ::n_replicates]

        return MDSFeatures(
            self.n_features_per_dist_fn,
            features=features,
            stress=self.stress,
            indexer=Indexer(
                "2D_wasserstein" in self.indexer.distance_function_names,
                self.indexer.max_modules,
                simulation_replicates_per_paradigm=n_simulations_per_paradigm,
                include_true_data=self.indexer.include_true_data,
                distance_function_names=self.indexer.distance_function_names,
            ),
        )

    def subset_to_n_module_paradigms(self, n: int) -> "MDSFeatures":
        """
        Subset the datapoints to be only those from paradigms with 0 or 1 module
        """
        assert n > 0 or n == -1
        included_paradigm_indices = (
            [
                i
                for i, paradigm in enumerate(self.indexer.hypothetical_paradigms)
                if len(paradigm.get_module_names(include_base=False)) <= n
            ]
            if n > 0
            else [
                i
                for i, paradigm in enumerate(self.indexer.hypothetical_paradigms)
                if is_protection_selection_subset(
                    paradigm.get_module_names(include_base=True)
                )
            ]
        ) + ([-1] if self.indexer.include_true_data else [])
        return MDSFeatures(
            self.n_features_per_dist_fn,
            features=self.features[
                :,
                np.pad(
                    np.isin(
                        self.indexer.paradigm_index_labels, included_paradigm_indices
                    ),
                    (0, 1 if self.indexer.include_true_data else 0),
                    constant_values=True,  # include the true data in the subset
                ),
            ],
            stress=self.stress,
            indexer=Indexer(
                "2D_wasserstein" in self.indexer.distance_function_names,
                max_modules=n,
                simulation_replicates_per_paradigm=self.indexer.simulation_replicates_per_paradigm,
                include_true_data=self.indexer.include_true_data,
                distance_function_names=self.indexer.distance_function_names,
            ),
        )

    def subset_to_simulation_subset(
        self, logging_directory: str, run_id: str, subset_index: int
    ) -> "MDSFeatures":
        with np.load(
            f"{logging_directory}/{run_id}/classifiers/simulation_subset_masks/"
            f"subset_replicate_{subset_index}.npz"
        ) as data:
            paradigm_simulation_indices = (
                np.array(  # n_paradigms x subset_n_simulations_per_paradigm
                    [
                        data[paradigm.get_modules_string(abbreviated=True)]
                        for paradigm in self.indexer.hypothetical_paradigms
                    ]
                )
            )
        # paradigm_simulation_indices has shape (n_paradigms, subset_n_simulations_per_paradigm)
        # features has shape (n_distance_functions, n_datapoints, n_features_per_dist_fn)
        # if include_true_data, then n_datapoints = n_paradigms * simulation_replicates_per_paradigm + 1
        # otherwise n_datapoints = n_paradigms * simulation_replicates_per_paradigm
        # we want to include the -1th datapoint if include_true_data, at the -1th position, so read that off first
        # we're subsetting the datapoints to be only the included simulations for that paradigm

        # first expand the features to be (n_distance_functions, n_paradigms, n_simulations_per_paradigm, n_features_per_dist_fn)
        # then subset the 3rd dimension according to the simulation indices matching the paradigm in the 2nd dimension
        # the collapse the 2nd and 3rd dimensions to get the new features
        if self.indexer.include_true_data:
            true_data_features = self.features[:, -1]
            features = self.features[:, :-1]
        else:
            true_data_features = None
            features = self.features

        assert paradigm_simulation_indices.shape[0] == self.indexer.n_paradigms
        assert (
            paradigm_simulation_indices.shape[1]
            < self.indexer.simulation_replicates_per_paradigm
        )

        features_by_paradigm = (
            features.transpose(0, 2, 1)
            .reshape(
                features.shape[0],
                self.n_features_per_dist_fn,
                self.indexer.n_paradigms,
                self.indexer.simulation_replicates_per_paradigm,
            )
            .transpose(0, 2, 3, 1)
        )
        subsetted_features_by_paradigm = np.empty(
            (
                features.shape[0],
                paradigm_simulation_indices.shape[0],
                paradigm_simulation_indices.shape[1],
                self.n_features_per_dist_fn,
            )
        )
        for i, paradigm_simulation_index_array in enumerate(
            paradigm_simulation_indices
        ):
            subsetted_features_by_paradigm[:, i] = features_by_paradigm[
                :, i, paradigm_simulation_index_array, :
            ]
        subsetted_features = (
            subsetted_features_by_paradigm.transpose((0, 3, 1, 2))
            .reshape(
                (
                    features.shape[0],
                    self.n_features_per_dist_fn,
                    paradigm_simulation_indices.shape[0]
                    * paradigm_simulation_indices.shape[1],
                )
            )
            .transpose(0, 2, 1)
        )  # shape (n_distance_functions, n_datapoints, n_features_per_dist_fn)
        if (
            self.indexer.include_true_data
        ):  # true_data_features (n_distance_functions, n_features_per_dist_fn)
            assert true_data_features is not None
            subsetted_features = np.concatenate(
                [subsetted_features, true_data_features[:, np.newaxis]], axis=1
            )
        return MDSFeatures(
            self.n_features_per_dist_fn,
            features=subsetted_features,
            stress=self.stress,
            indexer=Indexer(
                "2D_wasserstein" in self.indexer.distance_function_names,
                self.indexer.max_modules,
                simulation_replicates_per_paradigm=paradigm_simulation_indices.shape[1],
                include_true_data=self.indexer.include_true_data,
                distance_function_names=self.indexer.distance_function_names,
            ),
        )


def read_mds_features(save_dir: str, indexer: Indexer) -> MDSFeatures:
    features = np.load(f"{save_dir}/features.npy")
    stress = np.load(f"{save_dir}/stress.npy")
    n_features_per_dist_fn = features.shape[-1]
    with open(f"{save_dir}/indexer.json", "r", encoding="utf-8") as f:
        recorded_indexer = Indexer(**json.load(f))
    if indexer == recorded_indexer:
        return MDSFeatures(
            n_features_per_dist_fn, features=features, stress=stress, indexer=indexer
        )
    return MDSFeatures(
        n_features_per_dist_fn,
        features=features,
        stress=stress,
        indexer=recorded_indexer,
    ).subset_distance_functions(indexer)
