import os
import time
import numpy as np
import pandas as pd  # type: ignore

from .classifier import (
    fit_and_score_classifier,
    read_classifier_output,
    CLASSIFIER_NAMES,
    ClassifierIndexer,
)
from .generate_distance_matrix import read_idt_distances
from .generate_mds_features import MDSFeatures, read_mds_features
from .indexer import Indexer
from .plot import (
    plot_if_files_exist,
    plot_cross_val_scores,
    plot_stresses,
    plot_dist_fn_importances,
    plot_confusion_matrices,
    plot_true_data_predictions,
    plot_module_level_true_data_predictions,
)

from ..calculate_distances import DISTANCE_FUNCTIONS
from ...parse_cmd_line_args import parse_id_test_arguments


def run_classifiers(
    logging_directory: str,
    run_id: str,
    max_modules: int,
    include_2d_wasserstein: bool,
    simulation_replicates_per_paradigm: int,
    n_features_per_dist_fn_options: list[int],
    mds_replicate_count: int = 10,
    distance_function_names: list[str] | None = None,
    patient_subset: str | None = None,
    restrict_to_first_replicate: bool = False,
    restrict_to_n_module_paradigms: int | None = None,
    simulation_subset_index: int | None = None,
    include_true_data: bool = False,
) -> None:
    mds_directory, classifier_directory = get_classifier_directory(
        logging_directory,
        run_id,
        distance_function_names,
        patient_subset,
        restrict_to_first_replicate,
        restrict_to_n_module_paradigms,
        simulation_subset_index,
        include_true_data,
    )
    print(f"MDS dir: {mds_directory}; Classifier dir: {classifier_directory}")
    if plot_if_files_exist(classifier_directory):
        return
    cross_val_scores = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [
                n_features_per_dist_fn_options,
                range(mds_replicate_count),
                CLASSIFIER_NAMES,
            ],
            names=["n_features_per_dist_fn", "mds_replicate_index", "classifier_name"],
        ),
        columns=["cross_val_score"],
    )

    dist_fn_importances: list[pd.DataFrame] = []
    stresses: list[pd.DataFrame] = []
    confusion_matrices: list[pd.DataFrame] = []
    true_data_predictions: list[pd.DataFrame] = []

    distance_matrix = read_idt_distances(
        logging_directory,
        run_id,
        Indexer(
            include_2d_wasserstein,
            max_modules,
            simulation_replicates_per_paradigm,
            include_true_data,
            distance_function_names,
        ),
        mds_directory,
        patient_subset,
    )
    classifier_indexer = ClassifierIndexer(
        distance_function_names=distance_matrix.indexer.distance_function_names,
        n_features_per_dist_fn_options=n_features_per_dist_fn_options,
    )
    for n_features_per_dist_fn in n_features_per_dist_fn_options:
        print(
            f"n_features_per_dist_fn = {n_features_per_dist_fn} "
            f"({mds_replicate_count} replicates)",
            end=" ",
            flush=True,
        )
        nf_start_time = time.time()
        for mds_replicate_index in range(mds_replicate_count):
            print("MDS ", end="", flush=True)
            save_dir = (
                f"{mds_directory}/mds_features/"
                f"{n_features_per_dist_fn}_features_per_dist_fn/"
                f"replicate_{mds_replicate_index}/"
            )
            if os.path.exists(f"{save_dir}/features.npy"):
                mds_features_data = read_mds_features(save_dir, distance_matrix.indexer)
            else:
                mds_features_data = MDSFeatures(
                    n_features_per_dist_fn, distance_matrix, seed=mds_replicate_index
                )
                os.makedirs(save_dir, exist_ok=True)
                mds_features_data.write(
                    f"{mds_directory}/mds_features/"
                    f"{n_features_per_dist_fn}_features_per_dist_fn/"
                    f"replicate_{mds_replicate_index}/"
                )
            if restrict_to_first_replicate:
                mds_features_data = mds_features_data.subset_to_first_replicate(3)
            if restrict_to_n_module_paradigms:
                mds_features_data = mds_features_data.subset_to_n_module_paradigms(
                    restrict_to_n_module_paradigms
                )
                # not worth plotting 2D MDS for all paradigms, too many colours
                if n_features_per_dist_fn == 2:
                    mds_features_data.plot_all_options(
                        f"{save_dir}/plots/"
                        + (
                            "protection_selection_paradigms"
                            if restrict_to_n_module_paradigms == -1
                            else f"{restrict_to_n_module_paradigms}_module_paradigms"
                        ),
                    )
            if simulation_subset_index is not None:
                mds_features_data = mds_features_data.subset_to_simulation_subset(
                    logging_directory, run_id, simulation_subset_index
                )
            stresses.append(
                dataframe_from_ndarray(
                    mds_features_data.stress,
                    ["distance_function_name"],
                    [classifier_indexer.distance_function_names],
                )
                .assign(
                    n_features_per_dist_fn=n_features_per_dist_fn,
                    mds_replicate_index=mds_replicate_index,
                )
                .rename(columns={"value": "stress"})
            )
            for classifier_name in CLASSIFIER_NAMES:
                print(
                    "".join([word[0].upper() for word in classifier_name.split("_")]),
                    end="",
                    flush=True,
                )
                save_dir = (
                    f"{classifier_directory}/classifier_outputs/{classifier_name}/"
                    f"{n_features_per_dist_fn}_features_per_dist_fn/"
                    f"replicate_{mds_replicate_index}/"
                )
                try:
                    classifier_output = read_classifier_output(f"{save_dir}/output.npz")
                except (FileNotFoundError, KeyError):
                    classifier_output = fit_and_score_classifier(
                        mds_features_data, classifier_name
                    )
                    os.makedirs(save_dir, exist_ok=True)
                    classifier_output.write(f"{save_dir}/output.npz")
                print(".", end="", flush=True)

                cross_val_scores.loc[
                    (n_features_per_dist_fn, mds_replicate_index, classifier_name),
                    "cross_val_score",
                ] = classifier_output.cross_val_score
                dist_fn_importances.append(
                    dataframe_from_ndarray(
                        classifier_output.dist_fn_importances,
                        ["distance_function_name"],
                        [mds_features_data.indexer.distance_function_names],
                    )
                    .assign(
                        n_features_per_dist_fn=n_features_per_dist_fn,
                        mds_replicate_index=mds_replicate_index,
                        classifier_name=classifier_name,
                    )
                    .rename(columns={"value": "dist_fn_importance"})
                )
                confusion_matrices.append(
                    dataframe_from_ndarray(
                        classifier_output.confusion_matrix,
                        ["true_paradigm", "predicted_paradigm"],
                        [
                            mds_features_data.indexer.get_paradigm_names(),
                            mds_features_data.indexer.get_paradigm_names(),
                        ],
                    )
                    .assign(
                        n_features_per_dist_fn=n_features_per_dist_fn,
                        mds_replicate_index=mds_replicate_index,
                        classifier_name=classifier_name,
                    )
                    .rename(columns={"value": "confusion"})
                )
                if (
                    include_true_data
                    and classifier_output.true_data_predictions is not None
                ):
                    true_data_predictions.append(
                        pd.DataFrame(
                            {
                                "predicted_paradigm": (
                                    map(
                                        mds_features_data.indexer.get_paradigm_names().__getitem__,
                                        classifier_output.true_data_predictions.astype(
                                            int
                                        ),
                                    )
                                )
                            }
                        ).assign(
                            n_features_per_dist_fn=n_features_per_dist_fn,
                            mds_replicate_index=mds_replicate_index,
                            classifier_name=classifier_name,
                        )
                    )
            print("; ", end="", flush=True)
        print(
            "done in",
            time.strftime("%H:%M:%S", time.gmtime(time.time() - nf_start_time)),
        )
    os.makedirs(classifier_directory, exist_ok=True)

    cross_val_scores_df = cross_val_scores.reset_index()
    dist_fn_importances_df = pd.concat(dist_fn_importances)
    stresses_df = pd.concat(stresses)
    confusion_matrices_df = pd.concat(confusion_matrices)

    cross_val_scores_df.to_csv(
        f"{classifier_directory}/cross_val_scores.csv", index=False
    )
    dist_fn_importances_df.to_csv(
        f"{classifier_directory}/dist_fn_importances.csv", index=False
    )
    stresses_df.to_csv(f"{classifier_directory}/stresses.csv", index=False)
    confusion_matrices_df.to_csv(
        f"{classifier_directory}/confusion_matrices.csv", index=False
    )
    true_data_predictions_df = pd.concat(true_data_predictions)
    if include_true_data:
        true_data_predictions_df.to_csv(
            f"{classifier_directory}/true_data_predictions.csv", index=False
        )

    plot_cross_val_scores(cross_val_scores_df, classifier_directory)
    plot_dist_fn_importances(
        dist_fn_importances_df, classifier_indexer, classifier_directory
    )
    plot_stresses(stresses_df, classifier_directory)
    plot_confusion_matrices(confusion_matrices_df, classifier_directory)
    if include_true_data:
        if len(true_data_predictions_df.predicted_paradigm.unique()) <= 10:
            plot_true_data_predictions(true_data_predictions_df, classifier_directory)
        plot_module_level_true_data_predictions(
            true_data_predictions_df, classifier_directory
        )


def get_classifier_directory(
    logging_directory: str,
    run_id: str,
    distance_function_names: list[str] | None = None,
    patient_subset: str | None = None,
    restrict_to_first_replicate: bool = False,
    restrict_to_n_module_paradigms: int | None = None,
    simulation_subset_index: int | None = None,
    include_true_data: bool = False,
) -> tuple[str, str]:
    mds_directory = (
        f"{logging_directory}/{run_id}/classifiers"
        + ("/all_pts" if patient_subset is None else f"/pt_subsets/{patient_subset}")
        + ("/including_true_data" if include_true_data else "/excluding_true_data")
    )
    classifier_directory = (
        mds_directory
        + ("/first_replicate" if restrict_to_first_replicate else "/all_replicates")
        + (
            "/protection_selection_paradigms"
            if restrict_to_n_module_paradigms == -1
            else (
                f"/{restrict_to_n_module_paradigms}_module_paradigms"
                if restrict_to_n_module_paradigms
                else "/all_paradigms"
            )
        )
        + (
            "/all_dfs"
            if distance_function_names is None
            else (
                "/df_subsets/"
                + "_".join(
                    sorted(
                        "".join(x[0] for x in df_name.split("_"))
                        for df_name in distance_function_names
                    )
                )
            )
        )
        + (
            "/all_simulations"
            if simulation_subset_index is None
            else f"/simulation_subsets/subset_replicate_{simulation_subset_index}"
        )
    )
    return mds_directory, classifier_directory


def dataframe_from_ndarray(
    array: np.ndarray,
    index_names: list[str],
    index_values: list[list] | None = None,
) -> pd.DataFrame:
    # Reshape the array to 2D while keeping track of indices
    indices = np.indices(array.shape).reshape(array.ndim, -1).T
    index_name_array = np.empty(indices.shape, dtype=object)
    assert len(index_names) == array.ndim, f"{len(index_names)} != {array.ndim}"
    if index_values is not None:
        assert len(index_values) == array.ndim, f"{len(index_values)} != {array.ndim}"
        for i in range(array.ndim):
            assert (
                len(index_values[i]) == array.shape[i]
            ), f"{len(index_values[i])} != {array.shape[i]} for {index_names[i]}"
            index_name_array[:, i] = np.array(index_values[i])[indices[:, i]]
    df = pd.DataFrame(
        np.concatenate((index_name_array, array.flatten()[:, None]), axis=1),
        columns=index_names + ["value"],
    )
    return df


if __name__ == "__main__":
    parsed_args = parse_id_test_arguments()
    for restrict_to_n_module_paradigms in [None, -1]:
        start_time = time.time()
        print("restrict_to_n_module_paradigms:", restrict_to_n_module_paradigms)
        run_classifiers(
            logging_directory=parsed_args.logging_directory,
            run_id=parsed_args.run_id,
            max_modules=parsed_args.max_modules,
            include_2d_wasserstein=parsed_args.compare_smoking_signature_mutations,
            simulation_replicates_per_paradigm=(
                parsed_args.simulations_per_paradigm * parsed_args.replicate_count
            ),
            n_features_per_dist_fn_options=[2, 3, 5, 10, 20],  # , 50, 100],
            mds_replicate_count=3,
            # distance_function_names=[
            #     "abs_j_one",
            #     "branch_length_wasserstein",
            #     "mm_larger_means_sq_diff",
            #     "mm_smaller_means_sq_diff",
            #     "mm_larger_weight_sq_diff",
            #     "smoking_sig_only",
            #     "wasserstein",
            # ],
            distance_function_names=[
                df_name
                for df_name in DISTANCE_FUNCTIONS
                if df_name
                not in [
                    "total_branch_length_squared",
                    "sum_control",
                    "zero_control",
                    # "mean_subtracted_2D_simplified",
                    # "l2_j_one",
                    "2D_wasserstein",
                    ## two DFs removed because they didn't map properly in the idt_2025-02-05_14-53-34 run
                    # "mean_subtracted",
                    # "total_branch_length",
                    #
                    ## extra DFs removed for true data comparison analysis
                    # "random_control",
                    # 0th percentile idt_2025-04-07_10-16-59
                    # - none
                    # 10th percentile idt_2025-04-07_10-16-59
                    # "mean_subtracted",
                    # "mean_subtracted_2D_simplified",
                    # "abs_j_one",
                    # "l2_j_one",
                    # # 20th percentile (as well as above) idt_2025-04-07_10-16-59
                    # "z_values",
                    # "branch_length_wasserstein",
                    # # 25th percentile (as well as above) idt_2025-04-07_10-16-59
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "mm_larger_means_sq_diff",
                    # "total_branch_length",
                    #
                    # # 5th percentile idt_2025-02-05_14-53-34
                    # "mean_subtracted_2D_simplified",
                    # "l2_j_one",
                    # # 10th percentile (as well as 5th) idt_2025-02-05_14-53-34
                    # "abs_j_one",
                    # # 25th percentile (as well as above) idt_2025-02-05_14-53-34
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "mm_larger_means_sq_diff",
                    # "branch_length_wasserstein",
                    # # 5th percentile idt_2024-10-03_19-04-47
                    # "abs_j_one",
                    # "l2_j_one",
                    # # 10th percentile (as well as 5th) idt_2024-10-03_19-04-47
                    # "mean_subtracted",
                    # "mean_subtracted_2D_simplified",
                    # # 25th percentile (as well as above) idt_2024-10-03_19-04-47
                    # "wasserstein",
                    # "2D_wasserstein_simplified",
                    # "smoking_sig_only",
                    # "z_values",
                    # "mm_larger_means_sq_diff",
                ]
                # and "control" not in df_name
            ],
            # restrict_to_first_replicate=True,
            restrict_to_n_module_paradigms=restrict_to_n_module_paradigms,
            include_true_data=True,
        )
        print(
            "Total time taken:",
            time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
        )
