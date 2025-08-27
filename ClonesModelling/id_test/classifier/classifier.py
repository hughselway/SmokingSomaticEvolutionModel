from dataclasses import dataclass
from typing import Callable
import numpy as np

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.inspection import permutation_importance  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # type: ignore
from sklearn.model_selection import KFold  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.svm import SVC  # type: ignore

from .indexer import Indexer
from .generate_mds_features import MDSFeatures

CLASSIFIER_NAMES = [
    "logistic_regression",
    "support_vector_machine",
    "random_forest",
]


def fit_and_score_model(
    model: LogisticRegression | RandomForestClassifier | SVC,
    extract_importances_function: Callable[
        [LogisticRegression | RandomForestClassifier | SVC, np.ndarray, np.ndarray],
        np.ndarray,
    ],
    concatenated_features: np.ndarray,
    indexer: Indexer,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None]:
    scaler = StandardScaler()
    rescaled_features = scaler.fit_transform(concatenated_features)
    cv = KFold(n_splits := 5, shuffle=True, random_state=0)

    n_features = rescaled_features.shape[1]

    cross_val_scores = np.zeros(n_splits)
    confusion_matrices = np.zeros((n_splits, indexer.n_paradigms, indexer.n_paradigms))
    feature_importances = np.zeros((n_splits, n_features))
    true_data_predictions = np.zeros(n_splits) if indexer.include_true_data else None

    for i, (train_index, test_index) in enumerate(
        cv.split(
            rescaled_features[:-1] if indexer.include_true_data else rescaled_features
        )
    ):
        x_train, x_test = rescaled_features[train_index], rescaled_features[test_index]
        y_train, y_test = (
            indexer.paradigm_index_labels[train_index],
            indexer.paradigm_index_labels[test_index],
        )

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        cross_val_scores[i] = model.score(x_test, y_test)
        this_fold_confusion = sk_confusion_matrix(
            y_test, predictions, labels=range(indexer.n_paradigms)
        )
        if this_fold_confusion.shape[0] < indexer.n_paradigms:
            raise ValueError(
                "Confusion matrix shape is not equal to the number of paradigms: "
                f"{this_fold_confusion.shape}, {indexer.n_paradigms}"
            )
        confusion_matrices[i] = this_fold_confusion
        feature_importances[i] = extract_importances_function(
            model, rescaled_features, indexer.paradigm_index_labels
        )

        if indexer.include_true_data:
            assert true_data_predictions is not None
            true_data_predictions[i] = model.predict(
                rescaled_features[-1].reshape(1, -1)
            )

    cross_val_score_mean = np.mean(cross_val_scores)
    aggregated_confusion_matrix = np.sum(confusion_matrices, axis=0)
    aggregated_confusion_matrix = aggregated_confusion_matrix / np.sum(
        aggregated_confusion_matrix, axis=1, keepdims=True
    )
    mean_feature_importances = np.mean(feature_importances, axis=0)

    return (
        cross_val_score_mean,  # type: ignore
        mean_feature_importances,
        aggregated_confusion_matrix,
        true_data_predictions,
    )


def extract_importances(
    classifier_name: str,
    model: LogisticRegression | RandomForestClassifier | SVC,
    rescaled_features: np.ndarray,
    paradigm_indices: np.ndarray,
    include_true_data: bool,
) -> np.ndarray:
    if classifier_name == "random_forest":
        return model.feature_importances_  # type: ignore
    if classifier_name == "logistic_regression":
        return np.abs(model.coef_).mean(axis=0)  # type: ignore
    assert classifier_name == "support_vector_machine"
    return permutation_importance(  # type: ignore
        model,
        rescaled_features[:-1] if include_true_data else rescaled_features,
        paradigm_indices,
        n_jobs=-1,
    ).importances_mean  # type: ignore
    ## Optional alternative to avoid slow permutation_importance:
    # return np.ones(rescaled_features.shape[1])  # type: ignore


def get_classifier(
    classifier_name: str,
) -> Callable[
    [np.ndarray, Indexer], tuple[float, np.ndarray, np.ndarray, np.ndarray | None]
]:
    """
    Return the classifier function given the classifier name.
    """
    return lambda concatenated_features, indexer: fit_and_score_model(
        {
            "random_forest": RandomForestClassifier(random_state=0, n_jobs=-1),
            "support_vector_machine": SVC(random_state=0),
            # "support_vector_machine": SVC(kernel="linear", C=0.1),
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=0, n_jobs=-1
            ),
        }[classifier_name],
        lambda model, rescaled_features, paradigm_indices, include_true_data_=indexer.include_true_data: extract_importances(
            classifier_name,
            model,
            rescaled_features,
            paradigm_indices,
            include_true_data_,
        ),
        concatenated_features,
        indexer,
    )


@dataclass
class ClassifierIndexer:
    distance_function_names: list[str]
    n_features_per_dist_fn_options: list[int]

    def __init__(
        self,
        distance_function_names: list[str],
        n_features_per_dist_fn_options: list[int] | None = None,
    ) -> None:
        if n_features_per_dist_fn_options is None:
            n_features_per_dist_fn_options = [2, 5, 10, 20]
        self.distance_function_names = distance_function_names
        self.n_features_per_dist_fn_options = n_features_per_dist_fn_options

    @property
    def n_distance_functions(self) -> int:
        return len(self.distance_function_names)


@dataclass
class ClassifierOutput:
    """
    cross_val_score: float
    feature_importances_by_dist: np.ndarray of shape (n_distance_functions,)
    confusion_matrix: np.ndarray of shape (n_paradigms, n_paradigms)
    true_data_predictions: np.ndarray of shape (n_splits,) or None
    """

    cross_val_score: float
    dist_fn_importances: np.ndarray
    confusion_matrix: np.ndarray
    true_data_predictions: np.ndarray | None = None

    @property
    def n_distance_functions(self) -> int:
        return self.dist_fn_importances.shape[1]

    def write(self, output_file: str) -> None:
        np.savez(
            output_file,
            cross_val_score=self.cross_val_score,
            feature_importances_by_dist=self.dist_fn_importances,
            confusion_matrix=self.confusion_matrix,
            true_data_predictions=(
                self.true_data_predictions
                if self.true_data_predictions is not None
                else np.array([])
            ),
        )


def read_classifier_output(output_file: str) -> ClassifierOutput:
    with np.load(output_file) as data:
        return ClassifierOutput(
            data["cross_val_score"],
            data["feature_importances_by_dist"],
            data["confusion_matrix"],
            (
                data["true_data_predictions"]
                if "true_data_predictions" in data
                and len(data["true_data_predictions"]) > 0
                else None
            ),
        )


def fit_and_score_classifier(
    mds_features_data: MDSFeatures, classifier_name: str
) -> ClassifierOutput:
    concatenated_features_data = mds_features_data.concatenate_df_features()

    (cross_val_score, feature_importances, confusion_matrix, true_data_predictions) = (
        get_classifier(classifier_name)(
            concatenated_features_data, mds_features_data.indexer
        )
    )
    dist_fn_importances = mean_over_distance_function_features(
        feature_importances,
        n_features_per_dist_fn=mds_features_data.n_features_per_dist_fn,
        n_distance_functions=mds_features_data.indexer.n_distance_functions,
        features_axis=0,
    )
    classifier_output = ClassifierOutput(
        cross_val_score, dist_fn_importances, confusion_matrix, true_data_predictions
    )
    return classifier_output


def mean_over_distance_function_features(
    array: np.ndarray,
    n_features_per_dist_fn: int,
    n_distance_functions: int,
    features_axis: int = 0,
) -> np.ndarray:
    assert array.shape[features_axis] == n_distance_functions * n_features_per_dist_fn
    transformed_array = array.reshape(
        array.shape[:features_axis]
        + (n_distance_functions, n_features_per_dist_fn)
        + array.shape[features_axis + 1 :]
    ).mean(axis=features_axis + 1)
    assert isinstance(transformed_array, np.ndarray)
    assert (
        transformed_array.shape[:features_axis]
        + transformed_array.shape[features_axis + 1 :]
        == array.shape[:features_axis] + array.shape[features_axis + 1 :]
    )
    assert transformed_array.shape[features_axis] == n_distance_functions
    return transformed_array
