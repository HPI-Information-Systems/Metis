import json
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

from metis.dismis.detection.detection import DETECTORS_LITERAL, run_detection_algorithms
from metis.dismis.utils.logging import dismis_logger
from metis.dismis.utils.pathutils import require_exists
from metis.dismis.utils.types import COLUMN_TYPES

warnings.filterwarnings("ignore")

SHORTCUT_TEXT_FEATURES = [
    "semantic_placeholder",
    "semantic_comments",
    "semantic_unsure",
]
# Type mapping
type_mapping = {"numeric": 0, "date": 1, "categorical": 2, "text": 3}
target_type_str_map = {v: k for k, v in type_mapping.items()}


def _load_trained_model_and_features_for_column_type(
    models_dir: Path | str | None, column_type: COLUMN_TYPES
) -> Tuple[Any, List[str]]:
    models_dir = require_exists(models_dir, "Models directory")

    model_dir = models_dir / column_type
    metadata = json.load(open(model_dir / "metadata.json", "r", encoding="utf-8"))

    if column_type in {"numeric", "date"}:
        return load(model_dir / "parsable_model.joblib"), metadata["parsable_features"]

    if column_type in {"categorical", "text"}:
        return load(model_dir / "model.joblib"), metadata["features"]

    raise ValueError(f"Unsupported type '{column_type}' for model loading.")


def _load_trained_models_and_features(
    models_dir: Path | str | None,
) -> Dict[COLUMN_TYPES, Tuple[Any, List[str]]]:
    return {
        "numeric": _load_trained_model_and_features_for_column_type(
            models_dir, "numeric"
        ),
        "date": _load_trained_model_and_features_for_column_type(models_dir, "date"),
        "categorical": _load_trained_model_and_features_for_column_type(
            models_dir, "categorical"
        ),
        "text": _load_trained_model_and_features_for_column_type(models_dir, "text"),
    }


def _create_column_feature_matrix(
    column_scores_per_detector: Dict[str, pd.Series],
    features: List[str],
    length: int,
):
    feature_data: Dict[str, np.ndarray] = {}
    for feature in features:
        if feature == "type":
            continue
        feature_name = feature.replace("_30b8b", "")
        feature_data[feature_name] = column_scores_per_detector.get(
            feature_name, pd.Series(np.zeros(length, dtype=float))
        ).to_numpy()
        if "semantic_valid" in feature_name:
            feature_data[feature_name] = 1 - feature_data[feature_name]
    return pd.DataFrame(feature_data)


def _parse_numeric(series: pd.Series) -> np.ndarray:
    parsed = pd.to_numeric(series, errors="coerce")
    raw = series.astype(str).str.strip()
    invalid = parsed.isna() & series.notna() & (raw != "")
    return invalid.to_numpy()


def _parse_date(series: pd.Series) -> np.ndarray:
    parsed = pd.to_datetime(series, errors="coerce")
    raw = series.astype(str).str.strip()
    invalid = parsed.isna() & series.notna() & (raw != "")
    return invalid.to_numpy()


def _shortcut_text_flags(column_scores_per_detector: Dict[str, pd.Series], length: int):
    flags = np.zeros(length, dtype=bool)
    for feature_name in SHORTCUT_TEXT_FEATURES:
        scores = column_scores_per_detector.get(
            feature_name, pd.Series(np.zeros(length, dtype=float))
        ).to_numpy()
        flags |= scores > 0.95
    return flags


def predict_with_ensemble(
    detection_results,
    dataset: pd.DataFrame,
    column_types: Dict[str, COLUMN_TYPES],
    trained_models: Dict[COLUMN_TYPES, Tuple[Any, List[str]]],
):
    """
    Use trained models to predict DMVs and create ensemble predictions.

    Args:
        detection_results: Dictionary of detector results (detector_name -> (df_score, df_predict))
                          where df_score has same shape as input dataset (rows x columns)
                          and each cell contains the feature value for that position
        column_types: Dictionary mapping column names to their types
        trained_models: Dictionary of trained models by classifier and type

    Returns:
        Updated detection_results with added ensemble predictions
    """
    print("\n" + "=" * 80)
    print("Running Ensemble Predictions")
    print("=" * 80)

    pred_df = pd.DataFrame(index=dataset.index)
    proba_df = pd.DataFrame(index=dataset.index)
    for column, col_type in column_types.items():
        column_data = dataset[column]
        model, features = trained_models[col_type]
        column_scores_per_detector = {
            detector: scores[column]
            for detector, (scores, _) in detection_results.items()
        }
        input_features = _create_column_feature_matrix(
            column_scores_per_detector,
            features,
            len(column_data),
        )
        proba = model.predict_proba(input_features.fillna(0))[:, 1]
        pred = (proba >= 0.5).astype(int)

        if col_type in {"numeric", "date"}:
            invalid = (
                _parse_date(column_data)
                if col_type == "date"
                else _parse_numeric(column_data)
            )
            pred[invalid] = 1
            proba[invalid] = 1.0
        elif col_type in {"categorical", "text"}:
            shortcut = _shortcut_text_flags(
                column_scores_per_detector, len(column_data)
            )
            pred[shortcut] = 1
            proba[shortcut] = np.maximum(proba[shortcut], 1.0)

        pred_df[column] = pred
        proba_df[column] = proba

    return (proba_df, pred_df)


def run_dismis_detection(
    *,
    detectors: List[DETECTORS_LITERAL],
    dataset: pd.DataFrame,
    column_types: Dict[str, COLUMN_TYPES],
    models_dir: Path | str,
    value_embeddings_path: Path | str,
    example_dmvs_path: Path | str,
    example_embeddings_path: Path | str,
    results_path: Path | str | None = None,
    embedding_dim=128,
):
    trained_models = _load_trained_models_and_features(models_dir)
    all_embeddings = None

    with require_exists(value_embeddings_path, "Value embeddings").open("r") as f:
        all_embeddings = json.load(f)

    time_measurements: Dict[str, float] = {
        "loading": 0,
        "detection": 0,
        "prediction": 0,
        "evaluation": 0,
        "saving": 0,
        "total": 0,
    }
    detectors_time_measurements: Dict[str, Dict[int, float]] = {}

    # 1. Load dataset
    loading_starttime = time.time()
    target_columns = dataset.columns.to_list()
    polluted_dataset = dataset.copy()

    for col, col_type in column_types.items():
        if col_type == "numeric":
            polluted_dataset[col] = (
                polluted_dataset[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("%", "")
            )

    with require_exists(example_dmvs_path, "Example DMVs").open("r") as f:
        example_dmvs = json.load(f)

    with require_exists(example_embeddings_path, "Example embeddings").open("r") as f:
        example_embeddings = json.load(f)

    for col in example_dmvs.keys():
        for type, values in example_dmvs[col].items():
            try:
                example_dmvs[col][type] = [
                    example_embeddings[str(v)][:embedding_dim] for v in values
                ]
            except:
                raise ValueError(
                    f"Some of values '{values}' not found in precomputed embeddings for column '{col}'. Please make sure all example DMVs are covered in the precomputed embeddings."
                )

    del example_embeddings
    embeddings = {}
    for col in column_types.keys():
        if column_types[col] not in ["text", "categorical"]:
            continue

        emb = []
        for val in polluted_dataset[col].astype(str).tolist():
            if val in all_embeddings[col]:
                emb.append(all_embeddings[col][val][:embedding_dim])
            else:
                emb.append([0.0] * embedding_dim)
        embeddings[col] = np.array(emb, dtype=np.float32)

    time_measurements["loading"] += time.time() - loading_starttime

    # 2. Run detection algorithms (these produce features)
    detection_starttime = time.time()

    detection_results, detection_timings, _ = run_detection_algorithms(
        polluted_dataset,
        detectors=detectors,
        column_types=column_types,
        target_columns=target_columns,
        example_DMVs=example_dmvs,
        embeddings=embeddings,
    )

    time_measurements["detection"] += time.time() - detection_starttime
    for detector_name, times in detection_timings.items():
        detectors_time_measurements.setdefault(detector_name, {})
        for step, duration in times.items():
            detectors_time_measurements[detector_name][step] = (
                detectors_time_measurements[detector_name].get(step, 0) + duration
            )

    # 3. Use trained models to create ensemble predictions
    prediction_starttime = time.time()
    scores, predictions = predict_with_ensemble(
        detection_results, dataset, column_types, trained_models
    )
    time_measurements["prediction"] = time.time() - prediction_starttime

    # Write results to output directory for later inspection
    if results_path is not None:
        results_path = Path(results_path)
        results_path.mkdir(parents=True, exist_ok=True)
        scores.to_csv(results_path / "dismis_scores.csv", index=False)
        predictions.to_csv(results_path / "dismis_predictions.csv", index=False)
        for detector_name, (df_score, df_predict) in detection_results.items():
            detector_dir = results_path / detector_name
            detector_dir.mkdir(parents=True, exist_ok=True)
            df_score.to_csv(detector_dir / "scores.csv", index=False)
            df_predict.to_csv(detector_dir / "predictions.csv", index=False)
        with (results_path / "timings.json").open("w") as f:
            json.dump(time_measurements, f, indent=4)
        with (results_path / "detectors_timings.json").open("w") as f:
            json.dump(detectors_time_measurements, f, indent=4)

    return (scores, predictions)
