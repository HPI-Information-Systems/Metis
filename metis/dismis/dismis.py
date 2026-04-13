import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from detection.detection import run_detection_algorithms
from tqdm import tqdm

from metis.dismis.utils.logging import dismis_logger
from metis.dismis.utils.pathutils import require_exists

warnings.filterwarnings("ignore")

# Type mapping
type_mapping = {"numeric": 0, "date": 1, "categorical": 2, "text": 3}
target_type_str_map = {v: k for k, v in type_mapping.items()}


def load_trained_models(model_path: Path):
    """Load pretrained xgboost, random_forest, and mlp models."""
    # These imports are needed to load the trained models, which include xgboost and sklearn models. Without these imports, python throws a segfault
    import sklearn
    import xgboost

    with require_exists(model_path, "Model").open("rb") as f:
        return pickle.load(f)


def predict_with_ensemble(detection_results, types, trained_models):
    """
    Use trained models to predict DMVs and create ensemble predictions.

    Args:
        detection_results: Dictionary of detector results (detector_name -> (df_score, df_predict))
                          where df_score has same shape as input dataset (rows x columns)
                          and each cell contains the feature value for that position
        types: Dictionary mapping column names to their types
        trained_models: Dictionary of trained models by classifier and type

    Returns:
        Updated detection_results with added ensemble predictions
    """
    print("\n" + "=" * 80)
    print("Running Ensemble Predictions")
    print("=" * 80)

    # Get the shape of the dataset from the first detector's df_score
    first_detector = list(detection_results.keys())[0]
    df_score_sample, _ = detection_results[first_detector]
    n_rows, n_cols = df_score_sample.shape
    column_names = df_score_sample.columns.tolist()

    print(f"Dataset shape: {n_rows} rows x {n_cols} columns")

    df_xgboost_scores = pd.DataFrame(index=range(n_rows), columns=column_names)
    df_xgboost_predictions = pd.DataFrame(index=range(n_rows), columns=column_names)

    # Process each column
    for col in tqdm(column_names, desc="Predicting DMVs"):
        if col not in types:
            print(f"Warning: Column '{col}' not found in types, skipping...")
            continue

        col_type = types[col]
        type_id = type_mapping.get(col_type)

        # Build feature matrix for this column
        # Each row in the feature matrix corresponds to a row in the dataset
        # Each column in the feature matrix is a feature from detection_results
        available_features = list(detection_results.keys())
        feature_data = []

        for detector_name in available_features:
            df_score, _ = detection_results[detector_name]
            feature_data.append(df_score[col].values)

        feature_data = pd.DataFrame(
            np.column_stack(feature_data), columns=available_features
        )

        model_info = trained_models[type_id]
        model = model_info["model"]
        required_features = model_info["features"]

        missing_features = []
        mapped_features = []

        for feat_name in required_features:
            if any(feat_name.startswith(prefix) for prefix in available_features):
                mapped_feature = [
                    prefix
                    for prefix in sorted(
                        available_features, key=lambda x: len(x), reverse=True
                    )
                    if feat_name.startswith(prefix)
                ][0]
                # print(f"Mapped {feat_name} to {mapped_feature}")
                mapped_features.append(mapped_feature)
            elif feat_name == "type":
                feature_data["type"] = [type_mapping[types[col]]] * n_rows
                mapped_features.append("type")
            else:
                missing_features.append(feat_name)

        if len(missing_features) > 0 and len(missing_features) < 5:
            dismis_logger.warning(
                f"Warning: Column '{col}' missing {len(missing_features)} features: {missing_features[:5]}"
            )
        elif len(missing_features) >= 5:
            dismis_logger.warning(
                f"Warning: Column '{col}' missing {len(missing_features)} features"
            )

        preds = model.predict(feature_data[mapped_features].values)
        probas = model.predict_proba(feature_data[mapped_features].values)[:, 1]

        df_xgboost_scores[col] = probas
        df_xgboost_predictions[col] = preds

    # Add to detection_results
    detection_results["DISMIS"] = (df_xgboost_scores, df_xgboost_predictions)

    return (df_xgboost_scores, df_xgboost_predictions)


def run_dismis_detection(
    evaluation_config_path: Path,
    dataset_path: Path,
    types_path: Path,
    pollution_mask_path: Path,
    model_path: Path,
    value_embeddings_path: Path,
    example_dmvs_path: Path,
    example_embeddings_path: Path,
    embedding_dim=128,
):
    trained_models = load_trained_models(model_path)

    with open(evaluation_config_path, "r") as f:
        config = json.load(f)

    dismis_logger.info(f"Dataset: {dataset_path}")
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
    polluted_dataset = pd.read_csv(
        require_exists(dataset_path, "Dataset"), keep_default_na=False, na_values=[""]
    )
    dmv_labels = pd.read_csv(require_exists(pollution_mask_path, "Pollution mask"))
    target_columns = polluted_dataset.columns.to_list()

    with require_exists(types_path, "Types").open("r") as f:
        types = json.load(f)

    for col, col_type in types.items():
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
    for col in types.keys():
        if types[col] not in ["text", "categorical"]:
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

    # This is to run Raha as competitor.
    # Raha implicitly generates its labels by comparing the dirty and clean dataset. Where both differ, it counts the cell as error.
    # Since DMVs are obviously never explicit missing values, we set them to pd.NA in the cleaned dataset to allow Raha to detect them.
    cleaned_dataset = polluted_dataset.copy()
    for col in dmv_labels.columns:
        cleaned_dataset.loc[dmv_labels[col] == 1, col] = pd.NA

    detection_results, detection_timings, _ = (
        run_detection_algorithms(
            polluted_dataset,
            cleaned_dataset,
            detectors=config["detectors"],
            types=types,
            target_columns=target_columns,
            example_DMVs=example_dmvs,
            embeddings=embeddings,
        )
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
        detection_results, types, trained_models
    )
    time_measurements["prediction"] = time.time() - prediction_starttime

    return (scores, predictions)
