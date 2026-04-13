import argparse
import json
import os
import pickle
import random
import time
import warnings

import numpy as np
import pandas as pd
from detection.detection import run_detection_algorithms
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from metis.dismis.evaluation.evaluation import evaluate_detection_results
from metis.dismis.pollution.conditions import create_conditions, create_labels
from metis.dismis.pollution.pollution import introduce_errors

warnings.filterwarnings("ignore")

USE_EMBEDDINGS = True

# Type mapping
type_mapping = {"numeric": 0, "date": 1, "categorical": 2, "text": 3}
target_type_str_map = {v: k for k, v in type_mapping.items()}


def load_trained_models(model_path):
    """Load pretrained xgboost, random_forest, and mlp models."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Please run train_dismis.py first."
        )
    with open(model_path, "rb") as f:
        trained_models = pickle.load(f)

    return trained_models


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
            print(
                f"Warning: Column '{col}' missing {len(missing_features)} features: {missing_features[:5]}"
            )
        elif len(missing_features) >= 5:
            print(f"Warning: Column '{col}' missing {len(missing_features)} features")

        preds = model.predict(feature_data[mapped_features].values)
        probas = model.predict_proba(feature_data[mapped_features].values)[:, 1]

        df_xgboost_scores[col] = probas
        df_xgboost_predictions[col] = preds

    # Add to detection_results
    detection_results["DISMIS"] = (df_xgboost_scores, df_xgboost_predictions)

    return detection_results


def run_dismis(
    evaluation_config_path,
    dataset_path,
    model_path,
    value_embeddings="_vllm_embeddings.json",
    example_dmvs="example_dmvs_detection_5.json",
    example_embeddings="precomputed_example_embeddings_5.json",
    embedding_dim=128,
    suffix="",
):
    LLM = None

    # Load pretrained models
    if model_path is not None:
        print("Loading pretrained models...")
        trained_models = load_trained_models(model_path)

    with open(evaluation_config_path, "r") as f:
        config = json.load(f)

    print("Dataset:", dataset_path)
    all_embeddings = None
    if USE_EMBEDDINGS and os.path.exists(
        os.path.join(dataset_path, dataset_path.split("/")[-2] + value_embeddings)
    ):
        print("Loading existing embeddings...")
        with open(
            os.path.join(dataset_path, dataset_path.split("/")[-2] + value_embeddings),
            "r",
        ) as f:
            all_embeddings = json.load(f)

    time_measurements = {
        "loading": 0,
        "detection": 0,
        "prediction": 0,
        "evaluation": 0,
        "saving": 0,
        "total": 0,
        "detectors": {},
    }

    all_results = []

    # 1. Load dataset
    loading_starttime = time.time()
    print(dataset_path)
    print(os.path.join(dataset_path, dataset_path.split("/")[-2] + ".csv"))
    polluted_dataset = pd.read_csv(
        os.path.join(dataset_path, dataset_path.split("/")[-2] + ".csv"),
        keep_default_na=False,
        na_values=[""],
    )
    dmv_labels = pd.read_csv(
        os.path.join(dataset_path, dataset_path.split("/")[-2] + "_labels.csv")
    )
    target_columns = polluted_dataset.columns.to_list()
    mv_labels = polluted_dataset.isna().astype(int)

    with open(
        os.path.join(dataset_path, dataset_path.split("/")[-2] + "_types.json"), "r"
    ) as f:
        types = json.load(f)

    for col, col_type in types.items():
        if col_type == "numeric":
            polluted_dataset[col] = (
                polluted_dataset[col]
                .astype(str)
                .str.replace(",", "")
                .str.replace("%", "")
            )

    if USE_EMBEDDINGS:
        example_DMVs = {}
        with open(os.path.join(dataset_path, example_dmvs), "r") as f:
            example_DMVs = json.load(f)

        with open(os.path.join(dataset_path, example_embeddings), "r") as f:
            example_embeddings = json.load(f)

        for col in example_DMVs.keys():
            for type, values in example_DMVs[col].items():
                try:
                    example_DMVs[col][type] = [
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

            if all_embeddings is None:
                emb = LLM.embed_column(polluted_dataset[col].astype(str).tolist(), col)
                emb = np.array([e[:embedding_dim] for e in emb], dtype=np.float32)
            else:
                emb = []
                for val in polluted_dataset[col].astype(str).tolist():
                    if val in all_embeddings[col]:
                        emb.append(all_embeddings[col][val][:embedding_dim])
                    else:
                        emb.append([0.0] * embedding_dim)
                emb = np.array(emb, dtype=np.float32)
            embeddings[col] = emb
    else:
        example_DMVs = {}
        embeddings = {}

    time_measurements["loading"] += time.time() - loading_starttime

    # 2. Run detection algorithms (these produce features)
    detection_starttime = time.time()

    # This is to run Raha as competitor.
    # Raha implicitly generates its labels by comparing the dirty and clean dataset. Where both differ, it counts the cell as error.
    # Since DMVs are obviously never explicit missing values, we set them to pd.NA in the cleaned dataset to allow Raha to detect them.
    cleaned_dataset = polluted_dataset.copy()
    for col in dmv_labels.columns:
        cleaned_dataset.loc[dmv_labels[col] == 1, col] = pd.NA

    detection_results, detection_timings, assessed_columns_per_detector = (
        run_detection_algorithms(
            polluted_dataset,
            cleaned_dataset,
            detectors=config["detectors"],
            types=types,
            target_columns=target_columns,
            example_DMVs=example_DMVs,
            embeddings=embeddings,
            LLM=LLM,
        )
    )

    time_measurements["detection"] += time.time() - detection_starttime
    for detector_name, times in detection_timings.items():
        if detector_name not in time_measurements["detectors"]:
            time_measurements["detectors"][detector_name] = {}
        for step, duration in times.items():
            if step not in time_measurements["detectors"][detector_name]:
                time_measurements["detectors"][detector_name][step] = 0
            time_measurements["detectors"][detector_name][step] += duration

    # 3. Use trained models to create ensemble predictions
    if model_path is not None:
        prediction_starttime = time.time()
        detection_results = predict_with_ensemble(
            detection_results, types, trained_models
        )
        time_measurements["prediction"] = time.time() - prediction_starttime

    # 4. Evaluate all results
    evaluation_starttime = time.time()
    results = evaluate_detection_results(
        detection_results,
        mv_labels=mv_labels,
        dmv_labels=dmv_labels,
        target_columns=[
            col for col in polluted_dataset.columns if dmv_labels[col].sum() > 0
        ],
        assessed_columns_per_detector=assessed_columns_per_detector,
    )
    time_measurements["evaluation"] += time.time() - evaluation_starttime

    # 5. Save results
    saving_starttime = time.time()
    folder_name = dataset_path.split("/")[-3]
    dataset_name = dataset_path.split("/")[-2]
    for detector_name, result in detection_results.items():
        df_score, df_predict = result
        os.makedirs(
            f"{config['output_dir']}/{folder_name}/{dataset_name}/{detector_name}{suffix}/",
            exist_ok=True,
        )
        df_score.to_csv(
            f"{config['output_dir']}/{folder_name}/{dataset_name}/{detector_name}{suffix}/scores.csv",
            index=False,
        )
        df_predict.to_csv(
            f"{config['output_dir']}/{folder_name}/{dataset_name}/{detector_name}{suffix}/predictions.csv",
            index=False,
        )
    time_measurements["saving"] += time.time() - saving_starttime

    all_results = pd.DataFrame(results)

    if os.path.exists(
        f"{config['output_dir']}/{folder_name}/{dataset_name}/time_measurements.json"
    ):
        with open(
            f"{config['output_dir']}/{folder_name}/{dataset_name}/time_measurements.json",
            "r",
        ) as f:
            existing_time_measurements = json.load(f)
        if isinstance(existing_time_measurements, dict):
            time_measurements = [existing_time_measurements, time_measurements]
        else:
            time_measurements = existing_time_measurements + [time_measurements]

    with open(
        f"{config['output_dir']}/{folder_name}/{dataset_name}/time_measurements.json",
        "w",
    ) as f:
        json.dump(time_measurements, f, indent=4)

    print("Unique detectors in results:", all_results["detector"].unique())
    for detector in all_results["detector"].unique():
        print(f"Saving results for detector: {detector}")
        detector_results = all_results[all_results["detector"] == detector]
        print(
            f"Saving {len(detector_results)} to:",
            f"{config['output_dir']}/{folder_name}/{dataset_name}/{detector}{suffix}_results.csv",
        )
        detector_results.to_csv(
            f"{config['output_dir']}/{folder_name}/{dataset_name}/{detector}{suffix}_results.csv"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DISMIS - DMV detection with ensemble predictions."
    )
    parser.add_argument(
        "--evaluation_config", required=True, help="Path to evaluation config JSON file"
    )
    parser.add_argument(
        "--dataset", required=True, help="Path to dataset root directory"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to trained model"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Dimension of embeddings to use"
    )
    parser.add_argument(
        "--value_embeddings",
        type=str,
        default="_vllm_embeddings.json",
        help="Filename of value embeddings JSON file",
    )
    parser.add_argument(
        "--example_dmvs",
        type=str,
        default="example_dmvs_detection_5.json",
        help="Filename of example DMVs JSON file",
    )
    parser.add_argument(
        "--example_embeddings",
        type=str,
        default="precomputed_example_embeddings_5.json",
        help="Filename of example embeddings JSON file",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Detector suffix for output files"
    )

    args = parser.parse_args()

    dataset = args.dataset
    if dataset.startswith('"') and dataset.endswith('"'):
        dataset = dataset[1:-1]

    run_dismis(
        args.evaluation_config,
        dataset,
        model_path=args.model_path,
        value_embeddings=args.value_embeddings,
        example_dmvs=args.example_dmvs,
        example_embeddings=args.example_embeddings,
        embedding_dim=args.embedding_dim,
        suffix=args.suffix,
    )
