"""
Model Training Pipeline for Electricity Forecasting & Cyber-Physical Anomaly Detection.
Trains a high-performance Gradient Boosted Forecaster and an Unsupervised
Isolation Forest Intrusion Detection System (IDS) on smart grid telemetry.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_or_create_dataset
from src.features import engineer_features, get_feature_columns
from src.cyber_attacks import CyberAttackSimulator


def train_models(
    models_dir: str = "models",
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end model training:
    1. Loads raw multi-year smart grid telemetry.
    2. Constructs lag, cyclic, and temporal features.
    3. Performs temporal train/test split (no lookahead leakage).
    4. Trains Gradient Boosted Forecaster and Baseline Ridge Regressor.
    5. Calibrates residual error distributions and dynamic 3-sigma thresholds.
    6. Trains an Unsupervised Isolation Forest for multi-dimensional cyber IDS.
    7. Saves all serialized artifacts to `models_dir`.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print(" [TRAINING PIPELINE] Starting Model Training & Calibration")
    print("="*70)
    
    # 1. Ingest Data
    raw_df = load_or_create_dataset()
    
    # 2. Feature Engineering
    print("[Pipeline] Engineering temporal, cyclic, and autoregressive features...")
    df_feat = engineer_features(raw_df)
    feature_cols = get_feature_columns()
    
    # 3. Chronological Train-Test Split (Strictly preserving time order)
    n_samples = len(df_feat)
    split_idx = int(n_samples * (1 - test_ratio))
    
    train_df = df_feat.iloc[:split_idx].copy()
    test_df = df_feat.iloc[split_idx:].copy()
    
    print(f"[Pipeline] Total Samples: {n_samples} | Train: {len(train_df)} | Clean Test: {len(test_df)}")
    print(f"[Pipeline] Training Date Range: {train_df['datetime'].min()} -> {train_df['datetime'].max()}")
    print(f"[Pipeline] Testing Date Range:  {test_df['datetime'].min()} -> {test_df['datetime'].max()}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df["consumption_mw"].values
    
    X_test_clean = test_df[feature_cols].values
    y_test_clean = test_df["consumption_mw"].values
    
    # 4. Standard Scaler for IDS feature space
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_clean)
    
    # 5. Train Baseline Regressor (Ridge) for performance benchmarking
    print("\n[Pipeline] Fitting Baseline Linear Ridge Regressor...")
    baseline_model = Ridge(alpha=1.0)
    baseline_model.fit(X_train_scaled, y_train)
    baseline_pred = baseline_model.predict(X_test_scaled)
    baseline_r2 = r2_score(y_test_clean, baseline_pred)
    baseline_mae = mean_absolute_error(y_test_clean, baseline_pred)
    print(f" -> Baseline Ridge R^2: {baseline_r2:.4f} | MAE: {baseline_mae:.2f} MW")
    
    # 6. Train Primary Forecaster: Histogram-based Gradient Boosting Regressor
    print("\n[Pipeline] Training High-Precision HistGradientBoosting Forecaster...")
    forecaster = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=41,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=random_state
    )
    forecaster.fit(X_train, y_train)
    
    train_pred = forecaster.predict(X_train)
    test_pred_clean = forecaster.predict(X_test_clean)
    
    forecaster_r2 = r2_score(y_test_clean, test_pred_clean)
    forecaster_mae = mean_absolute_error(y_test_clean, test_pred_clean)
    forecaster_rmse = root_mean_squared_error(y_test_clean, test_pred_clean)
    forecaster_mape = np.mean(np.abs((y_test_clean - test_pred_clean) / y_test_clean)) * 100
    
    print(f" -> HistGradientBoosting Results on Clean Test Horizon:")
    print(f"    R^2 Score: {forecaster_r2:.4f}")
    print(f"    MAE:       {forecaster_mae:.2f} MW")
    print(f"    RMSE:      {forecaster_rmse:.2f} MW")
    print(f"    MAPE:      {forecaster_mape:.2f}%")
    
    # 7. Calibrate Residual Statistics & Dynamic 3-Sigma Alert Boundaries
    train_residuals = np.abs(y_train - train_pred)
    residual_mean = float(np.mean(train_residuals))
    residual_std = float(np.std(train_residuals))
    # 3.29 sigma corresponds to 99.9% confidence boundary under normal noise
    residual_threshold_3sigma = float(residual_mean + 3.29 * residual_std)
    
    print(f"\n[Pipeline] Dynamic Residual Boundary Calibration:")
    print(f"    Mean Abs Residual (Normal): {residual_mean:.2f} MW")
    print(f"    Std Residual:              {residual_std:.2f} MW")
    print(f"    Cyber Alert Threshold (3-Sigma): {residual_threshold_3sigma:.2f} MW")
    
    # 8. Train Unsupervised Isolation Forest for Cybersecurity Intrusion Detection
    print("\n[Pipeline] Training Unsupervised Isolation Forest Cyber IDS...")
    # Features for IDS: Normalized residual, load ratio, 24h rolling difference
    res_normalized = (train_residuals - residual_mean) / (residual_std + 1e-5)
    load_ratio = train_df["load_ratio_24h"].values
    rolling_diff = ((train_df["lag_1"] - train_df["rolling_mean_24h"]) / (train_df["rolling_std_24h"] + 1e-5)).values
    
    ids_train_features = np.column_stack([res_normalized, load_ratio, rolling_diff])
    
    # Smart grid expected anomaly contamination rate (~2% baseline noise)
    anomaly_detector = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=random_state,
        n_jobs=-1
    )
    anomaly_detector.fit(ids_train_features)
    print(" -> Isolation Forest IDS successfully fitted on baseline grid telemetry.")
    
    # 9. Save Artifacts & Metadata
    forecaster_path = os.path.join(models_dir, "forecaster_model.joblib")
    ids_path = os.path.join(models_dir, "anomaly_detector.joblib")
    meta_path = os.path.join(models_dir, "scaler_and_meta.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")
    
    joblib.dump(forecaster, forecaster_path)
    joblib.dump(anomaly_detector, ids_path)
    
    metadata = {
        "feature_cols": feature_cols,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_threshold_3sigma": residual_threshold_3sigma,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "baseline_r2": baseline_r2,
        "baseline_mae": baseline_mae,
        "forecaster_r2": forecaster_r2,
        "forecaster_mae": forecaster_mae,
        "forecaster_rmse": forecaster_rmse,
        "forecaster_mape": forecaster_mape
    }
    joblib.dump(metadata, meta_path)
    
    with open(metrics_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\n[Pipeline] All models and metadata saved successfully to '{models_dir}/'.")
    print("="*70 + "\n")
    
    return {
        "forecaster": forecaster,
        "anomaly_detector": anomaly_detector,
        "metadata": metadata,
        "train_df": train_df,
        "test_df": test_df
    }


if __name__ == "__main__":
    train_models()
