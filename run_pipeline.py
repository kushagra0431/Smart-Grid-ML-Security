"""
Master Orchestration Script for Electricity Forecasting & Cyber-Physical Defense System.
Executes the full pipeline end-to-end:
1. Data Ingestion & Synthesis (Smart Grid Telemetry)
2. Feature Engineering & Chronological Train/Test Partitioning
3. Model Training (Gradient Boosted Forecaster & Isolation Forest IDS)
4. Evaluation, Cyber-Attack Scenario Testing, and Plot Generation
5. Live Inference Engine Self-Test
"""

import os
import sys
import time
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_or_create_dataset
from src.train_model import train_models
from src.evaluate import run_evaluation
from src.inference import CyberGridInferenceEngine


def main():
    print("=" * 80)
    print("      SMART GRID CYBERSECURITY & ELECTRICITY FORECASTING SYSTEM")
    print("         Critical Infrastructure Intrusion Detection Pipeline")
    print("=" * 80)
    start_time = time.time()

    # Step 1: Ingest Data
    print("\n[STEP 1/5] Ingesting Smart Grid Telemetry Data...")
    df_raw = load_or_create_dataset()
    print(f" -> Telemetry records loaded: {len(df_raw):,} hourly samples.")
    print(f" -> Date span: {df_raw['datetime'].min()} to {df_raw['datetime'].max()}")

    # Step 2 & 3: Model Training
    print("\n[STEP 2/5] Training Forecasting and Cybersecurity Models...")
    train_results = train_models(models_dir="models")
    meta = train_results["metadata"]
    print(f" -> Forecaster R^2: {meta['forecaster_r2']:.4f} | MAE: {meta['forecaster_mae']:.2f} MW")
    print(f" -> Dynamic 3-Sigma Boundary: {meta['residual_threshold_3sigma']:.2f} MW")

    # Step 4: Comprehensive Evaluation & Figure Generation
    print("\n[STEP 3/5] Evaluating Cyber Threat Scenarios & Generating Diagnostic Plots...")
    eval_metrics = run_evaluation(
        models_dir="models",
        output_figures_dir="reports/figures",
        processed_data_dir="data/processed"
    )

    # Step 5: Real-Time Inference Self-Test
    print("\n[STEP 4/5] Validating Real-Time Inference Engine...")
    engine = CyberGridInferenceEngine(models_dir="models")
    
    # Test 1: Benign Telemetry
    sample_normal = engine.predict_single_step(
        dt_str="2023-09-01 18:00:00",
        reported_mw=36500.0,
        lag_1=36200.0, lag_2=35800.0, lag_24=36100.0, lag_48=35900.0, lag_168=36400.0,
        rolling_mean_6h=35900.0, rolling_mean_24h=33800.0, rolling_std_24h=1600.0,
        rolling_min_24h=30000.0, rolling_max_24h=37000.0
    )
    print(" -> Validation Test 1 [Benign Load]:", sample_normal["threat_type"], f"(Alarm={sample_normal['is_alarm']})")
    
    # Test 2: Injected DoS Load Surge
    sample_surge = engine.predict_single_step(
        dt_str="2023-09-01 18:00:00",
        reported_mw=52000.0,  # Massive 16,000 MW surge
        lag_1=36200.0, lag_2=35800.0, lag_24=36100.0, lag_48=35900.0, lag_168=36400.0,
        rolling_mean_6h=35900.0, rolling_mean_24h=33800.0, rolling_std_24h=1600.0,
        rolling_min_24h=30000.0, rolling_max_24h=37000.0
    )
    print(" -> Validation Test 2 [DoS Attack]: ", sample_surge["threat_type"], f"(Alarm={sample_surge['is_alarm']}, Severity={sample_surge['severity']})")

    # Step 5: Execution Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(" [EXECUTION COMPLETE] Pipeline finished successfully in {:.2f} seconds".format(elapsed))
    print(" Generated Artifacts:")
    print("   * Models:   models/forecaster_model.joblib, models/anomaly_detector.joblib")
    print("   * Metadata: models/scaler_and_meta.joblib, models/metrics.json")
    print("   * Scenarios: data/processed/cybersecurity_test_scenarios.csv")
    print("   * Figures:  reports/figures/forecast_vs_actual.png")
    print("               reports/figures/residual_analysis.png")
    print("               reports/figures/cyber_attack_detection.png")
    print("               reports/figures/feature_importance.png")
    print("               reports/figures/roc_confusion_matrix.png")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
