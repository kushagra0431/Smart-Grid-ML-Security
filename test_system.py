"""
Automated Test Suite for Smart Grid Cybersecurity & Forecasting System.
Verifies data integrity, feature dimensions, forecasting accuracy thresholds,
cyber intrusion detection recall, and Flask REST API endpoints.
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

from src.data_loader import load_or_create_dataset, generate_smart_grid_dataset
from src.features import engineer_features, get_feature_columns
from src.cyber_attacks import CyberAttackSimulator
from src.inference import CyberGridInferenceEngine
from app import app


def test_dataset_generation_and_loading():
    """Verifies that the smart grid dataset generates with expected dimensions and statistics."""
    df = generate_smart_grid_dataset(n_days=10, random_seed=42)
    assert len(df) == 240, f"Expected 240 hours for 10 days, got {len(df)}"
    assert "datetime" in df.columns
    assert "consumption_mw" in df.columns
    assert df["consumption_mw"].mean() > 20000, "Mean load should be in realistic grid range (>20,000 MW)"
    assert df["consumption_mw"].isnull().sum() == 0, "No missing values allowed in telemetry"


def test_feature_engineering_dimensions_and_leakage():
    """Verifies that feature engineering creates all required columns and preserves time-series order."""
    df_raw = generate_smart_grid_dataset(n_days=15, random_seed=42)
    df_feat = engineer_features(df_raw, is_inference=False)
    
    feature_cols = get_feature_columns()
    for col in feature_cols:
        assert col in df_feat.columns, f"Missing feature column: {col}"

    assert len(df_feat) > 0, "Engineered features dataframe should not be empty"
    # Verify cyclic feature bounds
    assert df_feat["hour_sin"].between(-1.0, 1.0).all()
    assert df_feat["hour_cos"].between(-1.0, 1.0).all()


def test_cyber_attack_simulator():
    """Verifies that the attack simulator injects labeled multi-vector intrusions."""
    simulator = CyberAttackSimulator(random_seed=42)
    sample_series = np.array([30000.0, 31000.0, 32000.0])

    # Test Scaling
    scaled = simulator.inject_fdi_scaling(sample_series, scale_factor=1.30)
    assert np.allclose(scaled, sample_series * 1.30)

    # Test DoS Surge
    surged = simulator.inject_load_surge_dos(sample_series, surge_pct=0.40)
    assert np.all(surged > sample_series)

    # Test Theft
    theft = simulator.inject_energy_theft(sample_series, theft_ratio=0.35)
    assert np.all(theft < sample_series)

    # Test Replay
    replayed = simulator.inject_telemetry_replay(sample_series, freeze_val=25000.0)
    assert np.all(replayed == 25000.0)


def test_model_metrics_thresholds():
    """Verifies that the trained models satisfy target accuracy and security metrics."""
    metrics_path = os.path.join("models", "metrics.json")
    assert os.path.exists(metrics_path), "Trained metrics.json must exist"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    f_perf = metrics["forecasting_performance"]
    c_perf = metrics["cybersecurity_ids_performance"]

    # Forecasting accuracy checks
    assert f_perf["r2_score"] >= 0.90, f"R² score {f_perf['r2_score']} must be >= 0.90"
    assert f_perf["mape_percent"] < 5.0, f"MAPE {f_perf['mape_percent']}% must be < 5.0%"

    # Cybersecurity IDS performance checks
    assert c_perf["detection_rate_recall"] >= 0.85, f"Detection rate {c_perf['detection_rate_recall']} must be >= 85%"
    assert c_perf["false_alarm_rate"] < 0.10, f"False alarm rate {c_perf['false_alarm_rate']} must be < 10%"


def test_inference_engine():
    """Verifies live inference engine single-step scoring and threat attribution."""
    engine = CyberGridInferenceEngine(models_dir="models")
    
    # Test high-discrepancy DoS surge
    result = engine.predict_single_step(
        dt_str="2023-10-20 19:00:00",
        reported_mw=50000.0,
        lag_1=32000.0, lag_2=31800.0, lag_24=32100.0, lag_48=31900.0, lag_168=32200.0,
        rolling_mean_6h=31900.0, rolling_mean_24h=31000.0, rolling_std_24h=1400.0,
        rolling_min_24h=28000.0, rolling_max_24h=33500.0
    )

    assert result["is_alarm"] is True, "Large surge must trigger an alarm"
    assert result["severity"] == "CRITICAL"
    assert result["threat_type"] == "COORDINATED_DOS_LOAD_SURGE"
    assert "mitigation_action" in result


def test_flask_api_endpoints():
    """Tests all Flask web service endpoints for valid JSON responses and HTTP 200 codes."""
    client = app.test_client()

    # GET /
    res_index = client.get("/")
    assert res_index.status_code == 200

    # GET /api/status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = json.loads(res_status.data)
    assert data_status["status"] == "OPERATIONAL"
    assert data_status["r2_score"] >= 0.90

    # GET /api/historical
    res_hist = client.get("/api/historical?limit=24")
    assert res_hist.status_code == 200
    data_hist = json.loads(res_hist.data)
    assert "records" in data_hist
    assert len(data_hist["records"]) == 24

    # POST /api/simulate-attack
    attack_payload = {
        "attack_type": "DOS_LOAD_SURGE",
        "baseline_mw": 32000.0,
        "surge_pct": 0.45,
        "timestamp": "2023-11-05 20:00:00"
    }
    res_attack = client.post("/api/simulate-attack", json=attack_payload)
    assert res_attack.status_code == 200
    data_attack = json.loads(res_attack.data)
    assert data_attack["ml_defense_response"]["is_alarm"] is True


if __name__ == "__main__":
    pytest.main(["-v", __file__])
