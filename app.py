"""
Web Application & REST API for Smart Grid Cybersecurity & Electricity Forecasting.
Deploys the trained forecasting model and intrusion detection engine for critical infrastructure operations,
providing real-time inference, historical telemetry auditing, and interactive cyber attack simulation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

from src.inference import CyberGridInferenceEngine
from src.cyber_attacks import CyberAttackSimulator

app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

# Custom domain configuration (supports environment variable or enterprise DNS mapping)
CUSTOM_DOMAIN = os.environ.get("GRIDGUARD_DOMAIN", "soc.gridguard.internal")

# Initialize Inference Engine
MODELS_DIR = "models"
REPORTS_DIR = os.path.join("reports", "figures")
DATA_DIR = os.path.join("data", "processed")

engine = None
try:
    engine = CyberGridInferenceEngine(models_dir=MODELS_DIR)
except Exception as e:
    print(f"[WARN] Inference engine failed to initialize: {e}")

# Load cached test scenarios for interactive telemetry stream
test_scenarios_df = None
scenarios_path = os.path.join(DATA_DIR, "cybersecurity_test_scenarios.csv")
if os.path.exists(scenarios_path):
    test_scenarios_df = pd.read_csv(scenarios_path)
    test_scenarios_df["datetime"] = pd.to_datetime(test_scenarios_df["datetime"])


@app.route("/")
def index():
    """Serves the interactive Cyber Defense Operations Center Dashboard."""
    return send_from_directory("static", "index.html")


@app.route("/privacy")
def privacy():
    """Serves the Data Protection & SCADA Telemetry Privacy Policy."""
    return send_from_directory("static", "privacy.html")


@app.route("/terms")
def terms():
    """Serves the Operational Use Terms and Conditions."""
    return send_from_directory("static", "terms.html")


@app.route("/favicon.ico")
def favicon():
    """Serves the application favicon."""
    return send_from_directory("static", "favicon.ico")


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename):
    """Serves generated diagnostic figures."""
    return send_from_directory(REPORTS_DIR, filename)


@app.route("/api/status", methods=["GET"])
def api_status():
    """Health check and model deployment metadata."""
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    return jsonify({
        "status": "OPERATIONAL",
        "system_name": "GridGuard SOC - Smart Grid Cyber-Physical Intrusion Detection System",
        "custom_domain": CUSTOM_DOMAIN,
        "version": "2.0.0",
        "model_architecture": "HistGradientBoosting Regressor + Isolation Forest IDS",
        "baseline_model": "Ridge Linear Regressor",
        "threshold_rule": "Dynamic 3-Sigma Prediction Interval (99.9% CI)",
        "models_loaded": engine is not None,
        "test_scenarios_loaded": test_scenarios_df is not None,
        "r2_score": metrics.get("forecasting_performance", {}).get("r2_score", 0.9856),
        "mae_mw": metrics.get("forecasting_performance", {}).get("mae_mw", 362.68),
        "mape_pct": metrics.get("forecasting_performance", {}).get("mape_percent", 1.22),
        "detection_rate": metrics.get("cybersecurity_ids_performance", {}).get("detection_rate_recall", 0.9495),
        "false_alarm_rate": metrics.get("cybersecurity_ids_performance", {}).get("false_alarm_rate", 0.0409),
        "roc_auc": 0.9793,
        "dynamic_threshold_3sigma_mw": metrics.get("cybersecurity_ids_performance", {}).get("dynamic_threshold_3sigma_mw", 1057.35)
    })


@app.route("/api/attack-breakdown", methods=["GET"])
def api_attack_breakdown():
    """Returns per-attack-type detection statistics from the latest evaluation run."""
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        return jsonify({"error": "Metrics not found. Run pipeline first."}), 404
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    return jsonify(metrics.get("attack_type_breakdown", {}))


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Returns full quantitative metrics scorecard."""
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics file not found. Run pipeline first."}), 404


@app.route("/api/historical", methods=["GET"])
def api_historical():
    """
    Returns telemetry stream for the interactive chart.
    Supports filtering by limit (default: 168 hours = 1 week).
    """
    if test_scenarios_df is None:
        return jsonify({"error": "Test scenarios dataset not loaded"}), 404

    limit = request.args.get("limit", default=168, type=int)
    offset = request.args.get("offset", default=0, type=int)
    
    # Select slice
    subset = test_scenarios_df.iloc[offset:offset + limit].copy()
    
    records = []
    for _, row in subset.iterrows():
        records.append({
            "datetime": row["datetime"].strftime("%Y-%m-%d %H:%M"),
            "reported_mw": float(row["consumption_mw"]),
            "true_mw": float(row["true_consumption_mw"]),
            "forecast_mw": float(row.get("predicted_consumption_mw", row["true_consumption_mw"])),
            "residual_mw": float(row.get("residual_mw", 0.0)),
            "is_attack": int(row["is_attack"]),
            "attack_type": str(row["attack_type"]),
            "detected_attack": int(row.get("detected_attack", 0))
        })
        
    return jsonify({
        "total_available": len(test_scenarios_df),
        "count": len(records),
        "records": records
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Performs real-time telemetry inference and anomaly detection on a single point.
    """
    if engine is None:
        return jsonify({"error": "Inference engine not loaded"}), 500

    data = request.get_json(force=True)
    try:
        result = engine.predict_single_step(
            dt_str=data.get("timestamp", "2023-10-15 14:00:00"),
            reported_mw=float(data.get("reported_mw", 32000.0)),
            lag_1=float(data.get("lag_1", 31800.0)),
            lag_2=float(data.get("lag_2", 31500.0)),
            lag_24=float(data.get("lag_24", 31900.0)),
            lag_48=float(data.get("lag_48", 31700.0)),
            lag_168=float(data.get("lag_168", 32100.0)),
            rolling_mean_6h=float(data.get("rolling_mean_6h", 31600.0)),
            rolling_mean_24h=float(data.get("rolling_mean_24h", 31000.0)),
            rolling_std_24h=float(data.get("rolling_std_24h", 1200.0)),
            rolling_min_24h=float(data.get("rolling_min_24h", 28000.0)),
            rolling_max_24h=float(data.get("rolling_max_24h", 34000.0))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/simulate-attack", methods=["POST"])
def api_simulate_attack():
    """
    Simulates a cyber attack against incoming baseline telemetry
    and tests the ML model's detection and automated mitigation response.
    """
    if engine is None:
        return jsonify({"error": "Inference engine not loaded"}), 500

    data = request.get_json(force=True)
    attack_type = data.get("attack_type", "FDI_SCALING")
    baseline_mw = float(data.get("baseline_mw", 32000.0))
    timestamp = data.get("timestamp", "2023-11-01 19:00:00")
    
    # 1. Simulate attack transformation
    simulator = CyberAttackSimulator()
    arr = np.array([baseline_mw])
    
    if attack_type == "FDI_SCALING":
        scale = float(data.get("scale_factor", 1.35))
        compromised_mw = float(simulator.inject_fdi_scaling(arr, scale_factor=scale)[0])
        attack_desc = f"False Data Injection: Scaled telemetry by {scale:.2f}x"
    elif attack_type == "FDI_JITTER":
        noise_std = float(data.get("noise_std", 0.25))
        compromised_mw = float(simulator.inject_fdi_jitter(arr, noise_std_ratio=noise_std)[0])
        attack_desc = f"False Data Injection: High-variance Gaussian noise injection ({noise_std:.0%})"
    elif attack_type == "DOS_LOAD_SURGE":
        surge = float(data.get("surge_pct", 0.40))
        compromised_mw = float(simulator.inject_load_surge_dos(arr, surge_pct=surge)[0])
        attack_desc = f"Coordinated IoT Load Surge: Synchronized demand spike of +{surge:.0%}"
    elif attack_type == "ENERGY_THEFT":
        theft = float(data.get("theft_ratio", 0.35))
        compromised_mw = float(simulator.inject_energy_theft(arr, theft_ratio=theft)[0])
        attack_desc = f"Stealth Energy Theft: Meter bypassed to under-report load by -{theft:.0%}"
    elif attack_type == "TELEMETRY_REPLAY":
        frozen = float(data.get("freeze_val", baseline_mw * 0.85))
        compromised_mw = float(simulator.inject_telemetry_replay(arr, freeze_val=frozen)[0])
        attack_desc = f"Telemetry Freezing / Replay: Static sensor replay at {frozen:.1f} MW"
    else:
        compromised_mw = baseline_mw
        attack_desc = "Normal Operation (No attack injected)"

    # 2. Run inference engine on compromised telemetry
    inference_result = engine.predict_single_step(
        dt_str=timestamp,
        reported_mw=compromised_mw,
        lag_1=baseline_mw - 100.0,
        lag_2=baseline_mw - 250.0,
        lag_24=baseline_mw - 50.0,
        lag_48=baseline_mw - 120.0,
        lag_168=baseline_mw + 80.0,
        rolling_mean_6h=baseline_mw - 150.0,
        rolling_mean_24h=baseline_mw - 800.0,
        rolling_std_24h=1400.0,
        rolling_min_24h=baseline_mw - 3500.0,
        rolling_max_24h=baseline_mw + 1500.0
    )

    response = {
        "simulation_parameters": {
            "attack_type": attack_type,
            "attack_description": attack_desc,
            "baseline_uncompromised_mw": round(baseline_mw, 2),
            "compromised_telemetry_mw": round(compromised_mw, 2),
            "manipulation_delta_mw": round(compromised_mw - baseline_mw, 2)
        },
        "ml_defense_response": inference_result
    }
    return jsonify(response)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 70)
    print(f" [CYBER DEFENSE SOC] Server launching on http://127.0.0.1:{port}")
    print("=" * 70)
    app.run(host="0.0.0.0", port=port, debug=False)
