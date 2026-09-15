"""
Real-Time Inference and Threat Attribution Engine for Grid Cyber Defense.
Executes single-step and batch scoring against incoming SCADA/AMI telemetry,
evaluates dynamic prediction intervals, and attributes detected anomalies to cyber attack signatures.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime


class CyberGridInferenceEngine:
    """
    Production-grade inference and intrusion detection engine for electrical grid telemetry.
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized models, metadata, and decision boundaries."""
        forecaster_path = os.path.join(self.models_dir, "forecaster_model.joblib")
        anomaly_path = os.path.join(self.models_dir, "anomaly_detector.joblib")
        meta_path = os.path.join(self.models_dir, "scaler_and_meta.joblib")

        if not os.path.exists(forecaster_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Model artifacts not found in '{self.models_dir}'. Please run train_model.py first."
            )

        self.forecaster = joblib.load(forecaster_path)
        self.anomaly_detector = joblib.load(anomaly_path) if os.path.exists(anomaly_path) else None
        self.meta = joblib.load(meta_path)
        
        self.feature_cols = self.meta["feature_cols"]
        self.threshold_3sigma = float(self.meta["residual_threshold_3sigma"])
        self.res_mean = float(self.meta["residual_mean"])
        self.res_std = float(self.meta["residual_std"])

    def predict_single_step(
        self,
        dt_str: str,
        reported_mw: float,
        lag_1: float,
        lag_2: float,
        lag_24: float,
        lag_48: float,
        lag_168: float,
        rolling_mean_6h: float,
        rolling_mean_24h: float,
        rolling_std_24h: float,
        rolling_min_24h: float,
        rolling_max_24h: float
    ) -> Dict[str, Any]:
        """
        Evaluates a single incoming telemetry point against the predictive baseline.
        """
        dt = pd.to_datetime(dt_str)
        hour = dt.hour
        dayofweek = dt.dayofweek
        dayofyear = dt.dayofyear
        month = dt.month
        is_weekend = 1 if dayofweek >= 5 else 0
        is_peak_hour = 1 if (is_weekend == 0 and (7 <= hour <= 10 or 17 <= hour <= 21)) else 0

        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12.0)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12.0)
        dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7.0)
        dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7.0)

        load_ratio_24h = np.clip(lag_1 / (rolling_mean_24h + 1e-5), 0.5, 2.0)

        features_dict = {
            "hour": hour, "dayofweek": dayofweek, "dayofyear": dayofyear, "month": month,
            "is_weekend": is_weekend, "is_peak_hour": is_peak_hour,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "month_sin": month_sin, "month_cos": month_cos,
            "dayofweek_sin": dayofweek_sin, "dayofweek_cos": dayofweek_cos,
            "lag_1": lag_1, "lag_2": lag_2, "lag_24": lag_24, "lag_48": lag_48, "lag_168": lag_168,
            "rolling_mean_6h": rolling_mean_6h, "rolling_mean_24h": rolling_mean_24h,
            "rolling_std_24h": rolling_std_24h, "rolling_min_24h": rolling_min_24h,
            "rolling_max_24h": rolling_max_24h, "load_ratio_24h": load_ratio_24h
        }

        feature_vector = np.array([[features_dict[col] for col in self.feature_cols]])
        
        # 1. Forecaster Prediction
        pred_mw = float(self.forecaster.predict(feature_vector)[0])
        residual_mw = abs(reported_mw - pred_mw)

        # 2. Threshold Check
        is_threshold_breached = residual_mw > self.threshold_3sigma

        # 3. Isolation Forest Check
        res_norm = (residual_mw - self.res_mean) / (self.res_std + 1e-5)
        rolling_diff = (lag_1 - rolling_mean_24h) / (rolling_std_24h + 1e-5)
        ids_input = np.array([[res_norm, load_ratio_24h, rolling_diff]])

        if_flag = 0
        anomaly_score = 0.0
        if self.anomaly_detector is not None:
            if_flag = int(self.anomaly_detector.predict(ids_input)[0] == -1)
            anomaly_score = float(-self.anomaly_detector.score_samples(ids_input)[0])

        is_alarm = bool(is_threshold_breached or if_flag == 1)

        # 4. Threat Attribution and Triage
        threat_type, severity, mitigation = self._attribute_threat(
            reported_mw=reported_mw,
            pred_mw=pred_mw,
            residual_mw=residual_mw,
            is_alarm=is_alarm,
            lag_1=lag_1,
            rolling_std_24h=rolling_std_24h
        )

        return {
            "timestamp": dt_str,
            "reported_mw": round(float(reported_mw), 2),
            "predicted_mw": round(pred_mw, 2),
            "residual_mw": round(residual_mw, 2),
            "threshold_3sigma_mw": round(self.threshold_3sigma, 2),
            "is_alarm": is_alarm,
            "threat_type": threat_type,
            "severity": severity,
            "anomaly_score": round(anomaly_score, 4),
            "mitigation_action": mitigation
        }

    def _attribute_threat(
        self,
        reported_mw: float,
        pred_mw: float,
        residual_mw: float,
        is_alarm: bool,
        lag_1: float,
        rolling_std_24h: float
    ) -> Tuple[str, str, str]:
        """Classifies the cyber-physical signature into operational threat categories."""
        if not is_alarm:
            return (
                "NORMAL_OPERATION",
                "LOW",
                "Telemetry within normal statistical limits. No action required."
            )

        pct_diff = (reported_mw - pred_mw) / (pred_mw + 1e-5)

        # Check for telemetry freeze / replay
        if abs(reported_mw - lag_1) < 1.0 and rolling_std_24h > 100.0:
            return (
                "TELEMETRY_REPLAY_OR_FREEZE",
                "HIGH",
                "Sensor data frozen or repeated. Alert SCADA network engineers to verify RTU communication health."
            )

        # Massive positive jump -> Coordinated DoS Load Surge
        if pct_diff > 0.30:
            return (
                "COORDINATED_DOS_LOAD_SURGE",
                "CRITICAL",
                "Severe artificial demand spike. Initiate emergency reserve dispatch, shed non-critical loads, isolate compromised IoT segments."
            )

        # Large negative jump -> Stealth Energy Theft / Meter Bypass
        if pct_diff < -0.20:
            return (
                "ENERGY_THEFT_METER_BYPASS",
                "HIGH",
                "Sustained under-reporting detected. Dispatch field inspection to inspect physical metering bypass and audit firmware hash."
            )

        # General high discrepancy -> False Data Injection (Scaling or Jitter)
        if abs(pct_diff) > 0.15:
            return (
                "FALSE_DATA_INJECTION_ATTACK",
                "CRITICAL",
                "Malicious telemetry manipulation detected. Quarantine compromised telemetry stream and switch state estimator to secondary backup."
            )

        return (
            "ANOMALOUS_GRID_DEVIATION",
            "MEDIUM",
            "Elevated prediction discrepancy. Cross-reference substation meteorological sensor feeds."
        )


if __name__ == "__main__":
    from train_model import train_models
    train_models()
    engine = CyberGridInferenceEngine()
    
    # Test sample: Normal
    res_norm = engine.predict_single_step(
        dt_str="2023-08-15 14:00:00",
        reported_mw=34200.0,
        lag_1=33900.0, lag_2=33500.0, lag_24=34000.0, lag_48=33800.0, lag_168=34100.0,
        rolling_mean_6h=33800.0, rolling_mean_24h=32500.0, rolling_std_24h=1400.0,
        rolling_min_24h=29000.0, rolling_max_24h=35000.0
    )
    print("Normal Telemetry Test Result:")
    print(res_norm)
    
    # Test sample: Cyber Attack (FDI Surge)
    res_attack = engine.predict_single_step(
        dt_str="2023-08-15 14:00:00",
        reported_mw=49000.0,  # 15,000 MW unexpected surge!
        lag_1=33900.0, lag_2=33500.0, lag_24=34000.0, lag_48=33800.0, lag_168=34100.0,
        rolling_mean_6h=33800.0, rolling_mean_24h=32500.0, rolling_std_24h=1400.0,
        rolling_min_24h=29000.0, rolling_max_24h=35000.0
    )
    print("\nCyber Attack Surge Test Result:")
    print(res_attack)
