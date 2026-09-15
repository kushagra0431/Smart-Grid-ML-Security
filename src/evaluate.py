"""
Evaluation and Visualization Module for Forecasting and Cybersecurity Threat Detection.
Generates comprehensive performance scorecards, intrusion detection metrics,
and high-resolution visualization plots for operational security auditing.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Any
from sklearn.metrics import (
    r2_score, mean_absolute_error, root_mean_squared_error,
    confusion_matrix, classification_report, roc_curve, auc
)

from src.cyber_attacks import CyberAttackSimulator
from src.features import engineer_features, get_feature_columns


def run_evaluation(
    models_dir: str = "models",
    output_figures_dir: str = "reports/figures",
    processed_data_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Executes full evaluation pipeline:
    1. Loads trained models, metadata, and test dataset.
    2. Injects cyber attacks into the test set (FDI, DoS, Theft, Replay).
    3. Runs forecaster and cyber intrusion detector.
    4. Computes forecasting and cybersecurity metrics.
    5. Saves 5 publication-ready diagnostic plots to reports/figures/.
    """
    os.makedirs(output_figures_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    
    # 1. Load Artifacts
    forecaster = joblib.load(os.path.join(models_dir, "forecaster_model.joblib"))
    anomaly_detector = joblib.load(os.path.join(models_dir, "anomaly_detector.joblib"))
    meta = joblib.load(os.path.join(models_dir, "scaler_and_meta.joblib"))
    
    feature_cols = meta["feature_cols"]
    threshold_3sigma = meta["residual_threshold_3sigma"]
    res_mean = meta["residual_mean"]
    res_std = meta["residual_std"]
    
    # 2. Ingest raw data and extract clean test horizon
    raw_df = pd.read_csv("data/raw/electricity_consumption_raw.csv")
    raw_df["datetime"] = pd.to_datetime(raw_df["datetime"])
    
    df_feat = engineer_features(raw_df)
    test_split_idx = int(len(df_feat) * 0.80)
    clean_test_df = df_feat.iloc[test_split_idx:].copy().reset_index(drop=True)
    
    # 3. Synthesize Cybersecurity Attack Scenarios
    simulator = CyberAttackSimulator(random_seed=42)
    cyber_df = simulator.create_cyber_test_scenarios(clean_test_df)
    
    # Save processed test scenarios for API and reporting
    cyber_scenario_path = os.path.join(processed_data_dir, "cybersecurity_test_scenarios.csv")
    cyber_df.to_csv(cyber_scenario_path, index=False)
    
    # 4. Generate Predictions & Dynamic Residuals
    X_test = cyber_df[feature_cols].values
    forecast_mw = forecaster.predict(X_test)
    cyber_df["predicted_consumption_mw"] = np.round(forecast_mw, 2)
    
    # Residual calculation against incoming reported telemetry
    residuals = np.abs(cyber_df["consumption_mw"] - cyber_df["predicted_consumption_mw"])
    cyber_df["residual_mw"] = np.round(residuals, 2)
    
    # 5. Intrusion Detection Decisions
    # Rule 1: Dynamic 3-sigma statistical threshold
    threshold_flag = (residuals > threshold_3sigma).astype(int)
    
    # Rule 2: Multi-dimensional Isolation Forest score
    res_norm = (residuals - res_mean) / (res_std + 1e-5)
    load_ratio = cyber_df["load_ratio_24h"].values
    rolling_diff = ((cyber_df["lag_1"] - cyber_df["rolling_mean_24h"]) / (cyber_df["rolling_std_24h"] + 1e-5)).values
    ids_features = np.column_stack([res_norm, load_ratio, rolling_diff])
    
    if_scores = anomaly_detector.score_samples(ids_features)
    if_preds = (anomaly_detector.predict(ids_features) == -1).astype(int)
    
    # Ensemble Intrusion Decision (Union of dynamic threshold or high anomaly confidence)
    cyber_df["anomaly_score"] = np.round(-if_scores, 4)
    cyber_df["detected_attack"] = ((threshold_flag == 1) | (if_preds == 1)).astype(int)
    
    # 6. Forecasting Accuracy Metrics (evaluated on normal uncompromised periods)
    normal_mask = cyber_df["is_attack"] == 0
    y_true_normal = cyber_df.loc[normal_mask, "true_consumption_mw"].values
    y_pred_normal = cyber_df.loc[normal_mask, "predicted_consumption_mw"].values
    
    clean_r2 = r2_score(y_true_normal, y_pred_normal)
    clean_mae = mean_absolute_error(y_true_normal, y_pred_normal)
    clean_rmse = root_mean_squared_error(y_true_normal, y_pred_normal)
    clean_mape = np.mean(np.abs((y_true_normal - y_pred_normal) / y_true_normal)) * 100
    
    # 7. Cybersecurity Intrusion Detection Metrics
    y_true_cyber = cyber_df["is_attack"].values
    y_pred_cyber = cyber_df["detected_attack"].values
    
    cm = confusion_matrix(y_true_cyber, y_pred_cyber)
    tn, fp, fn, tp = cm.ravel()
    
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall / TPR
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # FPR
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0.0
    
    # Attack Type Specific Breakdown
    attack_breakdown = {}
    for code, name in simulator.ATTACK_NAMES.items():
        if code == 0:
            continue
        subset = cyber_df[cyber_df["attack_code"] == code]
        detected = subset["detected_attack"].sum()
        total = len(subset)
        attack_breakdown[name] = {
            "total_attack_hours": int(total),
            "detected_hours": int(detected),
            "recall": float(detected / total) if total > 0 else 0.0
        }

    # Consolidated Metrics Dictionary
    full_metrics = {
        "forecasting_performance": {
            "r2_score": round(float(clean_r2), 4),
            "mae_mw": round(float(clean_mae), 2),
            "rmse_mw": round(float(clean_rmse), 2),
            "mape_percent": round(float(clean_mape), 2),
            "baseline_r2": round(float(meta["baseline_r2"]), 4),
            "baseline_mae": round(float(meta["baseline_mae"]), 2)
        },
        "cybersecurity_ids_performance": {
            "detection_rate_recall": round(float(detection_rate), 4),
            "false_alarm_rate": round(float(false_alarm_rate), 4),
            "precision": round(float(precision), 4),
            "f1_score": round(float(f1), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "total_test_hours": len(cyber_df),
            "attack_hours": int(tp + fn),
            "dynamic_threshold_3sigma_mw": round(float(threshold_3sigma), 2)
        },
        "attack_type_breakdown": attack_breakdown
    }
    
    # Save metrics to JSON
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=4)
        
    print("\n" + "="*70)
    print(" [EVALUATION SCORECARD] Electricity Forecasting & Cybersecurity IDS")
    print("="*70)
    print(f" Forecasting Accuracy (Clean Telemetry):")
    print(f"  * R^2 Score:                {clean_r2:.4f}  (Baseline Ridge: {meta['baseline_r2']:.4f})")
    print(f"  * Mean Absolute Error:      {clean_mae:.2f} MW")
    print(f"  * Root Mean Squared Error:  {clean_rmse:.2f} MW")
    print(f"  * MAPE:                     {clean_mape:.2f}%")
    print(f"\n Cybersecurity Intrusion Detection System (IDS):")
    print(f"  * Attack Detection Rate:    {detection_rate:.2%} (True Positive Rate)")
    print(f"  * False Alarm Rate (FAR):   {false_alarm_rate:.2%}")
    print(f"  * IDS Precision:            {precision:.2%}")
    print(f"  * IDS F1-Score:             {f1:.4f}")
    print(f"  * Dynamic 3-Sigma Boundary: {threshold_3sigma:.2f} MW")
    print(" Attack Type Detection Breakdown:")
    for name, stats in attack_breakdown.items():
        print(f"  * {name:<20}: {stats['recall']:.1%} ({stats['detected_hours']}/{stats['total_attack_hours']} hrs detected)")
    print("="*70 + "\n")

    # 8. Generate Visualizations
    _generate_plots(cyber_df, full_metrics, threshold_3sigma, output_figures_dir, forecaster, feature_cols)
    
    return full_metrics


def _generate_plots(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    threshold_3sigma: float,
    figures_dir: str,
    forecaster: Any,
    feature_cols: list
):
    """Generates all 5 diagnostic and presentation figures."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # Color palette
    c_actual = "#1f77b4"      # Deep blue
    c_pred = "#2ca02c"        # Electric green
    c_alert = "#d62728"       # Alert crimson
    c_shade = "#ff9896"       # Soft salmon
    c_thresh = "#ff7f0e"      # Amber threshold
    
    # -------------------------------------------------------------
    # Figure 1: Actual vs Forecasted Consumption (Clean Horizon Zoom)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False, gridspec_kw={"height_ratios": [2, 1]})
    
    # Zoom on a 14-day normal window for detailed comparison
    zoom_slice = df.iloc[100:100 + 336].copy()  # 14 days = 336 hours
    
    ax1.plot(zoom_slice["datetime"], zoom_slice["true_consumption_mw"], label="Actual Grid Consumption", color=c_actual, lw=2.0)
    ax1.plot(zoom_slice["datetime"], zoom_slice["predicted_consumption_mw"], label="ML Forecaster (HistGradientBoosting)", color=c_pred, lw=1.8, ls="--")
    ax1.set_title("Smart Grid Electricity Consumption: Actual vs. ML Forecast (14-Day Horizon)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Power Demand (MW)", fontsize=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    
    # Residual error plot on zoom slice
    zoom_res = np.abs(zoom_slice["true_consumption_mw"] - zoom_slice["predicted_consumption_mw"])
    ax2.plot(zoom_slice["datetime"], zoom_res, color="purple", lw=1.5, label="Absolute Prediction Error (|y - ŷ|)")
    ax2.axhline(threshold_3sigma, color=c_thresh, ls=":", lw=2, label=f"Dynamic 3σ Boundary ({threshold_3sigma:.1f} MW)")
    ax2.set_ylabel("Residual (MW)", fontsize=12)
    ax2.set_xlabel("Timestamp (UTC)", fontsize=12)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "forecast_vs_actual.png"), dpi=300)
    plt.close(fig)
    print(" [Visualizer] Saved forecast_vs_actual.png")

    # -------------------------------------------------------------
    # Figure 2: Residual Analysis and Dynamic Anomaly Boundary
    # -------------------------------------------------------------
    normal_residuals = df.loc[df["is_attack"] == 0, "residual_mw"].values
    attack_residuals = df.loc[df["is_attack"] == 1, "residual_mw"].values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of residuals: Normal vs Cyber Attack
    ax1.hist(normal_residuals, bins=45, density=True, alpha=0.6, color=c_actual, label="Normal Operation Residuals")
    ax1.hist(attack_residuals, bins=45, density=True, alpha=0.5, color=c_alert, label="Cyber Attack Residuals")
    ax1.axvline(threshold_3sigma, color=c_thresh, lw=2.5, ls="--", label=f"3σ Cyber Alert Threshold ({threshold_3sigma:.1f} MW)")
    ax1.set_title("Forecast Residual Error Distribution (Normal vs. Attack)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Absolute Residual Error (MW)", fontsize=11)
    ax1.set_ylabel("Probability Density", fontsize=11)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Cumulative error drift
    sorted_norm_res = np.sort(normal_residuals)
    cum_prob = np.linspace(0, 1, len(sorted_norm_res))
    ax2.plot(sorted_norm_res, cum_prob, color=c_actual, lw=2.2, label="Empirical CDF (Normal)")
    ax2.axvline(threshold_3sigma, color=c_thresh, lw=2.5, ls="--", label="3σ Confidence Boundary (99.9%)")
    ax2.set_title("Cumulative Distribution Function & Confidence Level", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Residual Error (MW)", fontsize=11)
    ax2.set_ylabel("Cumulative Probability", fontsize=11)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "residual_analysis.png"), dpi=300)
    plt.close(fig)
    print(" [Visualizer] Saved residual_analysis.png")

    # -------------------------------------------------------------
    # Figure 3: Full Horizon Cyber Attack Detection & Intrusion Alerts
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})
    
    time_series = df["datetime"]
    ax1.plot(time_series, df["true_consumption_mw"], color="#6c757d", alpha=0.7, lw=1.2, label="Uncompromised Grid Ground Truth")
    ax1.plot(time_series, df["consumption_mw"], color=c_actual, lw=1.6, label="Reported Telemetry (with Cyber Injections)")
    ax1.plot(time_series, df["predicted_consumption_mw"], color=c_pred, lw=1.5, ls="--", label="ML Baseline Forecaster")
    
    # Highlight attack episodes
    attack_indices = df[df["is_attack"] == 1].index
    if len(attack_indices) > 0:
        # Group contiguous attack blocks
        diffs = np.diff(attack_indices)
        split_points = np.where(diffs > 1)[0] + 1
        blocks = np.split(attack_indices, split_points)
        for b in blocks:
            start_time = df.loc[b[0], "datetime"]
            end_time = df.loc[b[-1], "datetime"]
            attack_name = df.loc[b[0], "attack_type"]
            ax1.axvspan(start_time, end_time, color=c_shade, alpha=0.35)
            mid_time = df.loc[b[len(b)//2], "datetime"]
            ax1.text(mid_time, df["consumption_mw"].max() * 0.96, attack_name,
                     ha="center", va="top", fontsize=9, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=c_alert, alpha=0.85))
            
    ax1.set_title("Critical Infrastructure Telemetry: Real-Time Cyber Attack Injection & ML Defense", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Power Telemetry (MW)", fontsize=12)
    ax1.legend(loc="lower left", frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: ML Intrusion Detection Flags
    detected_mask = df["detected_attack"] == 1
    ax2.plot(time_series, df["residual_mw"], color="#495057", lw=1.2, label="Forecast Discrepancy (|y - ŷ|)")
    ax2.axhline(threshold_3sigma, color=c_thresh, ls="--", lw=2, label=f"3σ Alert Threshold ({threshold_3sigma:.1f} MW)")
    ax2.scatter(df.loc[detected_mask, "datetime"], df.loc[detected_mask, "residual_mw"],
                color=c_alert, s=28, marker="x", zorder=5, label="ML Intrusion Detection Alarm")
    ax2.set_ylabel("Residual (MW)", fontsize=12)
    ax2.set_xlabel("Datetime", fontsize=12)
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "cyber_attack_detection.png"), dpi=300)
    plt.close(fig)
    print(" [Visualizer] Saved cyber_attack_detection.png")

    # -------------------------------------------------------------
    # Figure 4: Feature Importance Analysis
    # -------------------------------------------------------------
    # Calculate feature importances via surrogate Random Forest or feature variance
    from sklearn.inspection import permutation_importance
    clean_sample = df.iloc[:400]
    perm = permutation_importance(forecaster, clean_sample[feature_cols].values,
                                  clean_sample["true_consumption_mw"].values,
                                  n_repeats=5, random_state=42)
    
    sorted_idx = np.argsort(perm.importances_mean)[-12:]
    top_features = [feature_cols[i] for i in sorted_idx]
    top_scores = perm.importances_mean[sorted_idx]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(top_scores)), top_scores, color="#0d6efd", edgecolor="#084298")
    ax.set_yticks(range(len(top_scores)))
    ax.set_yticklabels(top_features, fontsize=11)
    ax.set_xlabel("Permutation Feature Importance (Drop in R² Score)", fontsize=12)
    ax.set_title("Key Predictive Drivers for Grid Load Forecasting", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2, f"{w:.3f}",
                va="center", fontsize=10, color="#212529")
        
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "feature_importance.png"), dpi=300)
    plt.close(fig)
    print(" [Visualizer] Saved feature_importance.png")

    # -------------------------------------------------------------
    # Figure 5: Confusion Matrix & ROC Curve (Cybersecurity IDS)
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Confusion Matrix
    y_true = df["is_attack"].values
    y_pred = df["detected_attack"].values
    cm = confusion_matrix(y_true, y_pred)
    
    cax = ax1.matshow(cm, cmap="Blues", alpha=0.85)
    for (i, j), val in np.ndenumerate(cm):
        ax1.text(j, i, f"{val:,}\n({val/len(y_true):.1%})", ha="center", va="center",
                 fontsize=13, fontweight="bold", color="black" if val < np.max(cm)/2 else "white")
        
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["Normal (Benign)", "Cyber Attack"], fontsize=11)
    ax1.set_yticklabels(["Normal (Benign)", "Cyber Attack"], fontsize=11)
    ax1.set_xlabel("ML Model Prediction", fontsize=12, labelpad=10)
    ax1.set_ylabel("True Telemetry Status", fontsize=12)
    ax1.set_title("Intrusion Detection Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    
    # ROC Curve
    # Use normalized residual as continuous anomaly decision score
    y_scores = df["residual_mw"].values
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color=c_pred, lw=2.5, label=f"Cyber IDS Model (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="gray", lw=1.5, ls="--", label="Random Classifier (AUC = 0.50)")
    ax2.set_xlim([-0.02, 1.0])
    ax2.set_ylim([0.0, 1.02])
    ax2.set_xlabel("False Alarm Rate (False Positive Rate)", fontsize=12)
    ax2.set_ylabel("Detection Rate (True Positive Rate)", fontsize=12)
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "roc_confusion_matrix.png"), dpi=300)
    plt.close(fig)
    print(" [Visualizer] Saved roc_confusion_matrix.png")
    print(f"[Visualizer] All 5 plots successfully rendered to '{figures_dir}/'.\n")


if __name__ == "__main__":
    run_evaluation()
