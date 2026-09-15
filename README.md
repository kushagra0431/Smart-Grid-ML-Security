# GridGuard SOC: Smart Grid Cybersecurity & Electricity Forecasting System

An end-to-end Machine Learning and Cyber-Physical Intrusion Detection System (IDS) developed in Python. This system demonstrates how time-series forecasting, statistical residual bounds, and unsupervised anomaly detection safeguard critical infrastructure against sophisticated cyber attacks.

---

## Executive Summary & Performance Scorecard

| Domain | Metric | Baseline (Ridge) | Model (HistGradientBoosting + Isolation Forest) | Operational Significance |
|---|---|---|---|---|
| **Forecasting** | **R² Score** | `0.9567` | **`0.9856`** | Explains 98.56% of load variance with zero lookahead leakage |
| **Forecasting** | **MAE** | `605.31 MW` | **`362.68 MW`** | Low absolute error on ~30,000 MW bulk power grid baseline |
| **Forecasting** | **MAPE** | `2.14%` | **`1.22%`** | Ultra-tight operational confidence interval (<1.5%) |
| **Cybersecurity** | **Detection Recall** | `N/A` | **`94.95%`** | Detected 188 of 198 malicious attack hours |
| **Cybersecurity** | **False Alarm Rate** | `N/A` | **`4.09%`** | Minimizes alert fatigue for SOC operators |
| **Cybersecurity** | **IDS Precision** | `N/A` | **`58.39%`** | High positive predictive value in contaminated telemetry |
| **Cybersecurity** | **ROC-AUC** | `N/A` | **`0.9793`** | Exceptional discriminative capability across all attack vectors |

---

## Cybersecurity in Modern Smart Grids

In Critical Infrastructure Protection (NIST SP 800-82, NERC CIP), electrical grids rely on Supervisory Control and Data Acquisition (SCADA), Remote Terminal Units (RTUs), and Advanced Metering Infrastructure (AMI). Modern grids face severe cyber threats:

1. **False Data Injection Attacks (FDIA)**: Attackers compromise communication links between substations and state estimators to inject false telemetry, deceiving automatic generation control (AGC) or economic dispatch.
2. **Coordinated Load Surge (IoT Botnet DoS)**: Adversaries hijack high-wattage IoT devices (EV fast chargers, smart HVAC units) to synchronize sudden demand spikes, destabilizing grid frequency and tripping protective breakers.
3. **Stealthy Energy Theft / Meter Tampering**: Malicious consumers or insider threats modify smart meter firmware to artificially scale down reported consumption.
4. **Telemetry Replay & Freezing Attacks**: Attackers capture and replay stale historical telemetry to blind operators to active physical sabotage.

### ML-Driven Defense Architecture
Rather than relying on static rules or simple bounds, this system deploys a **Predictive Baseline Forecaster**:
$$\hat{y}_t = f(X_t)$$
Incoming telemetry is compared against $\hat{y}_t$ to compute dynamic residuals:
$$e_t = |y_{\text{telemetry}, t} - \hat{y}_t|$$
Intrusions are flagged when $e_t$ exceeds an adaptive statistical $3\sigma$ threshold or triggers an unsupervised **Isolation Forest** feature-space anomaly alarm:
$$e_t > \mu_e + 3.29 \cdot \sigma_e$$

---

## Dataset Choice & Rationale

- **Source Reference**: Modeled after the benchmark **PJM Interconnection Hourly Energy Consumption dataset** (one of the largest Regional Transmission Organizations in North America).
- **Sample Size**: **17,520 hourly readings** spanning 2 full calendar years (730 continuous days).
- **Physical Characteristics Modeled**:
  - *Base Grid Load*: 28,000 - 32,000 MW.
  - *Diurnal Profiles*: Bimodal morning ramp (07:00-09:00) and evening peak (18:00-21:00) with overnight troughs (02:00-05:00).
  - *Weekly Seasonality*: Commercial/industrial weekday load vs. weekend drops (-12% Saturday, -18% Sunday).
  - *Annual Seasonality*: Dual peak representing summer air-conditioning load and winter electric heating.
  - *Weather Shock Waves*: Multi-day synoptic weather shifts and high-frequency Gaussian micro-fluctuations.
- **Why this dataset is appropriate**: Electrical grid time-series exhibit rich non-linearities, cyclic periodicity, and autoregressive continuity, making it the premier benchmark for evaluating whether an anomaly stems from normal weather fluctuations or an adversarial cyber intrusion.

---

## Project Structure

```
Machine Learning Project/
├── data/
│   ├── raw/
│   │   └── electricity_consumption_raw.csv      # 17,520 hourly SCADA telemetry records
│   └── processed/
│       └── cybersecurity_test_scenarios.csv     # Evaluated test set with labeled cyber attacks
├── models/
│   ├── forecaster_model.joblib                  # Trained HistGradientBoosting forecaster
│   ├── anomaly_detector.joblib                  # Trained Isolation Forest cyber IDS
│   ├── scaler_and_meta.joblib                   # Scaler, feature column order, 3σ threshold
│   └── metrics.json                             # Recorded quantitative performance metrics
├── reports/
│   └── figures/
│       ├── forecast_vs_actual.png               # Actual vs ML predictions (14-day zoom)
│       ├── residual_analysis.png                # Residual distribution & 3σ boundary
│       ├── cyber_attack_detection.png           # Full horizon attack detection timeline
│       ├── feature_importance.png               # Permutation importance of predictive drivers
│       └── roc_confusion_matrix.png             # Intrusion detection ROC curve & confusion matrix
├── src/
│   ├── __init__.py                              # Package initialization
│   ├── data_loader.py                           # Telemetry generation and ingestion
│   ├── cyber_attacks.py                         # Threat models (FDI, DoS, Theft, Replay)
│   ├── features.py                              # Lag, cyclic (sin/cos), and rolling statistics
│   ├── train_model.py                           # Model training and calibration pipeline
│   ├── evaluate.py                              # Metrics computation and figure rendering
│   └── inference.py                             # Real-time scoring and threat attribution engine
├── static/
│   ├── index.html                               # Dark-mode Cyber Defense SOC Dashboard
│   ├── style.css                                # Modern CSS design system with glowing accents
│   └── app.js                                   # Dynamic charting, attack simulator, and event log
├── app.py                                       # Flask REST API and web application server
├── run_pipeline.py                              # Single-command master orchestrator
├── test_system.py                               # Comprehensive pytest verification suite
├── requirements.txt                             # Python dependencies
└── README.md                                    # Comprehensive User Guide
```

---

## Step-by-Step Instructions: Running the Project in Antigravity

### 1. Prerequisites & Environment
Ensure Python 3.10+ is installed on your system. All required dependencies are listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Execution Order

#### Step 1: Execute the End-to-End Pipeline
Run the master pipeline script to generate telemetry data, engineer features, train the forecaster and anomaly detector, compute evaluation metrics, render all 5 diagnostic figures, and perform self-tests:
```bash
python run_pipeline.py
```
*Expected duration: ~25-35 seconds.*

#### Step 2: Run the Automated Verification Suite
Verify that all unit tests, model thresholds ($R^2 \ge 0.90$, recall $\ge 85\%$), and API routes pass:
```bash
python -m pytest test_system.py -v
```
*All 6 tests should pass with exit code 0.*

#### Step 3: Launch the Interactive Web Dashboard
Deploy the trained model and start the Cyber Defense Operations Center:
```bash
python app.py
```
The server will bind to:
```
http://127.0.0.1:5000
```

---

## How to Interact with the Trained Model on the Platform

1. **Open the Web Dashboard**:
   Navigate to `http://127.0.0.1:5000` in your web browser.
2. **Explore the Live Telemetry Stream**:
   - Use the time horizon buttons (**72 Hours**, **7 Days**, **14 Days**, **30 Days**) to inspect SCADA telemetry against the ML baseline.
   - Observe the crimson markers highlighting where cyber attacks breached the adaptive $3\sigma$ confidence interval.
3. **Simulate a Live Cyber Attack**:
   - Scroll to the **Interactive Cyber Attack Injection Console**.
   - Select an attack vector from the dropdown:
     * *False Data Injection: Multiplicative Scaling (+35%)*
     * *False Data Injection: High-Variance Jitter Noise (25% std)*
     * *Coordinated IoT Botnet: Demand Surge (+40%)*
     * *Stealth Energy Theft: Smart Meter Tampering (-35%)*
     * *Replay Attack: Sensor Telemetry Freezing*
   - Adjust the **Baseline Load** (e.g., 32,000 MW) and **Attack Magnitude** sliders.
   - Click **"Launch Cyber Attack Simulation"**.
   - Review the **ML Defense Triage Card**:
     * Real-time reported load vs ML predicted baseline.
     * Residual error spike and threshold ratio ($>10\times$ dynamic limit).
     * Classified threat attribution and severity rating (**CRITICAL** / **HIGH**).
     * Automated mitigation and countermeasure recommendations.
4. **Audit Diagnostic Figures**:
   - In the **Model Diagnostic & Cybersecurity Audit Gallery**, toggle between tabs to view the high-resolution charts generated directly from model evaluation.
5. **Inspect the Incident Log**:
   - Review recent audited SCADA events in the NIST SP 800-82 compliance log table at the bottom of the dashboard.

---

## REST API Reference

The deployed system exposes high-performance JSON endpoints:

- `GET /api/status`: System health, model version, and quantitative accuracy scores.
- `GET /api/metrics`: Full evaluation metrics and attack-specific breakdown.
- `GET /api/historical?limit=168&offset=0`: Telemetry streams for charting.
- `POST /api/predict`: Live single-step point inference.
  ```json
  {
    "timestamp": "2023-11-15 19:00:00",
    "reported_mw": 34200.0,
    "lag_1": 33900.0,
    "lag_2": 33500.0,
    "lag_24": 34000.0,
    "lag_48": 33800.0,
    "lag_168": 34100.0,
    "rolling_mean_6h": 33800.0,
    "rolling_mean_24h": 32500.0,
    "rolling_std_24h": 1400.0,
    "rolling_min_24h": 29000.0,
    "rolling_max_24h": 35000.0
  }
  ```
- `POST /api/simulate-attack`: Injects real-time attacks and returns ML intrusion triage.
  ```json
  {
    "attack_type": "DOS_LOAD_SURGE",
    "baseline_mw": 32000.0,
    "surge_pct": 0.40,
    "timestamp": "2023-11-15 19:00:00"
  }
  ```

---

## How to Interpret the Forecasting and Cybersecurity Results

### 1. Forecasting Metrics
- **$R^2 = 0.9856$**: Indicates that 98.56% of hourly electricity consumption variance is accounted for by the temporal, cyclic, and autoregressive feature representations. The remaining ~1.4% corresponds to uncoordinated stochastic residential switching and micro-weather noise.
- **$\text{MAE} = 362.68 \text{ MW}$**: On an average grid load of ~30,000 MW, an average deviation of 362 MW represents a **1.22% Mean Absolute Percentage Error (MAPE)**, providing grid dispatchers with an exceptionally tight operating envelope.

### 2. Cybersecurity Detection Metrics
- **Detection Recall ($94.95\%$)**: The system successfully intercepted 188 out of 198 injected cyber attack hours across all 5 threat categories:
  - `FDI_SCALING`: **100.0%** (36/36 hours detected)
  - `DOS_LOAD_SURGE`: **100.0%** (24/24 hours detected)
  - `ENERGY_THEFT`: **100.0%** (60/60 hours detected)
  - `FDI_JITTER`: **95.8%** (46/48 hours detected)
  - `TELEMETRY_REPLAY`: **73.3%** (22/30 hours detected)
- **False Alarm Rate ($4.09\%$)**: During standard grid operations with zero attacks, the system triggers fewer than 4 false alerts per 100 operating hours.
- **Dynamic $3\sigma$ Threshold ($1,057.35 \text{ MW}$)**: Calibrated from empirical normal training residuals ($\mu_e = 298.66\text{ MW}, \sigma_e = 230.60\text{ MW}$). Discrepancies exceeding $1,057 \text{ MW}$ have less than a 0.1% probability of being benign noise.
