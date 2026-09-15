# ⚡ Simplified Student Guide: GridGuard SOC
### Machine Learning in Cybersecurity for Electricity Consumption Forecasting

Welcome! If you feel overwhelmed by the technical jargon, don't worry. This guide breaks down the entire project in plain English, step-by-step, using simple real-world analogies.

---

## 1. What is the Big Idea? (The 10-Second Summary)

Imagine a power company monitoring electricity across an entire city. 

1. **Section 1: Only Relevant Operational Data**:
   - Power usage follows human habits (waking up, commercial hours, nighttime sleep).
   - Our ML model (`HistGradientBoosting`) learns this baseline and predicts normal MW demand with a **3-Sigma allowable tolerance limit**.
   - As long as incoming telemetry stays within this bound, the system reports **[SAFE / NORMAL - No Alarm]**.

2. **Section 2: Security Testing & Attack Interception**:
   - When any cyber attack is simulated (e.g., IoT Botnet DoS surge, False Data Injection, or Energy Theft), the difference between the attacked telemetry and the ML prediction explodes.
   - The system immediately triggers a **SIREN WARNING ALARM 🚨**.
   - The trained model **stops the attack**, quarantines fake data, and **pops up a window** detailing:
     - **Attack Type** (e.g., `COORDINATED_DOS_LOAD_SURGE`, `ENERGY_THEFT`)
     - **What Happened** (Why the attacker did it)
     - **How the ML Model Stopped It** (How the forecaster and 3-Sigma threshold rejected the tampered telemetry)
     - **System Mitigation Action** (Automated incident playbook)

---

## 2. Real-World Analogy: The Bank & The Credit Card

Think of this like **credit card fraud detection**:
- Your bank knows your daily spending habits (coffee in the morning, groceries on weekends).
- If your card suddenly charges \$5,000 at 3:00 AM in another country, the bank flags it as suspicious because it drastically violates your expected baseline.
- In this project:
  - **Your Spending Pattern** = Normal Electricity Usage.
  - **The Bank's Brain** = Our ML Forecasting Model (`HistGradientBoosting`).
  - **The Fraud Alert** = Our Intrusion Detection System (`3-Sigma Threshold` + `Isolation Forest`).

---

## 3. The 3 Building Blocks of the Project

```mermaid
flowchart LR
    A[1. Power Grid Data\n17,520 hours of readings] --> B[2. ML Forecaster\nPredicts expected load]
    C[Hacker Attacks\nInjected falsified readings] --> D[Calculate Difference / Residual\n|Actual - Predicted|]
    B --> D
    D --> E{Is difference too big?}
    E -- Yes --> F[🚨 Cyber Alarm Triggered\nAttack Detected!]
    E -- No --> G[✅ Normal Operation]
```

### Block 1: Electricity Data & Feature Engineering (`src/data_loader.py` & `src/features.py`)
To predict electricity, we give the model smart clues (features):
- **Cyclic Time Clues**: Hour of the day, day of the week, month. (We use sine & cosine so the model understands that hour 23 and hour 0 are right next to each other).
- **Lag Features ("Memory")**:
  - What was the load 1 hour ago? ($t-1$)
  - What was the load exactly at this time yesterday? ($t-24$)
  - What was the load at this time last week? ($t-168$)
- **Rolling Averages**: Average power over the last 6 hours and 24 hours (smooths out noise).

---

### Block 2: The ML Brain (`src/train_model.py`)
We use two simple, powerful algorithms:

1. **Forecaster (`HistGradientBoostingRegressor`)**:
   - Like a super-smart decision tree team.
   - It looks at the time, weather, and yesterday's power, and predicts: *"Right now, power should be around 31,500 MW."*
   - Achieves **98.5% accuracy ($R^2 = 0.985$)**.

2. **Error Residual ($e_t$)**:
   - We subtract: $\text{Residual} = |\text{Reported Reading} - \text{Predicted Reading}|$
   - Under normal conditions, the residual is small (just normal small fluctuations).

3. **Cyber Anomaly Detector (`Isolation Forest` & 3-Sigma Rule)**:
   - **3-Sigma Rule ($\mu + 3\sigma$)**: If the difference between reality and the ML prediction is more than 3 standard deviations away from normal, it's mathematically abnormal ($>99.7\%$ unlikely by chance).
   - **Isolation Forest**: An unsupervised algorithm that isolates weird data points. Because anomalous points are rare and different, they get separated in very few tree splits.

---

### Block 3: The 4 Cyber Attacks We Simulate (`src/cyber_attacks.py`)

To prove our ML works, we test it against real hacker strategies:

| Attack Name | What the Hacker Does | Why It Is Dangerous | How our ML Catches It |
| :--- | :--- | :--- | :--- |
| **1. FDI Scaling (False Data Injection)** | Multiplies reported load by $1.35\times$ | Tricks grid operators into over-generating electricity, wasting millions | The ML model predicts 30,000 MW, but the meter says 40,500 MW $\rightarrow$ Big Residual $\rightarrow$ Alert! |
| **2. DoS Load Surge** | Hacks thousands of smart EV chargers/ACs to turn on all at once ($+45\%$ jump) | Can trip circuit breakers and cause physical blackouts | Sudden spike deviates far above the 24h rolling trend $\rightarrow$ Alert! |
| **3. Energy Theft** | Hacks meter firmware to report $40\%$ less consumption | The customer steals power; utility loses revenue | Meter shows a trough during peak daytime business hours $\rightarrow$ Alert! |
| **4. Telemetry Replay / Freezing** | Freezes the sensor reading so it repeats yesterday's numbers | Blinds operators while hackers tamper with physical substations | Real grid has natural micro-fluctuations; a flat line triggers the anomaly detector $\rightarrow$ Alert! |

---

## 4. File-by-File Guide (Where Everything Lives)

Here is your map of the codebase so you never feel lost:

```
Machine Learning Project/
│
├── run_pipeline.py          ⭐ START HERE! One script that runs the entire ML pipeline.
│
├── src/                     📁 The core Python logic:
│   ├── data_loader.py       - Loads the hourly power consumption data.
│   ├── features.py          - Creates the ML features (lags, rolling averages, sine/cosine).
│   ├── cyber_attacks.py     - Code for the 4 hacker attacks.
│   ├── train_model.py       - Trains the Gradient Boosting model & Isolation Forest.
│   ├── evaluate.py          - Generates graphs and calculates detection accuracy (94.9%).
│   └── inference.py         - Scores new incoming readings in real-time.
│
├── app.py                   🌐 Flask web server that powers the interactive dashboard.
├── static/                  🎨 The web dashboard interface (HTML/CSS/JS).
│
├── reports/figures/         📊 Visual charts (residual plots, detection timelines, ROC curves).
└── test_system.py           🧪 Unit tests to ensure everything runs without errors.
```

---

## 5. Live Demonstration Guide: Showing How Your Project Works

Follow these exact steps to demonstrate your project to a teacher, examiner, or colleague:

### 🎬 Method A: The 1-Minute Console Demonstration
Open your terminal and run:
```powershell
python demo_quickstart.py
```
**What to point out:**
1. **Section 1 (Normal Operations)**: Point out how incoming telemetry (e.g. 32,150 MW) closely matches the ML prediction (31,111 MW). Show that the gap (1,039 MW) is strictly within the allowable $3\sigma$ threshold (1,057 MW), giving a **[SAFE / NORMAL]** status.
2. **Section 2 (Cyber Attack Simulation)**: 
   - Watch the **SIREN WARNING ALARM** trigger for 3 attacks (IoT Botnet DoS, FDI Scaling, Energy Theft).
   - Point out how the console immediately explains **WHAT HAPPENED** and **HOW THE ML MODEL STOPPED IT**.

---

### 🌐 Method B: The Interactive Web SOC Platform
With the server running (`python app.py`):
1. Open your browser at: **`http://localhost:5000`**
2. **Show Key Performance Indicators (Top Row)**:
   - Forecaster Accuracy: **98.56% $R^2$**
   - Cyber Attack Detection Rate: **94.95% Recall** (catches ~95 out of 100 attacks)
   - False Alarm Rate: **4.09%**
3. **Show Live Telemetry Stream (Chart)**:
   - Explain the green dashed line (what the ML model expects) versus the blue line (actual SCADA meter data).
4. **Trigger a Cyber Attack (Interactive Console)**:
   - Scroll down to the **"Interactive Cyber Attack Injection Console"**.
   - Select **"Coordinated IoT Botnet — Demand Surge (+40%)"** or **"Stealth Energy Theft (−35%)"**.
   - Click the red button: **"⚡ Launch Cyber Attack Simulation"**.
   - **Show the Pop-up Alert Modal**: The red siren window appears on screen displaying:
     - 🚨 `CYBER ATTACK INTERCEPTED!`
     - The exact attack category & severity
     - A clear explanation of the hacker's attempt
     - How the ML model halted the malicious data to protect the power grid.
5. **Show Model Diagnostic Proofs**:
   - Scroll to the bottom to switch tabs and display the **Confusion Matrix (ROC-AUC 0.979)**, **Residual Error Distribution**, and **Feature Importance** plots.

---

## 6. How to Explain This in an Exam, Interview, or Presentation

If anyone asks you: *"Explain your project"*, here is your winning 4-sentence answer:

> *"In smart grids, sensors can be hacked using False Data Injection or Botnet DoS surges to destabilize the power grid. In this project, I built a machine learning cybersecurity defense system. First, I trained a Gradient Boosted model on time-series telemetry to learn expected electricity demand patterns with a 98.5% $R^2$ accuracy. Then, I combined statistical residual thresholds ($3\sigma$) and an Isolation Forest to detect cyber intrusions in real-time, successfully detecting 95% of simulated cyber attacks with only a 4% false alarm rate."*

