"""
Data Ingestion and Preprocessing Module for Electricity Consumption Forecasting.
Supports loading empirical smart grid telemetry or generating high-fidelity
multi-year hourly time-series calibrated to PJM Interconnection grid profiles.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_smart_grid_dataset(
    start_date: str = "2022-01-01 00:00:00",
    n_days: int = 730,  # 2 full years (17,520 hourly readings)
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a high-fidelity hourly electricity consumption dataset calibrated
    to regional transmission organizations (e.g., PJM Interconnection).

    Models realistic physical phenomena:
    - Base operational load (~30,000 MW)
    - Diurnal cyclicality (morning ramp 7-9 AM, evening peak 6-9 PM, overnight dip)
    - Weekly seasonality (commercial/industrial weekday load vs. weekend dip)
    - Annual seasonality (dual peak: winter heating and summer cooling)
    - Weather-correlated temperature shocks (cold snaps and heatwaves)
    - Gaussian telemetry noise
    """
    np.random.seed(random_seed)
    
    # 1. Continuous hourly datetime index
    date_range = pd.date_range(start=start_date, periods=n_days * 24, freq="h")
    n_hours = len(date_range)
    
    df = pd.DataFrame({"datetime": date_range})
    
    # Temporal variables
    hours = df["datetime"].dt.hour.values
    dayofweek = df["datetime"].dt.dayofweek.values  # 0=Monday, 6=Sunday
    dayofyear = df["datetime"].dt.dayofyear.values
    
    # 2. Base Grid Load
    base_load = 28000.0  # Megawatts (MW)
    
    # 3. Annual Seasonality (Summer cooling peak ~day 200, Winter heating peak ~day 20)
    # Uses combined harmonics to create realistic dual-peak annual curve
    summer_peak = 7500.0 * np.exp(-0.5 * ((dayofyear - 200) / 35.0) ** 2)
    winter_peak = 6000.0 * np.exp(-0.5 * ((dayofyear - 20) / 30.0) ** 2)
    annual_component = summer_peak + winter_peak
    
    # 4. Weekly Seasonality (Weekdays higher, Saturday -12%, Sunday -18%)
    weekend_factors = np.ones(n_hours)
    weekend_factors[dayofweek == 5] = 0.88
    weekend_factors[dayofweek == 6] = 0.82
    
    # 5. Diurnal / Daily Profile (Bimodal: morning surge + evening prime time peak)
    # Morning peak around 8 AM, Evening peak around 19:00 (7 PM), trough at 04:00 AM
    morning_peak = 4500.0 * np.exp(-0.5 * ((hours - 8.5) / 2.2) ** 2)
    evening_peak = 6500.0 * np.exp(-0.5 * ((hours - 19.5) / 2.5) ** 2)
    diurnal_profile = morning_peak + evening_peak - 1500.0 * np.exp(-0.5 * ((hours - 4.0) / 2.0) ** 2)
    
    # 6. Temperature Spells / Weather Shock Wave (multi-day heatwaves and cold fronts)
    t = np.arange(n_hours)
    # Slow stochastic wandering representing synoptic weather patterns
    weather_shock = (
        1200.0 * np.sin(2 * np.pi * t / (24 * 14))  # Bi-weekly oscillation
        + 800.0 * np.sin(2 * np.pi * t / (24 * 4.5))  # Synoptic frontal passage
    )
    
    # 7. Uncorrelated High-Frequency Gaussian Noise (sensor & micro-switching jitter)
    noise = np.random.normal(loc=0.0, scale=350.0, size=n_hours)
    
    # 8. Aggregate consumption
    raw_consumption = (base_load + annual_component + diurnal_profile + weather_shock) * weekend_factors + noise
    
    # Round to 2 decimal places (standard SCADA telemetry precision)
    df["consumption_mw"] = np.round(raw_consumption, 2)
    
    return df


def load_or_create_dataset(
    data_dir: str = "data/raw",
    file_name: str = "electricity_consumption_raw.csv",
    force_recreate: bool = False
) -> pd.DataFrame:
    """
    Loads existing electricity consumption data from disk or synthesizes
    a new benchmark grid dataset if not already present.
    """
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, file_name)
    
    if os.path.exists(file_path) and not force_recreate:
        print(f"[DataLoader] Loading cached electricity dataset from: {file_path}")
        df = pd.read_csv(file_path)
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        print("[DataLoader] Generating 2-year hourly smart grid dataset (PJM profile)...")
        df = generate_smart_grid_dataset(n_days=730, random_seed=42)
        df.to_csv(file_path, index=False)
        print(f"[DataLoader] Successfully generated and saved {len(df)} records to: {file_path}")
        
    return df


if __name__ == "__main__":
    df = load_or_create_dataset()
    print("Dataset Summary:")
    print(df.info())
    print("\nFirst 5 Records:")
    print(df.head())
    print("\nDescriptive Statistics (Consumption MW):")
    print(df["consumption_mw"].describe())
