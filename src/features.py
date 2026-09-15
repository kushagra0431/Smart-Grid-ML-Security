"""
Feature Engineering Module for Smart Grid Electricity Forecasting & Cybersecurity IDS.
Extracts calendar, cyclic, autoregressive lag, and rolling statistical indicators.
Strictly ensures zero temporal data leakage by lagging all endogenous variables.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def engineer_features(
    df: pd.DataFrame,
    target_col: str = "consumption_mw",
    is_inference: bool = False
) -> pd.DataFrame:
    """
    Constructs an extensive feature matrix for time-series forecasting.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'datetime' and 'consumption_mw'.
        target_col (str): The column containing electrical load measurements.
        is_inference (bool): If True, retains rows even if trailing lags produce NaNs.
        
    Returns:
        pd.DataFrame: Augmented DataFrame with engineered feature columns.
    """
    data = df.copy()
    
    # Ensure datetime sorting
    if not pd.api.types.is_datetime64_any_dtype(data["datetime"]):
        data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values("datetime").reset_index(drop=True)
    
    # 1. Calendar / Temporal Properties
    dt = data["datetime"].dt
    data["hour"] = dt.hour
    data["dayofweek"] = dt.dayofweek
    data["dayofyear"] = dt.dayofyear
    data["month"] = dt.month
    data["year"] = dt.year
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)
    
    # Grid peak load indicator: Weekday commercial peak (07:00-10:00 and 17:00-21:00)
    data["is_peak_hour"] = (
        (data["is_weekend"] == 0) & 
        ((data["hour"].between(7, 10)) | (data["hour"].between(17, 21)))
    ).astype(int)
    
    # 2. Smooth Continuous Cyclic Encodings
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)
    data["dayofweek_sin"] = np.sin(2 * np.pi * data["dayofweek"] / 7.0)
    data["dayofweek_cos"] = np.cos(2 * np.pi * data["dayofweek"] / 7.0)
    
    # 3. Autoregressive Lags (Shifted by at least 1 step to prevent lookahead leakage)
    # Lags: 1 hour, 2 hours, 24 hours (previous day), 48 hours, 168 hours (previous week)
    series = data[target_col]
    data["lag_1"] = series.shift(1)
    data["lag_2"] = series.shift(2)
    data["lag_24"] = series.shift(24)
    data["lag_48"] = series.shift(48)
    data["lag_168"] = series.shift(168)
    
    # 4. Rolling Window Statistics (computed strictly on past values shifted by 1)
    past_series = series.shift(1)
    data["rolling_mean_6h"] = past_series.rolling(window=6, min_periods=1).mean()
    data["rolling_mean_24h"] = past_series.rolling(window=24, min_periods=1).mean()
    data["rolling_std_24h"] = past_series.rolling(window=24, min_periods=1).std().fillna(0)
    data["rolling_min_24h"] = past_series.rolling(window=24, min_periods=1).min()
    data["rolling_max_24h"] = past_series.rolling(window=24, min_periods=1).max()
    
    # Trend ratio: relative difference from 24h baseline
    data["load_ratio_24h"] = (data["lag_1"] / (data["rolling_mean_24h"] + 1e-5)).clip(0.5, 2.0)
    
    if not is_inference:
        # Drop warm-up rows where lag_168 is NaN
        data = data.dropna(subset=["lag_168"]).reset_index(drop=True)
    else:
        # For real-time inference fallback, backfill remaining NaNs
        data = data.bfill().ffill()
        
    return data


def get_feature_columns() -> List[str]:
    """Returns the ordered list of predictive feature column names."""
    return [
        "hour", "dayofweek", "dayofyear", "month", "is_weekend", "is_peak_hour",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek_sin", "dayofweek_cos",
        "lag_1", "lag_2", "lag_24", "lag_48", "lag_168",
        "rolling_mean_6h", "rolling_mean_24h", "rolling_std_24h",
        "rolling_min_24h", "rolling_max_24h", "load_ratio_24h"
    ]


if __name__ == "__main__":
    from data_loader import load_or_create_dataset
    df_raw = load_or_create_dataset()
    df_feat = engineer_features(df_raw)
    print(f"Engineered dataset shape: {df_feat.shape}")
    print("Features extracted:", get_feature_columns())
    print("\nSample features:")
    print(df_feat[get_feature_columns()].head(3))
