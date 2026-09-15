"""
Cybersecurity Attack Synthesis Module for Smart Grid Telemetry.
Simulates realistic cyber-physical threat vectors against electricity infrastructure:
1. False Data Injection Attacks (FDIA) - Scaling & Random Jitter
2. Coordinated Load Manipulation / Denial-of-Service (DoS) Surge Attacks
3. Energy Theft / Stealth Meter Tampering (Under-reporting)
4. Telemetry Freezing & Replay Attacks
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class CyberAttackSimulator:
    """
    Simulates cyber intrusions on electrical grid consumption telemetry.
    Generates ground-truth labeled scenarios for evaluating intrusion detection.
    """
    
    ATTACK_TYPES = {
        "NORMAL": 0,
        "FDI_SCALING": 1,
        "FDI_JITTER": 2,
        "DOS_LOAD_SURGE": 3,
        "ENERGY_THEFT": 4,
        "TELEMETRY_REPLAY": 5
    }

    ATTACK_NAMES = {v: k for k, v in ATTACK_TYPES.items()}

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

    def inject_fdi_scaling(
        self, series: np.ndarray, scale_factor: float = 1.35
    ) -> np.ndarray:
        """
        False Data Injection - Multiplicative scaling attack.
        Manipulates telemetry to report artificial scale increase/decrease.
        """
        return series * scale_factor

    def inject_fdi_jitter(
        self, series: np.ndarray, noise_std_ratio: float = 0.25
    ) -> np.ndarray:
        """
        False Data Injection - Random high-variance noise injection to destabilize
        automatic generation control (AGC) and state estimators.
        """
        noise_magnitude = np.mean(series) * noise_std_ratio
        noise = self.rng.normal(0, noise_magnitude, size=len(series))
        return np.maximum(0, series + noise)

    def inject_load_surge_dos(
        self, series: np.ndarray, surge_pct: float = 0.45
    ) -> np.ndarray:
        """
        Coordinated IoT Load Surge (DoS) Attack.
        Simulates synchronized manipulation of flexible loads (EVs, HVACs)
        causing severe artificial demand spikes to trip protective relays.
        """
        mean_val = np.mean(series)
        return series + (mean_val * surge_pct)

    def inject_energy_theft(
        self, series: np.ndarray, theft_ratio: float = 0.35
    ) -> np.ndarray:
        """
        Energy Theft / Meter Tampering Attack.
        Stealthy under-reporting of power usage to evade billing or mask diversion.
        """
        return series * (1.0 - theft_ratio)

    def inject_telemetry_replay(
        self, series: np.ndarray, freeze_val: float = None
    ) -> np.ndarray:
        """
        Telemetry Freezing / Replay Attack.
        Repeats static sensor values or frozen readings to blind operators to grid state changes.
        """
        if freeze_val is None:
            freeze_val = series[0]
        return np.full_like(series, fill_value=freeze_val)

    def create_cyber_test_scenarios(
        self, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Injects labeled attack intervals into a holdout test dataframe.
        Constructs realistic cyber intrusion episodes interleaved with normal operation.
        
        Returns:
            pd.DataFrame: Contains:
                - datetime
                - consumption_mw (compromised/reported telemetry)
                - true_consumption_mw (uncompromised ground truth)
                - is_attack (binary label: 0=benign, 1=cyber attack)
                - attack_type (categorical label)
                - attack_code (integer code)
        """
        df = test_df.copy().reset_index(drop=True)
        n = len(df)
        
        # Ground truth uncompromised telemetry
        df["true_consumption_mw"] = df["consumption_mw"].values
        df["is_attack"] = 0
        df["attack_code"] = self.ATTACK_TYPES["NORMAL"]
        df["attack_type"] = "NORMAL"

        # Define 5 distinct multi-hour attack episodes in the test set
        # Episode 1: False Data Injection (Scaling)
        ep1_start = int(n * 0.12)
        ep1_len = 36  # 36 hours
        ep1_end = min(n, ep1_start + ep1_len)
        df.loc[ep1_start:ep1_end - 1, "consumption_mw"] = self.inject_fdi_scaling(
            df.loc[ep1_start:ep1_end - 1, "true_consumption_mw"].values, scale_factor=1.30
        )
        df.loc[ep1_start:ep1_end - 1, "is_attack"] = 1
        df.loc[ep1_start:ep1_end - 1, "attack_code"] = self.ATTACK_TYPES["FDI_SCALING"]
        df.loc[ep1_start:ep1_end - 1, "attack_type"] = "FDI_SCALING"

        # Episode 2: False Data Injection (High-variance Jitter)
        ep2_start = int(n * 0.32)
        ep2_len = 48  # 48 hours
        ep2_end = min(n, ep2_start + ep2_len)
        df.loc[ep2_start:ep2_end - 1, "consumption_mw"] = self.inject_fdi_jitter(
            df.loc[ep2_start:ep2_end - 1, "true_consumption_mw"].values, noise_std_ratio=0.30
        )
        df.loc[ep2_start:ep2_end - 1, "is_attack"] = 1
        df.loc[ep2_start:ep2_end - 1, "attack_code"] = self.ATTACK_TYPES["FDI_JITTER"]
        df.loc[ep2_start:ep2_end - 1, "attack_type"] = "FDI_JITTER"

        # Episode 3: Coordinated IoT Load Surge / DoS
        ep3_start = int(n * 0.52)
        ep3_len = 24  # 24 hours
        ep3_end = min(n, ep3_start + ep3_len)
        df.loc[ep3_start:ep3_end - 1, "consumption_mw"] = self.inject_load_surge_dos(
            df.loc[ep3_start:ep3_end - 1, "true_consumption_mw"].values, surge_pct=0.40
        )
        df.loc[ep3_start:ep3_end - 1, "is_attack"] = 1
        df.loc[ep3_start:ep3_end - 1, "attack_code"] = self.ATTACK_TYPES["DOS_LOAD_SURGE"]
        df.loc[ep3_start:ep3_end - 1, "attack_type"] = "DOS_LOAD_SURGE"

        # Episode 4: Stealth Energy Theft (Bypass Under-reporting)
        ep4_start = int(n * 0.72)
        ep4_len = 60  # 60 hours
        ep4_end = min(n, ep4_start + ep4_len)
        df.loc[ep4_start:ep4_end - 1, "consumption_mw"] = self.inject_energy_theft(
            df.loc[ep4_start:ep4_end - 1, "true_consumption_mw"].values, theft_ratio=0.35
        )
        df.loc[ep4_start:ep4_end - 1, "is_attack"] = 1
        df.loc[ep4_start:ep4_end - 1, "attack_code"] = self.ATTACK_TYPES["ENERGY_THEFT"]
        df.loc[ep4_start:ep4_end - 1, "attack_type"] = "ENERGY_THEFT"

        # Episode 5: Telemetry Freezing / Replay Attack
        ep5_start = int(n * 0.88)
        ep5_len = 30  # 30 hours
        ep5_end = min(n, ep5_start + ep5_len)
        df.loc[ep5_start:ep5_end - 1, "consumption_mw"] = self.inject_telemetry_replay(
            df.loc[ep5_start:ep5_end - 1, "true_consumption_mw"].values
        )
        df.loc[ep5_start:ep5_end - 1, "is_attack"] = 1
        df.loc[ep5_start:ep5_end - 1, "attack_code"] = self.ATTACK_TYPES["TELEMETRY_REPLAY"]
        df.loc[ep5_start:ep5_end - 1, "attack_type"] = "TELEMETRY_REPLAY"

        return df


if __name__ == "__main__":
    from data_loader import load_or_create_dataset
    df_raw = load_or_create_dataset()
    simulator = CyberAttackSimulator()
    attack_df = simulator.create_cyber_test_scenarios(df_raw.tail(2000))
    
    print("\nAttack Simulation Completed:")
    print(attack_df["attack_type"].value_counts())
    print(f"\nTotal simulated records: {len(attack_df)}")
    print(f"Attacks injected: {attack_df['is_attack'].sum()} hours ({attack_df['is_attack'].mean():.2%})")
