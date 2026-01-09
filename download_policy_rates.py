#!/usr/bin/env python3
"""
One-time downloader for historical central bank policy rates from BIS (v2 API)
Saves to data/policy_rates_{currency}.csv
"""

import pandas as pd
import os

# Mapping: Currency → BIS v2 Series Key (D.XX format)
BIS_V2_CODES = {
    "USD": "D.US",
    "JPY": "D.JP",
    "CHF": "D.CH",
    "EUR": "D.XM",   # Euro Area
    "GBP": "D.GB",
    "CAD": "D.CA",
    "AUD": "D.AU",
    "NZD": "D.NZ",
    "DKK": "D.DK",
    "ILS": "D.IL",
    "MXN": "D.MX",
    "ZAR": "D.ZA",
    "KRW": "D.KR",
    "HKD": "D.HK",
    "CNY": "D.CN",
    "INR": "D.IN",
    
}

def download_all():
    os.makedirs("data", exist_ok=True)
    
    for currency, series_key in BIS_V2_CODES.items():
        print(f"📥 Downloading {currency} policy rates...")
        try:
            # BIS v2 API: dataflow/BIS/WS_CBPOL/1.0/{series_key}?format=csv
            url = f"https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/{series_key}?format=csv"
            df = pd.read_csv(url, on_bad_lines='skip', engine='python')
            if df.empty or "OBS_VALUE" not in df.columns:
                raise ValueError("No valid data returned")
            
            # Rename columns for clarity
            df = df.rename(columns={
                "TIME_PERIOD": "date",
                "OBS_VALUE": "rate"
            })
            
            
            # Keep only date and rate
            df = df[["date", "rate"]].dropna()
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            
            # Save
            output_path = f"data/policy_rates_{currency.lower()}.csv"
            df.to_csv(output_path, index=False)
            print(f"✅ Saved {output_path} ({len(df)} observations)")
            
        except Exception as e:
            print(f"❌ Failed {currency}: {e}")

if __name__ == "__main__":
    download_all()
