#!/usr/bin/env python3
"""
Forex Geopolitical Risk Forecaster
Predicts major forex pair returns using the Geopolitical Risk (GPR) Index via Bayesian MCMC.
- Auto-fetches GPR from official Iacoviello source
- Runs in terminal
- Suggests risk-adjusted position size
- 100% open-source (MIT-style)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import sys
import xlrd

# Supported forex pairs (Yahoo Finance format)
# Note: For JPY, CHF, CAD, HKD — Yahoo uses "XXX=X" to mean USD/XXX
FOREX_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "USD/JPY": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X",
    "USD/CAD": "CAD=X",
    "USD/CHF": "CHF=X",
    "NZD/USD": "NZDUSD=X",
    "USD/HKD": "HKD=X",
    "USD/CNY": "USDCNY=X",
}

def fetch_gpr_data():
    """Fetch GPR index from official academic source (Iacoviello)"""
    print("📥 Fetching Geopolitical Risk (GPR) Index...")
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    try:
        # Let pandas auto-detect the Excel format (handles .xls and .xlsx)
        df = pd.read_excel(url)
        
         # The actual columns are: 'month' (str) and 'GPR' (float)
        if 'month' not in df.columns or 'GPR' not in df.columns:
            raise ValueError("Expected columns 'month' and 'GPR' not found in Excel file")
        
        # Parse 'month' as datetime (format: "YYYY-MM" or "YYYY-MM-DD")
        df['Date'] = pd.to_datetime(df['month'], errors='coerce')
        df = df.dropna(subset=['Date', 'GPR'])
        df = df.set_index('Date').sort_index()
        
        return df[['GPR']]
    
    except Exception as e:
        print(f"❌ Failed to fetch or parse GPR data: {e}")
        print("💡 Tip: Visit https://www.matteoiacoviello.com/gpr.htm to verify the file structure.")
        sys.exit(1)

def get_forex_data(pair_symbol, start="2010-01-01"):
    """Fetch forex data from Yahoo Finance and flatten MultiIndex"""
    print(f"📥 Fetching {pair_symbol} data...")
    try:
        data = yf.download(pair_symbol, start=start)
        # Flatten columns if MultiIndex (happens with yfinance)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)  # keep 'Close', not (pair, 'Close')
        data["return"] = data["Close"].pct_change()
        data = data.dropna()
        # Ensure index is simple DatetimeIndex
        data.index.name = 'Date'
        return data
    except Exception as e:
        print(f"❌ Failed to fetch {pair} data: {e}")
        sys.exit(1)

def main():
    print("🌍 Forex Geopolitical Risk Forecaster")
    print("=" * 50)

    # === 1. Select currency pair ===
    print("\n💱 Available currency pairs:")
    for i, name in enumerate(FOREX_PAIRS.keys(), 1):
        print(f"  {i}. {name}")

    try:
        choice = int(input(f"\nSelect a pair (1-{len(FOREX_PAIRS)}): "))
        pair_name = list(FOREX_PAIRS.keys())[choice - 1]
        pair_symbol = FOREX_PAIRS[pair_name]
    except (ValueError, IndexError):
        print("⚠️ Invalid choice. Defaulting to EUR/USD.")
        pair_name, pair_symbol = "EUR/USD", "EURUSD=X"


    # === 2. Load data ===
    forex = get_forex_data(pair_symbol)
    gpr = fetch_gpr_data()

    # Resample GPR to daily (forward-fill)
    gpr_daily = gpr.resample("D").ffill()

    # Merge
    df = forex[["return"]].join(gpr_daily, how="inner").dropna()
    if len(df) < 100:
        print("❌ Not enough overlapping data. Try a different date range.")
        sys.exit(1)

    # Standardize GPR
    df["gpr_std"] = (df["GPR"] - df["GPR"].mean()) / df["GPR"].std()
    print(f"✅ Loaded {len(df)} days of data (from {df.index[0].date()} to {df.index[-1].date()})")

    # === 3. Bayesian model ===
    print("\n🔄 Running Bayesian MCMC model(4 chains for robustness)...")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=0.001)
        beta = pm.Normal("beta", mu=0, sigma=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        mu = alpha + beta * df["gpr_std"].values
        pm.Normal("returns", mu=mu, sigma=sigma, observed=df["return"].values)
        trace = pm.sample(draws=1000, tune=1000, chains=4, cores=4, random_seed=42, progressbar=True)

    # === 4. Forecast ===
    gpr_today = df["gpr_std"].iloc[-1]
    alpha_post = trace.posterior["alpha"].values.flatten()
    beta_post = trace.posterior["beta"].values.flatten()
    sigma_post = trace.posterior["sigma"].values.flatten()

    # Simulate next-day return
    n_samples = len(alpha_post)
    returns_pred = np.random.normal(
        loc=alpha_post + beta_post * gpr_today,
        scale=sigma_post,
        size=n_samples
    )

    median_ret = np.median(returns_pred)
    lower = np.percentile(returns_pred, 2.5)
    upper = np.percentile(returns_pred, 97.5)

    print(f"\n🔮 Forecast: Next-day {pair_name} return")
    print(f"   Median: {median_ret:+.4%}")
    print(f"   95% CI: [{lower:.4%}, {upper:.4%}]")

    # === 5. Position sizing ===
    print("\n⚖️  Position Sizing (Risk-Adjusted)")
    try:
        risk_input = input("Enter your risk tolerance (1-10, where 1=conservative, 10=aggressive): ")
        user_risk = float(risk_input)
        if not (1 <= user_risk <= 10):
            raise ValueError
    except:
        print("⚠️  Invalid input. Using default risk=5.")
        user_risk = 5.0

    # Kelly-inspired sizing (fractional + capped)
    expected_return = median_ret
    variance = np.var(returns_pred)
    kelly = 0.5 * expected_return / (variance + 1e-8) if variance > 0 else 0.0
    risk_factor = user_risk / 10.0
    position = kelly * risk_factor

    # Cap leverage
    position = max(-1.0, min(1.0, position))

    # Interpretation
    if position > 0:
        direction = "LONG"
        size_pct = f"{position * 100:.1f}%"
    elif position < 0:
        direction = "SHORT"
        size_pct = f"{abs(position) * 100:.1f}%"
    else:
        direction = "NEUTRAL"
        size_pct = "0%"

    print(f"\n🎯 Suggested action: {direction} {size_pct} of your account on {pair_name}")
    print("\n💡 Notes:")
    print("- This is not financial advice. Use at your own risk.")
    print("- GPR = Geopolitical Risk Index (Caldara & Iacoviello, 2022)(higher = more global tension)")
    print("- Higher GPR = more global tension → safe-havens (JPY, CHF) often strengthen")
    print("- Data source: https://www.matteoiacoviello.com/gpr.htm")
    print("- SHORT means bet it will fall; LONG means bet it will rise")

if __name__ == "__main__":
    main()
