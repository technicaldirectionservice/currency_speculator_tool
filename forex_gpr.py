#!/usr/bin/env python3
"""
Geopolitical Forex Forecaster — Dual Mode
Mode 1: USD → Foreign Currency (simple)
Mode 2: Any → Any Currency (advanced)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import sys

def load_policy_rate_as_of(currency, as_of_date):
    """Load central bank policy rate as of a given date using BIS historical data."""
    import os
    filepath = f"data/policy_rates_{currency.lower()}.csv"
    
    if not os.path.exists(filepath):
        # Fallback defaults (keep these updated)
        fallback = {
            "USD": 5.25, "JPY": 0.10, "CHF": 1.50, "EUR": 4.50,
            "GBP": 4.75, "CAD": 4.00, "AUD": 4.10, "NZD": 4.25,
            "DKK": 3.80, "ILS": 4.50, "MXN": 11.00, "ZAR": 8.25,
            "KRW": 3.50, "HKD": 5.00, "CNY": 3.45, "RUB": 21.00
        }
        return fallback.get(currency, 0.0)
    
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[df["date"] <= as_of_date].sort_values("date")
        if not df.empty:
            return df.iloc[-1]["rate"]
    except Exception:
        pass
    return fallback.get(currency, 0.0)

# Supported currencies (all quoted vs USD on Yahoo Finance)
CURRENCY_SYMBOLS = {
    "USD": None,
    "EUR": "EURUSD=X",
    "JPY": "JPY=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "CAD": "CAD=X",
    "CHF": "CHF=X",
    "NZD": "NZDUSD=X",
    "HKD": "HKD=X",
    "CNY": "USDCNY=X",
    "MXN": "MXN=X",
    "ZAR": "ZAR=X",
    "ILS": "ILS=X",
    "KRW": "KRW=X",
    "DKK": "DKK=X",
    "INR": "INR=X",
}

def fetch_gpr_data():
    print("📥 Fetching Geopolitical Risk (GPR) Index...")
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    try:
        df = pd.read_excel(url)
        if 'month' not in df.columns or 'GPR' not in df.columns:
            raise ValueError("Columns 'month' and 'GPR' not found")
        df['Date'] = pd.to_datetime(df['month'], errors='coerce')
        df = df.dropna(subset=['Date', 'GPR'])
        df = df.set_index('Date').sort_index()
        return df[['GPR']]
    except Exception as e:
        print(f"❌ GPR fetch failed: {e}")
        sys.exit(1)

def fetch_forex_series(symbol, start="2010-01-01"):
    """Fetch a single price series (Close) from Yahoo"""
    data = yf.download(symbol, start=start)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data['Close']

def get_usd_pair_returns(currency):
    """Get USD/XXX or XXX/USD returns"""
    if currency == "USD":
        raise ValueError("Cannot fetch USD/USD")
    symbol = CURRENCY_SYMBOLS[currency]
    close = fetch_forex_series(symbol)
    returns = close.pct_change().dropna()
    return returns.to_frame(name='return')

def get_cross_pair_returns(base, quote):
    """Compute base/quote returns using USD as bridge"""
    if base == quote:
        raise ValueError("Base and quote must differ")
    
    # Get USD/XXX series
    if base == "USD":
        base_series = pd.Series(1.0, index=pd.date_range(start="2010-01-01", end=pd.Timestamp.today(), freq='D'))
    else:
        base_series = fetch_forex_series(CURRENCY_SYMBOLS[base])
    
    if quote == "USD":
        quote_series = pd.Series(1.0, index=pd.date_range(start="2010-01-01", end=pd.Timestamp.today(), freq='D'))
    else:
        quote_series = fetch_forex_series(CURRENCY_SYMBOLS[quote])
    
    # Align and compute cross rate: base/quote = (USD/quote) / (USD/base)
    df = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
    cross_rate = df['quote_usd'] / df['base_usd']
    returns = cross_rate.pct_change().dropna()
    return returns.to_frame(name='return')

def run_bayesian_model(forex_returns, pair_name):
    """Shared modeling logic"""
    
    # 🔑 Parse base and quote from pair_name
    if "/" in pair_name:
        base, quote = pair_name.split("/")
    else:
        base = "USD"
        quote = pair_name

    # Load GPR and align with returns
    gpr = fetch_gpr_data()
    gpr_daily = gpr.resample("D").ffill()
    df = forex_returns.join(gpr_daily, how="inner").dropna()
    
    if len(df) < 100:
        print("❌ Not enough overlapping data.")
        sys.exit(1)

    # Compute policy rate differential as of latest date
    as_of_date = df.index[-1]
    base_rate = load_policy_rate_as_of(base, as_of_date)
    quote_rate = load_policy_rate_as_of(quote, as_of_date)
    rate_diff = base_rate - quote_rate

    # Prepare features
    df["gpr_std"] = (df["GPR"] - df["GPR"].mean()) / df["GPR"].std()
    df["rate_diff"] = rate_diff  # constant across history (simplification)
    print(f"✅ Loaded {len(df)} days (from {df.index[0].date()} to {df.index[-1].date()})")

    print("\n🔄 Running Bayesian MCMC model (4 chains)...")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=0.001)
        beta_gpr = pm.Normal("beta_gpr", mu=0, sigma=0.1)
        beta_rate = pm.Normal("beta_rate", mu=0, sigma=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        mu = alpha + beta_gpr * df["gpr_std"].values + beta_rate * df["rate_diff"].values
        pm.Normal("returns", mu=mu, sigma=sigma, observed=df["return"].values)
        trace = pm.sample(draws=800, tune=800, chains=4, cores=4, random_seed=42, progressbar=True)

    # Forecast using posterior
    gpr_today = df["gpr_std"].iloc[-1]
    gpr_today_raw = df["GPR"].iloc[-1]
    alpha_post = trace.posterior["alpha"].values.flatten()
    beta_gpr_post = trace.posterior["beta_gpr"].values.flatten()
    beta_rate_post = trace.posterior["beta_rate"].values.flatten()
    sigma_post = trace.posterior["sigma"].values.flatten()

    returns_pred = np.random.normal(
        loc=alpha_post + beta_gpr_post * gpr_today + beta_rate_post * rate_diff,
        scale=sigma_post,
        size=len(alpha_post)
    )

    median_ret = np.median(returns_pred)
    lower = np.percentile(returns_pred, 2.5)
    upper = np.percentile(returns_pred, 97.5)

    print(f"\n🔮 Forecast: Next-day {pair_name} return")
    print(f"   Median: {median_ret:+.4%}")
    print(f"   95% CI: [{lower:.4%}, {upper:.4%}]")

    # Crisis regime filter
    gpr_today_raw = df["GPR"].iloc[-1]
    gpr_80 = df["GPR"].quantile(0.8)
    vix = 0.0 
    try:
        vix_data = yf.download("^VIX", period="1d", auto_adjust=True, progress=False)
        if not vix_data.empty:
            vix = vix_data['Close'].iloc[-1].item()
            #vix = float(vix_data['Close'].iloc[-1])  # ← FORCE TO FLOAT
    except:
        vix = 0.0

    if gpr_today_raw <= gpr_80 or vix <= 20:
        print("\n🔇 GPR not in crisis regime — no actionable signal")
        print(f"   (Threshold: GPR > {gpr_80:.1f} and VIX > 20)")
        print("\n🎯 Suggested action: NEUTRAL (0%) on", pair_name)
        position = 0.0
    else:
        # Position sizing
        try:
            risk = float(input("\nRisk tolerance during crisis (1-10): "))
            risk = max(1, min(10, risk))
        except:
            risk = 5.0
        kelly = 0.5 * median_ret / (np.var(returns_pred) + 1e-8)
        position = np.clip(kelly * (risk / 10.0), -1.0, 1.0)
        direction = "LONG" if position > 0 else "SHORT" if position < 0 else "NEUTRAL"
        size = f"{abs(position)*100:.1f}%" if position != 0 else "0%"
        print(f"\n🎯 Suggested action: {direction} {size} on {pair_name}")

    print("\n🔍 Important Notes:")
    print("- GPR data is published with a ~1-month lag")
    print("- Forecast is for *next trading day only*")
    print("- This tool uses data-driven policy rate + GPR modeling (no 'safe haven' assumptions)")
    print("- This is not financial advice. Use at your own risk.")


def main():
    print("🌍 Geopolitical Forex Forecaster")
    print("=" * 50)
    print("\nChoose mode:")
    print("1. USD → Foreign Currency (e.g., USD/JPY)")
    print("2. Custom Pair (e.g., JPY/DKK, EUR/GBP)")

    mode = input("\nSelect mode (1 or 2): ").strip()
    
    if mode == "1":
        # Traditional USD → XXX mode
        print("\n💱 Select foreign currency:")
        currencies = list(CURRENCY_SYMBOLS.keys())
        for i, c in enumerate(currencies, 1):
            print(f"  {i}. {c}")
        try:
            idx = int(input(f"\nChoice (1-{len(currencies)}): ")) - 1
            currency = currencies[idx]
            pair_name = f"USD/{currency}"
            forex_returns = get_usd_pair_returns(currency)
        except:
            print("⚠️ Invalid. Defaulting to USD/JPY.")
            pair_name = "USD/JPY"
            forex_returns = get_usd_pair_returns("JPY")
    
    elif mode == "2":
        # Custom cross-currency mode
        all_currencies = list(CURRENCY_SYMBOLS.keys())
        print("\n💱 Select BASE currency (you hold this):")
        for i, c in enumerate(all_currencies, 1):
            print(f"  {i}. {c}")
        try:
            b = int(input(f"\nBase (1-{len(all_currencies)}): ")) - 1
            base = all_currencies[b]
        except:
            base = "JPY"

        print("\n💱 Select QUOTE currency:")
        for i, c in enumerate(all_currencies, 1):
            print(f"  {i}. {c}")
        try:
            q = int(input(f"\nQuote (1-{len(all_currencies)}): ")) - 1
            quote = all_currencies[q]
            if quote == base:
                raise ValueError
        except:
            quote = "USD"

        pair_name = f"{base}/{quote}"
        forex_returns = get_cross_pair_returns(base, quote)
    
    else:
        print("⚠️ Invalid mode. Defaulting to USD/JPY.")
        pair_name = "USD/JPY"
        forex_returns = get_usd_pair_returns("JPY")

    run_bayesian_model(forex_returns, pair_name)

if __name__ == "__main__":
    main()
