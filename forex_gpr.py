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

# Supported currencies (all quoted vs USD on Yahoo Finance)
CURRENCY_SYMBOLS = {
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
    "RUB": "RUB=X",
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
    gpr = fetch_gpr_data()
    gpr_daily = gpr.resample("D").ffill()
    df = forex_returns.join(gpr_daily, how="inner").dropna()
    
    if len(df) < 100:
        print("❌ Not enough overlapping data.")
        sys.exit(1)

    df["gpr_std"] = (df["GPR"] - df["GPR"].mean()) / df["GPR"].std()
    print(f"✅ Loaded {len(df)} days (from {df.index[0].date()} to {df.index[-1].date()})")

    print("\n🔄 Running Bayesian MCMC model (4 chains)...")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=0, sigma=0.001)
        beta = pm.Normal("beta", mu=0, sigma=0.1)
        sigma = pm.HalfNormal("sigma", sigma=0.02)
        mu = alpha + beta * df["gpr_std"].values
        pm.Normal("returns", mu=mu, sigma=sigma, observed=df["return"].values)
        trace = pm.sample(draws=800, tune=800, chains=4, cores=4, random_seed=42, progressbar=True)

    # Forecast
    gpr_today = df["gpr_std"].iloc[-1]
    alpha_post = trace.posterior["alpha"].values.flatten()
    beta_post = trace.posterior["beta"].values.flatten()
    sigma_post = trace.posterior["sigma"].values.flatten()

    returns_pred = np.random.normal(
        loc=alpha_post + beta_post * gpr_today,
        scale=sigma_post,
        size=len(alpha_post)
    )

    median_ret = np.median(returns_pred)
    lower = np.percentile(returns_pred, 2.5)
    upper = np.percentile(returns_pred, 97.5)

    print(f"\n🔮 Forecast: Next-day {pair_name} return")
    print(f"   Median: {median_ret:+.4%}")
    print(f"   95% CI: [{lower:.4%}, {upper:.4%}]")

    # Position sizing
    try:
        risk = float(input("\nRisk tolerance (1-10): "))
        risk = max(1, min(10, risk))
    except:
        risk = 5.0
    kelly = 0.5 * median_ret / (np.var(returns_pred) + 1e-8)
    position = np.clip(kelly * (risk / 10.0), -1.0, 1.0)
    direction = "LONG" if position > 0 else "SHORT" if position < 0 else "NEUTRAL"
    size = f"{abs(position)*100:.1f}%" if position != 0 else "0%"
    print(f"\n🎯 Suggested action: {direction} {size} on {pair_name}")

    print("\n💡 Note: This is not financial advice. Use at your own risk.")

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
        all_currencies = ["USD"] + list(CURRENCY_SYMBOLS.keys())
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
