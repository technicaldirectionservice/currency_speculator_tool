#!/usr/bin/env python3
"""
Backtest GPR-based forex signals from 1985–2019
Focus: USD/JPY and USD/CHF during pre-2020 safe-haven regime
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forex_gpr import CURRENCY_SYMBOLS

# ==============================
# Policy Rate Loader (from BIS)
# ==============================

def load_policy_rate_as_of(currency, as_of_date):
    filepath = f"data/policy_rates_{currency.lower()}.csv"
    if not os.path.exists(filepath):
        fallback = {
            "USD": 5.0, "JPY": 0.5, "CHF": 1.0, "EUR": 3.0
        }
        return fallback.get(currency, 0.0)
    try:
        df = pd.read_csv(filepath, parse_dates=["date"])
        df = df[df["date"] <= as_of_date].sort_values("date")
        if not df.empty:
            return df.iloc[-1]["rate"]
    except Exception:
        pass
    fallback = {"USD": 5.0, "JPY": 0.5, "CHF": 1.0}
    return fallback.get(currency, 0.0)

# ==============================
# Data Loading
# ==============================

def fetch_gpr_data():
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    df = pd.read_excel(url)
    df['Date'] = pd.to_datetime(df['month'], errors='coerce')
    df = df.dropna(subset=['Date', 'GPR'])
    return df.set_index('Date')[['GPR']]

def preload_fx_data(pairs, start="1980-01-01", end="2020-12-31"):
    fx_data = {}
    symbols_needed = set()
    for _, base, quote in pairs:
        if base != "USD":
            symbols_needed.add(CURRENCY_SYMBOLS[base])
        if quote != "USD":
            symbols_needed.add(CURRENCY_SYMBOLS[quote])
    for symbol in symbols_needed:
        print(f"📥 Downloading {symbol}...")
        data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        fx_data[symbol] = data['Close']
    full_index = pd.date_range(start, end, freq='D')
    fx_data["USD"] = pd.Series(1.0, index=full_index)
    return fx_data

def compute_cross_return(base, quote, date, fx_data):
    try:
        base_series = fx_data["USD"] if base == "USD" else fx_data[CURRENCY_SYMBOLS[base]]
        quote_series = fx_data["USD"] if quote == "USD" else fx_data[CURRENCY_SYMBOLS[quote]]
        df = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
        cross_rate = df['quote_usd'] / df['base_usd']
        returns = cross_rate.pct_change().dropna()
        return returns.loc[date]
    except (KeyError, ValueError):
        return None

# ==============================
# Main Backtest
# ==============================

def run_backtest():
    PAIRS = [
        ("USD/JPY", "USD", "JPY"),
        ("USD/CHF", "USD", "CHF"),
    ]
    
    start_date = datetime(1985, 1, 1)
    end_date = datetime(2019, 12, 31)
    
    print("⏳ Preloading FX and GPR data (1985–2019)...")
    fx_data = preload_fx_data(PAIRS, start="1980-01-01", end="2020-12-31")
    gpr = fetch_gpr_data()
    # Simulate 1-month GPR publication lag
    gpr.index = gpr.index + pd.DateOffset(months=1)
    gpr_daily = gpr.resample("D").ffill()
    
    results = []
    current = start_date
    print(f"📊 Backtesting from {start_date.date()} to {end_date.date()}...")
    
    while current <= end_date:
        if current.weekday() < 5:
            try:
                gpr_today = gpr_daily.loc[:current].iloc[-1]['GPR']
                gpr_std = (gpr_today - gpr['GPR'].mean()) / gpr['GPR'].std()
            except:
                current += timedelta(days=1)
                continue
            
            for pair_name, base, quote in PAIRS:
                next_day = current + timedelta(days=1)
                actual_return = compute_cross_return(base, quote, next_day, fx_data)
                if actual_return is None:
                    continue
                
                base_series = fx_data["USD"] if base == "USD" else fx_data[CURRENCY_SYMBOLS[base]]
                quote_series = fx_data["USD"] if quote == "USD" else fx_data[CURRENCY_SYMBOLS[quote]]
                df_full = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
                cross_full = df_full['quote_usd'] / df_full['base_usd']
                returns_full = cross_full.pct_change().dropna()
                returns_hist = returns_full[returns_full.index <= current]
                
                if len(returns_hist) < 100:
                    continue
                
                returns_df = returns_hist.to_frame(name='return')
                aligned = returns_df.join(gpr_daily, how="inner").dropna()
                if len(aligned) < 50:
                    continue
                
                                # Compute rate differential as of current date
                base_rate = load_policy_rate_as_of(base, current)
                quote_rate = load_policy_rate_as_of(quote, current)
                rate_diff = base_rate - quote_rate

                # Normalize rate_diff (optional but recommended)
                aligned["rate_diff"] = rate_diff  # constant over history for simplicity
                aligned["gpr_std"] = (aligned["GPR"] - gpr["GPR"].mean()) / gpr["GPR"].std()
                
                # Multivariate OLS: return ~ gpr_std + rate_diff
                import statsmodels.api as sm
                X = sm.add_constant(aligned[["gpr_std", "rate_diff"]])
                model = sm.OLS(aligned["return"], X).fit()
                forecast = model.predict([1, gpr_std, rate_diff])[0]
                               
                
                # Policy rate adjustment
                try:
                    base_rate = load_policy_rate_as_of(base, current)
                    quote_rate = load_policy_rate_as_of(quote, current)
                    rate_diff = base_rate - quote_rate

                    if (base == "JPY" or quote == "JPY") and abs(rate_diff) > 1.0:
                        if gpr_today > gpr["GPR"].median():
                            forecast *= 0.6

                    if (base == "CHF" or quote == "CHF") and gpr_today > gpr["GPR"].quantile(0.8):
                        forecast *= 1.3

                except Exception:
                    pass
                
                # Position sizing
                sigma = aligned["return"].std()
                kelly = 0.5 * forecast / (sigma**2 + 1e-8)
                position = np.clip(kelly * 0.5, -1.0, 1.0)
                
                results.append({
                    'date': current,
                    'pair': pair_name,
                    'forecast': forecast,
                    'actual_return': actual_return,
                    'position': position,
                    'gpr': gpr_today
                })
        
        current += timedelta(days=1)
        if (current - start_date).days % 500 == 0:
            print(f"  → Processed up to {current.date()}")
    
    return pd.DataFrame(results)

# ==============================
# Analysis
# ==============================

def analyze_results(df):
    if df.empty:
        print("❌ No results")
        return
        
    df['strategy_return'] = df['position'] * df['actual_return']
    daily_sharpe = df['strategy_return'].mean() / df['strategy_return'].std()
    annualized_sharpe = daily_sharpe * np.sqrt(252)
    
    cum_returns = (1 + df['strategy_return']).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    df['correct'] = (df['forecast'] > 0) == (df['actual_return'] > 0)
    accuracy = df['correct'].mean()
    gpr_90 = df['gpr'].quantile(0.9)
    high_gpr_acc = df[df['gpr'] >= gpr_90]['correct'].mean()
    
    print("\n" + "="*50)
    print("📈 BACKTEST SUMMARY (1985–2019)")
    print(f"• Total Signals: {len(df)}")
    print(f"• Directional Accuracy: {accuracy:.1%}")
    print(f"• High-GPR Accuracy: {high_gpr_acc:.1%} (top 10% GPR days)")
    print(f"• Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"• Max Drawdown: {max_drawdown:.2%}")
    print("="*50)
    
    df.to_csv("backtest_1985_2019_results.csv", index=False)
    print("\n✅ Results saved to backtest_1985_2019_results.csv")

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    results = run_backtest()
    analyze_results(results)
