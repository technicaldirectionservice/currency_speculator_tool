#!/usr/bin/env python3
"""
Backtest forex_gpr logic from 2020–2025
Includes Sharpe ratio and drawdown analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import currency symbols
from forex_gpr import CURRENCY_SYMBOLS

# ==============================
# DATA LOADING FUNCTIONS
# ==============================

def fetch_gpr_data():
    """Fetch GPR data from the official source (updated 2025)"""
    import pandas as pd
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    df = pd.read_excel(url)
    df['Date'] = pd.to_datetime(df['month'], errors='coerce')
    df = df.dropna(subset=['Date', 'GPR'])
    return df.set_index('Date')[['GPR']]

def preload_fx_data(pairs, start="2010-01-01", end="2025-12-31"):
    """Download all FX data once at the start"""
    fx_data = {}
    
    # Build list of unique symbols
    symbols_needed = set()
    for _, base, quote in pairs:
        if base != "USD":
            symbols_needed.add(CURRENCY_SYMBOLS[base])
        if quote != "USD":
            symbols_needed.add(CURRENCY_SYMBOLS[quote])
    
    # Download each symbol once
    for symbol in symbols_needed:
        print(f"📥 Downloading {symbol}...")
        data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        fx_data[symbol] = data['Close']
    
    # Add USD as constant 1.0
    full_index = pd.date_range(start, end, freq='D')
    fx_data["USD"] = pd.Series(1.0, index=full_index)
    
    return fx_data

def compute_cross_return(base, quote, date, fx_data):
    """Compute return for base/quote on a given date using preloaded data"""
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
# MAIN BACKTEST
# ==============================

def run_backtest():
    PAIRS = [
        ("USD/JPY", "USD", "JPY"),
        ("USD/CHF", "USD", "CHF"),
        ("EUR/USD", "EUR", "USD"),
        ("USD/ILS", "USD", "ILS"),
    ]
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # Preload ALL data once
    print("⏳ Preloading FX and GPR data...")
    fx_data = preload_fx_data(PAIRS, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    gpr = fetch_gpr_data()
    gpr_daily = gpr.resample("D").ffill()
    
    results = []
    current = start_date
    print(f"📊 Backtesting from {start_date.date()} to {end_date.date()}...")
    
    while current <= end_date:
        if current.weekday() < 5:  # Weekdays only
            # Get GPR as of current date
            try:
                gpr_today = gpr_daily.loc[:current].iloc[-1]['GPR']
                gpr_std = (gpr_today - gpr['GPR'].mean()) / gpr['GPR'].std()
            except:
                current += timedelta(days=1)
                continue
            
            for pair_name, base, quote in PAIRS:
                # Get actual return for NEXT DAY (what we're forecasting)
                next_day = current + timedelta(days=1)
                actual_return = compute_cross_return(base, quote, next_day, fx_data)
                if actual_return is None:
                    continue
                
                # Build historical returns up to current date
                base_series = fx_data["USD"] if base == "USD" else fx_data[CURRENCY_SYMBOLS[base]]
                quote_series = fx_data["USD"] if quote == "USD" else fx_data[CURRENCY_SYMBOLS[quote]]
                df_full = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
                cross_full = df_full['quote_usd'] / df_full['base_usd']
                returns_full = cross_full.pct_change().dropna()
                returns_hist = returns_full[returns_full.index <= current]
                
                if len(returns_hist) < 100:
                    continue
                
                # ✅ Explicitly name the return column
                returns_df = returns_hist.to_frame(name='return')
                aligned = returns_df.join(gpr_daily, how="inner").dropna()
                if len(aligned) < 50:
                    continue
                
                aligned["gpr_std"] = (aligned["GPR"] - gpr["GPR"].mean()) / gpr["GPR"].std()
                
                # OLS regression
                import statsmodels.api as sm
                X = sm.add_constant(aligned["gpr_std"])
                model = sm.OLS(aligned["return"], X).fit()
                forecast = model.predict([1, gpr_std])[0]
                
                # Position sizing (conservative)
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
        # Progress indicator every 100 days
        if (current - start_date).days % 100 == 0:
            print(f"  → Processed up to {current.date()}")
    
    return pd.DataFrame(results)

# ==============================
# ANALYSIS
# ==============================

def analyze_results(df):
    if df.empty:
        print("❌ No results to analyze")
        return
        
    df['strategy_return'] = df['position'] * df['actual_return']
    
    # Sharpe ratio
    daily_sharpe = df['strategy_return'].mean() / df['strategy_return'].std()
    annualized_sharpe = daily_sharpe * np.sqrt(252)
    
    # Drawdown
    cum_returns = (1 + df['strategy_return']).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Accuracy
    df['correct'] = (df['forecast'] > 0) == (df['actual_return'] > 0)
    accuracy = df['correct'].mean()
    gpr_90 = df['gpr'].quantile(0.9)
    high_gpr_acc = df[df['gpr'] >= gpr_90]['correct'].mean()
    
    print("\n" + "="*50)
    print("📈 BACKTEST SUMMARY (2020–2025)")
    print(f"• Total Signals: {len(df)}")
    print(f"• Directional Accuracy: {accuracy:.1%}")
    print(f"• High-GPR Accuracy: {high_gpr_acc:.1%} (top 10% GPR days)")
    print(f"• Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"• Max Drawdown: {max_drawdown:.2%}")
    print("="*50)
    
    df.to_csv("backtest_results.csv", index=False)
    print("\n✅ Results saved to backtest_results.csv")

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    results = run_backtest()
    analyze_results(results)
