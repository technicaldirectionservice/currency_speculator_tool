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

# Reuse your currency list
from forex_gpr import CURRENCY_SYMBOLS

def fetch_gpr_data():
    """Fetch GPR data (same as forex_gpr.py)"""
    import pandas as pd
    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    df = pd.read_excel(url)
    df['Date'] = pd.to_datetime(df['month'], errors='coerce')
    df = df.dropna(subset=['Date', 'GPR'])
    return df.set_index('Date')[['GPR']]

def get_fx_series(symbol, start="2010-01-01", end="2025-12-31"):
    """Fetch FX series"""
    data = yf.download(symbol, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data['Close']

def compute_cross_rate(base, quote, start="2010-01-01", end="2025-12-31"):
    """Compute base/quote cross rate"""
    from forex_gpr import CURRENCY_SYMBOLS

    #Handle USD as base quote
    if base == "USD":
        base_series = pd.Series(1.0, index=pd.date_range(start, end, freq='D'))
    else:
        if base not in CURRENCY_SYMBOLS:
            raise ValueError(f"Base currency {base} not supported")
        symbol= CURRENCY_SYMBOLS[base]
        base_series = get_fx_series(symbol, start, end)
    
    if quote == "USD":
        quote_series = pd.Series(1.0, index=pd.date_range(start, end, freq='D'))
    else:
        quote_series = get_fx_series(symbol, start, end)
    
    df = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
    cross_rate = df['quote_usd'] / df['base_usd']
    returns = cross_rate.pct_change().dropna()
    return returns

def run_backtest():
    PAIRS = [
        ("USD/JPY", "USD", "JPY"),
        ("USD/CHF", "USD", "CHF"),
        ("EUR/USD", "EUR", "USD"),
        ("USD/ILS", "USD", "ILS"),
    ]
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2025, 12, 31)
    
    # ✅ Preload ALL data once
    print("⏳ Preloading FX and GPR data...")
    fx_data = preload_fx_data(PAIRS, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    gpr = fetch_gpr_data()
    gpr_daily = gpr.resample("D").ffill()
    
    results = []
    current = start_date
    print(f"📊 Backtesting from {start_date.date()} to {end_date.date()}...")
    
    while current <= end_date:
        if current.weekday() < 5:  # Weekdays only
            try:
                gpr_today = gpr_daily.loc[:current].iloc[-1]['GPR']
                gpr_std = (gpr_today - gpr['GPR'].mean()) / gpr['GPR'].std()
            except:
                current += timedelta(days=1)
                continue
            
            for pair_name, base, quote in PAIRS:
                # Get actual return for this date
                actual_return = compute_cross_return(base, quote, current, fx_data)
                if actual_return is None:
                    continue
                
                # Get next-day return for "actual" (what we're predicting for)
                next_day = current + timedelta(days=1)
                next_return = compute_cross_return(base, quote, next_day, fx_data)
                if next_return is None:
                    continue
                
                # Simple regression on full history up to current
                base_series = fx_data["USD"] if base == "USD" else fx_data[CURRENCY_SYMBOLS[base]]
                quote_series = fx_data["USD"] if quote == "USD" else fx_data[CURRENCY_SYMBOLS[quote]]
                df_full = pd.DataFrame({'base_usd': base_series, 'quote_usd': quote_series}).dropna()
                cross_full = df_full['quote_usd'] / df_full['base_usd']
                returns_full = cross_full.pct_change().dropna()
                returns_hist = returns_full[returns_full.index <= current]
                
                if len(returns_hist) < 100:
                    continue
                
                aligned = returns_hist.to_frame().join(gpr_daily, how="inner").dropna()
                if len(aligned) < 50:
                    continue
                
                aligned["gpr_std"] = (aligned["GPR"] - gpr["GPR"].mean()) / gpr["GPR"].std()
                
                import statsmodels.api as sm
                X = sm.add_constant(aligned["gpr_std"])
                model = sm.OLS(aligned["return"], X).fit()
                forecast = model.predict([1, gpr_std])[0]
                
                # Position sizing
                sigma = aligned["return"].std()
                kelly = 0.5 * forecast / (sigma**2 + 1e-8)
                position = np.clip(kelly * 0.5, -1.0, 1.0)
                
                results.append({
                    'date': current,
                    'pair': pair_name,
                    'forecast': forecast,
                    'actual_return': next_return,  # next-day return
                    'position': position,
                    'gpr': gpr_today
                })
        
        current += timedelta(days=1)
        # Optional: progress indicator
        if (current - start_date).days % 100 == 0:
            print(f"  → Processed up to {current.date()}")
    
    return pd.DataFrame(results)


def analyze_results(df):
    if df.empty:
        print("❌ No results to analyze")
        return
        
    # Strategy returns
    df['strategy_return'] = df['position'] * df['actual_return']
    
    # Sharpe ratio (annualized)
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
    
    # High-GPR accuracy
    gpr_90 = df['gpr'].quantile(0.9)
    high_gpr_acc = df[df['gpr'] >= gpr_90]['correct'].mean()
    
    # Output
    print("\n" + "="*50)
    print("📈 BACKTEST SUMMARY (2020–2025)")
    print(f"• Total Signals: {len(df)}")
    print(f"• Directional Accuracy: {accuracy:.1%}")
    print(f"• High-GPR Accuracy: {high_gpr_acc:.1%} (top 10% GPR days)")
    print(f"• Annualized Sharpe Ratio: {annualized_sharpe:.2f}")
    print(f"• Max Drawdown: {max_drawdown:.2%}")
    print("="*50)
    
    # Save
    df.to_csv("backtest_results.csv", index=False)
    print("\n✅ Results saved to backtest_results.csv")

if __name__ == "__main__":
    results = run_backtest()
    analyze_results(results)
