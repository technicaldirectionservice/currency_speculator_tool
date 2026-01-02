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
    url = "https://www.matteoiacoviello.com/gpr_files/gpr_monthly.xlsx"
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
    if base == "USD":
        base_series = pd.Series(1.0, index=pd.date_range(start, end))
    else:
        base_series = get_fx_series(CURRENCY_SYMBOLS[base], start, end)
    
    if quote == "USD":
        quote_series = pd.Series(1.0, index=pd.date_range(start, end))
    else:
        quote_series = get_fx_series(CURRENCY_SYMBOLS[quote], start, end)
    
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
    current = start_date
    
    results = []
    gpr = fetch_gpr_data()
    gpr_daily = gpr.resample("D").ffill()
    
    print(f"📊 Backtesting from {start_date.date()} to {end_date.date()}...")
    
    while current <= end_date:
        if current.weekday() >= 5:  # Skip weekends
            current += timedelta(days=1)
            continue
            
        # Get GPR as of current date (last available)
        try:
            gpr_today = gpr_daily.loc[:current].iloc[-1]['GPR']
            gpr_std = (gpr_today - gpr['GPR'].mean()) / gpr['GPR'].std()
        except:
            current += timedelta(days=1)
            continue
        
        for pair_name, base, quote in PAIRS:
            try:
                returns = compute_cross_rate(base, quote, end=current.strftime("%Y-%m-%d"))
                if len(returns) < 100:
                    continue
                    
                # Simple regression: return ~ gpr_std
                aligned = returns.to_frame().join(gpr_daily, how="inner").dropna()
                if len(aligned) < 50:
                    continue
                    
                aligned["gpr_std"] = (aligned["GPR"] - gpr["GPR"].mean()) / gpr["GPR"].std()
                
                # OLS regression
                import statsmodels.api as sm
                X = sm.add_constant(aligned["gpr_std"])
                model = sm.OLS(aligned["return"], X).fit()
                forecast = model.predict([1, gpr_std])[0]
                
                # Actual next-day return
                next_day = current + timedelta(days=1)
                try:
                    actual = returns.loc[next_day]
                except KeyError:
                    continue
                
                # Position (fractional Kelly approx)
                sigma = aligned["return"].std()
                kelly = 0.5 * forecast / (sigma**2 + 1e-8)
                position = np.clip(kelly * 0.5, -1.0, 1.0)  # conservative risk
                
                results.append({
                    'date': current,
                    'pair': pair_name,
                    'forecast': forecast,
                    'actual_return': actual,
                    'position': position,
                    'gpr': gpr_today
                })
            except Exception as e:
                continue
                
        current += timedelta(days=1)
    
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