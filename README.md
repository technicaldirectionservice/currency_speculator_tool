 # Forex Geopolitical Risk Forecaster
A research tool for foreign exchange traders utilizing Markov Chain Monte Carlo predictive analysis, created using qwen.ai. 
A terminal-based Bayesian forecaster that predicts moves using real-world geopolitical risk data.
A true multi-currency geopolitical risk daily forcaster dashboard — all in your terminal.

## Features
- Probabilistic forecast (not just a point estimate)
- Uses Iacoviello and Caldara's [Geopolitical Risk (GPR) Index](https://www.matteoiacoviello.com/gpr.htm)
- Suggests risk-adjusted position sizes using fractional Kelly criterion
- 100% free and open — no API keys needed
- Forecasts next-day returns for 15+ major and emerging-market currency pairs
- Supports custom cross-currency pairs (e.g., `JPY/DKK`, `ILS/CAD`)


## ⚠️ Important Notes

- **Time Horizon**: Forecasts are for the **next trading day only** — *not* for swing or position trading.
- **GPR Lag**: Geo-Political Risk data is published with a **~1-month delay** (e.g., January data released in February).
- **Not Financial Advice**: This is a research tool. Always use stop-losses and never risk more than you can afford to lose.
- **Regime Awareness**: Since 2024, **JPY no longer behaves as a traditional safe haven** due to BoJ policy — while **CHF remains reliable**.
- Backtesting shows GPR provides a weak edge (51.3% accuracy), but monetary policy (especially BoJ) dominates in 2024–2025. Use this tool as a sentiment filter — not a standalone signal. Backtesting shows that the Geopolitical Risk (GPR) index did not produce a statistically significant edge in FX markets from 2020–2025. This appears due to dominant monetary policy effects (e.g., BoJ yield curve control, Fed QT) that overrode traditional safe-haven flows.

This tool should be used as a qualitative context layer, not a standalone trading signal. During true crises (GPR > 90th %ile + VIX > 25), it may offer useful framing — but always confirm with current policy settings.

## 🚀 Quick Start

### 1. Clone and set up
```bash
1. git clone https://github.com/technicaldirectionservice/currency_speculator_tool.git
2. cd currency_speculator_tool
3. python3 -m venv .venv
4. source .venv/bin/activate
5. Install Python 3.9+
6. pip install -r requirements.txt
7. python forex_gpr.py  (or python3 forex...)
8. Follow the propmts

## Data Sources
- **Forex**: Yahoo Finance (`EURUSD=X`...and the rest, add more if you like!)
- **GPR Index**: [Geopolitical-Risk-Index/GPR-Index](https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls)

## Disclaimer
This tool is for educational/research purposes only. Not financial advice. Trading involves risk of loss.
