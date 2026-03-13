# Market Volatility Spike Prediction

Binary classification model that predicts whether a volatility spike will occur in the next **H** time steps, given the previous **W** steps of market metrics derived from real OHLCV data.

---

## Problem Formulation

At each 5-minute bar `t`, the model answers:

> *"Given the last W=30 bars of market behaviour, will a volatility spike occur in the next H=10 bars?"*

**Incident definition** — a volatility spike at time `t` is defined as:

```
std(log_returns[t : t+H])  >  2.0  ×  rolling_baseline_vol[t]
```

The threshold is self-normalising: it adapts to the current market regime rather than using a fixed absolute level.

---

## Dataset

- **Source:** Yahoo Finance via `yfinance` (no API key required)
- **Ticker:** SPY (S&P 500 ETF)
- **Frequency:** 5-minute bars, last 60 days (~4 600 bars)
- **Raw columns:** Open, High, Low, Close, Volume

---

## Features

Five metrics are derived from raw OHLCV, then compressed into **35 features** (5 metrics × 7 statistics) per window:

| Metric | Formula | Signal |
|--------|---------|--------|
| `log_return` | `log(close_t / close_{t-1})` | Price change |
| `realised_vol` | Rolling std of log returns (20 bars) | Current vol regime |
| `price_range` | `(high - low) / close` | Bid-ask spread proxy |
| `volume_ratio` | `volume / rolling_mean_volume` | Volume surge |
| `momentum` | `close / rolling_mean_close - 1` | Price displacement |

**Per-metric statistics:** mean, std, min, max, last value, linear slope, fraction of bars above a soft threshold.

---

## Model

**LightGBM** gradient-boosted trees.

| Parameter | Value | Reason |
|-----------|-------|--------|
| `objective` | `binary` | Binary cross-entropy loss |
| `scale_pos_weight` | `neg / pos` | Handles class imbalance without oversampling |
| `num_leaves` | 31 | Moderate complexity for noisy financial data |
| `learning_rate` | 0.05 | Small steps with early stopping |
| `early_stopping_rounds` | 50 | Stops when validation loss plateaus |

---

## Evaluation

- **Split:** strict temporal — train 70% | val 10% | test 20% (no shuffling)
- **Alert threshold:** chosen by maximising F1 on the validation set
- **Primary metric:** PR-AUC (preferred over ROC-AUC for imbalanced classes)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.9478 |
| PR-AUC | 0.6785 |
| Precision | 0.7215 |
| Recall | 0.5000 |
| F1 | 0.5907 |

---

## Installation

```bash
pip install yfinance lightgbm scikit-learn pandas numpy
```

## Usage

```bash
python predict_market_incident.py
```

The script downloads data automatically, trains the model, and prints results to the console.

---

## Files

| File | Description |
|------|-------------|
| `predict_market_incident.py` | Clean, minimal implementation |
| `market_incident_prediction.py` | Fully documented version with detailed comments |

---

## Limitations

1. **Short history** — 60 days may not cover all market regimes (bull, bear, crisis).
2. **No order book** — `price_range` is a proxy for bid-ask spread; true microstructure data requires a paid feed.
3. **Alert deduplication** — consecutive windows may all fire for the same spike episode; a state machine would be needed in production.
4. **Concept drift** — market behaviour changes over time; the model should be periodically retrained.
5. **No transaction costs** — a live trading system would need to account for slippage and fees.
