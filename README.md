# ⚡ Denmark Electricity Spread Trading — A Leakage-Audited, Risk-Managed ML Pipeline

DISCLAIMER THIS CODE IS FOR EDUCATIONAL PURPUOSE ONLY NOT A FINANCIAL ADVICE THUS NO LIABILITY 
Attribution-NonCommercial-NoDerivatives 4.0 International

Canonical URL https://creativecommons.org/licenses/by-nc-nd/4.0/


> Hourly forecasting of the **DK1–DK2 day-ahead price spread** and its conversion into a
> cost-aware, tail-risk-managed trading strategy — with a full forensic audit of data
> leakage, lookahead bias, and economic viability.

![pipeline](https://img.shields.io/badge/pipeline-leakage--audited-brightgreen)
![validation](https://img.shields.io/badge/validation-rolling--window-blue)
![stress](https://img.shields.io/badge/cost%20headroom-9.6x-orange)

---

## 1. Overview

This project predicts the hourly spread between the Danish day-ahead electricity zones
(`spread = da_price_dk1_eur − da_price_dk2_eur`) and trades it with a fixed-size directional
position. Beyond predictive accuracy, the project's core contribution is **methodological
rigor**:

1. **A strict 24-hour information horizon** — every feature is computable from data available
   at decision time `t−24` for delivery hour `t`.
2. **An automated leakage audit** that fails the build if any contemporaneous target proxy
   survives into the feature matrix.
3. **Tail-risk forensics** proving that a handful of adverse predictions in unseen crash
   regimes can flush an otherwise profitable account.
4. **A learned, lookahead-free trading gate** (ensemble-uncertainty filter + TCN extreme
   imbalance forecaster).
5. **An economic stress test** (fees + slippage) that quantifies exactly how much friction
   the edge survives — and where the equity curve zeroes out.

---

## 2. Data

| Property | Value |
|---|---|
| Source | `04_denmark_energy_ml_with_weather.csv` (Kaggle: michelemil/dk-el-trade-dataset) |
| Frequency | Hourly (`delivery_ts`, UTC) |
| Shape | 24,070 rows × 81 columns |
| Coverage | ≈ 2024-01 → 2026-10 |
| Feature families | Zone day-ahead prices (DE, DK1, DK2, NO2, SE3, SE4), cross-border flows (`flow_dk1_*`), inter-zonal spreads, day-ahead weather forecasts (`fcst_*`: wind speed, temperature) |
| Target | `spread = da_price_dk1_eur − da_price_dk2_eur` (€/MWh) |
| Derived risk quantity | `imbalance = |Σ flow_dk1_*|` (MW), used for the tradability gate |

**Split:** chronological 70 / 30 after discarding a 192-row warm-up (24h horizon + 168h
rolling windows), so no feature ever contains a NaN-filled history.

---

## 3. Pipeline Architecture

```
raw CSV
  │  target & imbalance construction
  ▼
causal feature engine ── realized cols → shift(24) base → lags(24/48/168) + rolling(24/168)
  │                    ── fcst_* cols  → kept at face value (day-ahead publications)
  │                    ── calendar/cyclical features (known in advance)
  ▼
warm-up trim → chronological 70/30 split
  ▼
leakage audit (assert: no raw realized cols, max |corr(feature, target)| < 0.9)
  ▼
correlation pruning (>0.95 pairs, zero-variance)
  ▼
rolling-window OOF validation (40% window, 10% step)  →  honest IC / MAE / RMSE
  ▼
ensemble: LightGBM .40 │ XGBoost .30 │ RandomForest .20 │ scaled Ridge .10
  ▼
prediction clipping (train 1st–99th pct)
  ▼
trading gate: lag-24 imbalance ≤ 1000 MW  (+ optional TCN extreme-imbalance filter)
  ▼
PnL simulation → tail-risk forensics → fee & slippage stress-to-zero
```

---

## 4. Methodology Highlights

### 4.1 Information horizon & leakage audit
A trade for delivery hour `t` is decided at `t−24`. Therefore:
- All **realized** market quantities (prices, spreads, flows, load, actual generation) enter
  the model only through a `shift(24)` information base; raw columns are dropped.
- Rolling statistics are computed **on the shifted base**, never on contemporaneous values.
- The trading gate uses **lag-24 imbalance**, not realized imbalance (which is published
  post-delivery and would be lookahead).
- An assertion block verifies at runtime that no raw realized column and no near-perfect
  target proxy (`|corr| ≥ 0.9`) exists in the feature matrix.

> **Forensic finding:** the pre-audit model achieved IC = 0.979 — an impossibility in power
> markets — because pre-computed `spread_*` columns at delivery time were present in `X`.
> Post-audit IC falls to a realistic 0.0–0.8 rolling range. The audit is the project's
> central guarantee.

### 4.2 Validation protocol
Rolling-window out-of-fold validation (train on 40% of the train block, predict the next
10%, slide forward) produces truly out-of-sample predictions used for all residual, IC and
PnL diagnostics — no holdout is ever touched for tuning.

### 4.3 Ensemble
Weighted blend of LightGBM (0.40), XGBoost (0.30), RandomForest (0.20) and a
standardized Ridge (0.10). Ensemble **disagreement (std across models)** doubles as an
uncertainty signal for the risk gate.

### 4.4 Tail-risk forensics
- Test-set spread kurtosis **201.9** vs train **95.7**; skew **−8.84** vs **−3.87**.
- QQ-analysis shows test crashes to **−535 €/MWh** absent from training data.
- Error decomposition: near-zero error in the middle 98%, error ≈ |actual| in the tails —
  the model predicts "normal" during crashes.
- Account-flush demonstration: the top-5 adverse trades erase a dominant share of gross
  profit; low RMSE ≠ low tail risk.

### 4.5 Risk layer
- **Uncertainty filter:** skip trading when ensemble std > 95th percentile (thresholds
  derived from OOF validation, never from the holdout) or |prediction| exceeds a safe bound.
- **TCN extreme-imbalance forecaster (optional module):** a causal Temporal Convolutional
  Network over the top-20 features ranked by point-biserial **IC** against the binary
  extreme-imbalance label; 96-hour context windows; predicts `P(imbalance_t > 1000 MW |
  info at t−24)` and acts as a learned, lookahead-free gate complementing the lag-24 rule.

### 4.6 Economic validation
Two-legged cost model (fee + slippage per side; round-trip = 2×(fee+slip)) applied to every
executed trade; analytic break-even plus a fee×slippage "death-line" heatmap.

---

## 5. Key Results (30% chronological holdout, post-audit)

| Metric | Value |
|---|---|
| Executed test trades (filtered) | 3,808 |
| Gross PnL (zero cost) | **€549,443** |
| Avg gross edge per trade | €144.29 (€2.886/MWh round-trip) |
| Net PnL @ realistic costs (€0.05 fee + €0.10 slip per side) | **€492,323** (89.6% of gross) |
| Max drawdown @ realistic costs | −€17,489 |
| Break-even cost (equity → 0) | €1.443/side (€2.886 round-trip) |
| **Cost headroom** | **≈ 9.6× realistic costs** |
| Rolling IC (168h) range | 0.0 – 0.8 (min −0.34 in Feb-2026 regime) |
| Filter impact | Filtered equity ends above unfiltered (≈ +€35k, smoother curve) |
| TCN gate | val/test PR-AUC, IC and gate-match vs oracle reported per run |

**Verdict:** the edge is economically real — it survives conservative friction with ~90%
retention and an order-of-magnitude cost headroom.

---

## 6. Visualizations

| File | Content |
|---|---|
| `diagnostics_residuals.png` | Residuals vs predicted, distribution, over time, Q-Q |
| `diagnostics_feature_importance.png` | Top-25 LightGBM importances (pruned, causal set) |
| `tail_risk_analysis.png` | Train/test KDE, QQ-plot, error-vs-actual, imbalance log-KDE |
| `pnl_continuity.png` | Validation→test continuous equity curve |
| `account_flush_analysis.png` | Worst adverse trades annotated on the equity curve |
| `ic_and_mitigation_analysis.png` | Equity + rolling IC timeline with filter activations |
| `stress_test_fees_slippage.png` | Equity family under cost load, cost-tolerance curve, fee×slippage death-line heatmap |
| `tcn_extreme_imbalance.png` | TCN PR curves, probability-vs-events timeline, equity by gate rule |

---

## 7. Running the Project

Requirements: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`,
`seaborn`, `scipy`, `torch` (TCN module only).

The project is a single ordered notebook/script; execute cells top-to-bottom:
1. Load & target construction → 2. Causal features → 3. Leakage audit → 4. Pruning →
5. Rolling validation → 6. Ensemble → 7. Predict → 8. PnL & continuity →
9. Submission → 10–14. Diagnostics, tail risk, flush analysis, mitigation & IC →
15. TCN gate → 16. Stress test.

> ⚠️ Cells must be run in order: downstream analyses assert length alignment with the
> audited split (post-warm-up). Re-running out of order fails fast with a clear message.

---

## 8. Key Lessons

1. **IC ≈ 0.98 is a bug report, not a triumph.** Unrealistically high skill in markets
   almost always means leakage.
2. **Low RMSE ≠ low risk.** A model can be right 98% of the time and still be flushed by
   the 2% it gets wrong during unseen crash regimes.
3. **Lookahead hides in simulation code**, not just features — realized imbalance gates,
   contemporaneous rolling windows and test-derived thresholds are all silent leaks.
4. **Filters that hurt a leaky model may help an honest one.** The uncertainty filter lost
   money on the leaked model (it blocked "sure bets") and made money on the audited one.
5. **An edge is only real after costs.** Break-even analysis converts ML metrics into a
   business decision.

---

## 9. Limitations & Future Work

- Legging/execution risk between the two zones is not modeled.
- `fcst_*` columns are assumed to be day-ahead publications; if they are nowcasts, flip
  `ASSUME_FCST_KNOWN_DAY_AHEAD = False`.
- Volatility-targeted position sizing (currently fixed 50 MW) would further damp tail PnL.
- Live monitoring of trailing IC to trigger retraining on regime shift.

---

## 10. Acknowledgements

Grateful thanks to **Qwen AI** ("GWEN Ai") for pair-debugging this pipeline — from the
original `KeyError`s and shape mismatches to the forensic discovery of the IC-0.979
leakage and the design of the stress-to-zero framework.

---

*Disclaimer: research project for educational/competition purposes. Not investment advice.*
