# UseCase

Classification system that predicts the **direction of RPK  growth** for airline flights — categorized as **Down**, **Flat**, or **Up** — using historical flight  data.

---

## ⚙️ Feature Engineering

The following features are computed before modeling:

| Feature | Description |
|---|---|
| `PLF` | Passenger Load Factor = `(RPK / ASK) × 100` |
| `RPKG_pct` | Period-over-period RPK growth rate (%), clipped to `[-100, 200]` |
| `ASKG_pct` | Period-over-period ASK growth rate (%), clipped to `[-100, 200]` |
| `highD` | Boolean — RPK growth exceeded ASK growth |
| `lowD` | Boolean — RPK growth fell short of ASK growth |
| `Month` | Calendar month extracted from flight date |
| `FLTNUM_ENC` | Label-encoded flight number |

---

## 🎯 Target Variable

The target `TARGET` is derived by binning the **next period's RPK growth rate** into three classes:

| Class | Condition |
|---|---|
| `down` | Next `RPKG_pct` < −10% |
| `flat` | Next `RPKG_pct` between −10% and +5% |
| `up` | Next `RPKG_pct` > +5% |

The target is then label-encoded: `down=0`, `flat=1`, `up=2`.

---

## 🔀 Train/Test Split

- Flights with fewer than **30 records** are excluded.
- Data is split **temporally** (not randomly) at the **80th percentile** of flight dates to avoid leakage:
  - **FltSummary**: train up to `2026-10-17`, test after
  - **LegSummary**: train up to `2026-11-30`, test after
- Features are scaled using **MinMaxScaler** (fit on train, applied to test).
- **Class weights** are computed inversely proportional to class frequency to handle imbalance — no SMOTE is used.

---

## 🤖 Models Trained

Four model types are trained and evaluated on both FltSummary and LegSummary data:

### 1. LightGBM 
A gradient boosted tree model with `multiclass` objective. Configured with:
- Learning rate: `0.007`, 300 boosting rounds
- Monotonic constraints applied to RPK and overbooking features
- Class sample weights passed during training

### 2. Logistic Regression 
L2-regularized logistic regression using the `saga` solver. Serves as a linear baseline. Note: convergence warnings indicate 500 iterations were insufficient for full convergence.

### 3. Gradient Boosting 
Scikit-learn's standard gradient boosting with `log_loss`, depth 10, and 500 estimators. Uses `log2` max features for column subsampling.

### 4. HistGradientBoosting 
Scikit-learn's histogram-based gradient boosting — faster and more memory-efficient than standard GBM. Configured with pairwise interaction constraints, L2 regularization, and 30% validation fraction.

---

## 📊 Model Results

### FltSummary Results

| Model | Accuracy | F1 (down) | F1 (flat) | F1 (up) | F1 Macro |
|---|---|---|---|---|---|
| **LightGBM** | 0.57 | 0.62 | 0.51 | 0.46 | 0.53 |
| **Logistic Regression** | 0.61 | 0.75 | 0.02 | 0.36 | 0.38 |
| **GradientBoosting** | 0.59 | 0.69 | 0.40 | 0.48 | 0.52 |
| **HistGradientBoosting** | 0.61 | 0.73 | 0.34 | 0.48 | 0.52 |

**LightGBM** achieves the best **balanced performance** across all three classes. Logistic Regression scores high accuracy but almost entirely collapses on the `flat` class (F1 = 0.02).

---

### LegSummary Results

| Model | Accuracy | F1 (down) | F1 (flat) | F1 (up) | F1 Macro |
|---|---|---|---|---|---|
| **LightGBM** | 0.55 | 0.48 | 0.61 | 0.41 | 0.50 |
| **Logistic Regression** | 0.47 | 0.55 | 0.43 | 0.11 | 0.36 |
| **GradientBoosting** | 0.59 | 0.67 | 0.48 | 0.41 | 0.52 |
| **HistGradientBoosting** | 0.60 | 0.67 | 0.51 | 0.35 | 0.51 |

**HistGradientBoosting** and **GradientBoosting** perform comparably on LegSummary, slightly edging out LightGBM on raw accuracy, though LightGBM leads on `flat` F1.

---

## 🔝 Top Features (LightGBM — by gain)

**FltSummary:**
`TOTBKD` › `FLTNUM` › `Month` › `PLF` › `RPK` › `lowD` › `RPKG_pct`  › `highD`

**LegSummary:**
`TOTBKD` › `PLF` › `FLTNUM` › `Month` › `RPKG_pct` › `lowD` › `RPK` › `highD`

Total bookings (`TOTBKD`), load factor (`PLF`), and the flight identifier (`FLTNUM`) are consistently the most predictive signals.





)
# Returns: predicted class (down/flat/up) + probability vector
```
