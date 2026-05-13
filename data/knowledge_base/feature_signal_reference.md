# Feature Signal Reference — Fraud Detection System

This document maps each model feature to its fraud-detection rationale, expected range,
and which fraud typologies it is most predictive for.

---

## Deviation Features

### `z_amount` — Z-score of transaction amount vs. user rolling baseline
- **Formula:** `(amount - amount_mean_7tx) / (amount_std_7tx + ε)`
- **Range (clipped):** [-10, 10]
- **High-risk threshold:** |z| > 3.0 (3 standard deviations)
- **Most predictive for:** Romance scams (large outlier wire), structuring (near-threshold ATM)
- **Pitfall:** New users have std=0 → ε prevents division by zero, but first-transaction
  Z-scores are unreliable. Use EWMA as backup.

### `ewma_z_amount` — EWMA-normalised deviation
- **Formula:** `(amount - amount_ewma) / (amount_ewma_std + ε)`
- **Advantage over z_amount:** EWMA adapts to recent spending shifts (e.g., holiday season,
  new job) with exponential decay. A user who recently started spending more won't be
  flagged for every above-average purchase.
- **Most predictive for:** Gradual elderfinancial exploitation (EWMA catches slow drift)

---

## Velocity Features

### `tx_count_1h` — Transactions in the last 60 minutes
- **Computed by:** Rolling time-window count of prior transactions per user (shift(1) prevents look-ahead)
- **High-risk threshold:** > 5 transactions
- **Most predictive for:** Account takeover (ATO) burst patterns, lottery scam

### `tx_count_24h` — Transactions in the last 24 hours
- **High-risk threshold:** > 20 transactions
- **Most predictive for:** Multi-day ATO, gradual exploitation

### `amount_sum_24h` — Total amount transacted in last 24 hours
- **High-risk threshold:** > 3× user's typical daily spend
- **Most predictive for:** Romance scam (large single-day wire), account takeover

---

## Temporal Features

### `is_night` — Flag for transactions between 00:00–05:59 UTC
- **Binary:** 0 or 1
- **Rationale:** Card-not-present fraud, ATO, and lottery scam disproportionately occur
  during night hours when the legitimate cardholder is asleep
- **Note:** UTC time — accounts for US time zones being UTC-5 to UTC-8

### `hour_sin` / `hour_cos` — Cyclical encoding of hour
- **Rationale:** Raw hour (0–23) has a discontinuity at midnight (23 → 0). Cyclical encoding
  using sin/cos preserves the 23→0 continuity, which is important for linear models and
  neural networks that can't learn this wrap-around relationship from raw integers.
- **Formula:** `sin(2π × hour / 24)`, `cos(2π × hour / 24)`

### `dow_sin` / `dow_cos` — Cyclical encoding of day-of-week
- **Rationale:** Same principle — Sunday (6) and Monday (0) are adjacent in calendar terms
  but have a raw value gap of 6.

### `is_weekend` / `is_holiday`
- **Binary flags:** 1 if transaction occurs on weekend or US federal holiday
- **Rationale:** Reduced bank staffing on weekends/holidays is exploited by fraudsters
  who know disputes take longer to resolve

---

## Counterparty & Merchant Features

### `is_new_counterparty` — First-time payee flag
- **Binary:** 0 or 1
- **Computation:** Sliding 30-counterparty window per user — flag if counterparty never seen
- **Most predictive for:** Romance scams, lottery scams, ATO
- **Interview note:** "New-counterparty flag is one of the highest-signal features for elder
  fraud because romance scammers almost always appear as brand-new payees."

### `mcc_high_risk` — High-risk merchant category flag
- **High-risk MCCs:** 6099 (money transfer), 6012 (wire), 6011 (ATM), 7995 (gambling), 5912 (pharmacy)
- **Binary:** 0 or 1

---

## Geographic Features

### `geo_speed_kmh` — Travel speed between consecutive transactions (km/h)
- **Formula:** Haversine distance / elapsed hours between transactions
- **High-risk threshold:** > 900 km/h (faster than commercial flight = impossible)
- **Most predictive for:** Account takeover (stolen credentials used from remote location)

### `geo_impossible` — Binary impossible-travel flag
- **Binary:** 1 if `geo_speed_kmh > 900`
- **Note:** One of the strongest ATO signals — it's a near-physical-law constraint

---

## Interaction Features

### `amount_x_night` = `amount × is_night`
- **Rationale:** Large transactions at night are disproportionately fraudulent. The interaction
  term captures this joint effect explicitly for linear/log-linear models.

### `amount_x_new_payee` = `amount × is_new_counterparty`
- **Rationale:** Small payments to new payees are normal (new coffee shop). Large payments to
  new payees are suspicious (romance scam wire).

### `velocity_x_new_payee` = `tx_count_1h × is_new_counterparty`
- **Rationale:** Multiple fast transactions to new payees = account takeover signature

---

## One-Hot Encoded Transaction Types

| Column | Payment Rail | Risk Notes |
|--------|-------------|------------|
| `tx_type_WIRE` | Wire transfer (SWIFT/DOM) | Highest risk — large amounts, irreversible |
| `tx_type_ACH` | ACH batch | Medium risk — lottery scams, exploitation |
| `tx_type_ZELLE` | Zelle P2P | High for ATO — instant, irreversible |
| `tx_type_ATM` | ATM cash | Structuring risk |
| `tx_type_ONLINE` | Card-not-present e-commerce | CNP fraud risk |
| `tx_type_POS` | Point-of-sale chip/tap | Lowest risk |

---

## Feature Importance (Typical XGBoost Ranking)

From typical training runs on synthetic dataset:

1. `z_amount` / `ewma_z_amount` — Amount deviation (strongest signal)
2. `is_new_counterparty` — New payee
3. `tx_count_1h` — Velocity burst
4. `mcc_high_risk` — High-risk merchant
5. `geo_impossible` — Impossible travel
6. `is_night` — Night transaction
7. `amount_x_new_payee` — Interaction: large + new
8. `tx_type_WIRE` — Wire transfer dummy
9. `amount_sum_24h` — Daily total
10. `hour_sin` / `hour_cos` — Cyclical hour

---

## Training-Serving Skew Prevention

All features in this list use `shift(1)` (exclude current transaction) in rolling windows
to prevent look-ahead leakage:
- `amount_mean_7tx`, `amount_std_7tx`, `amount_ewma`, `amount_ewma_std`
- `tx_count_1h`, `tx_count_24h`, `tx_count_7d`, `amount_sum_24h`

Verify this in production by comparing feature distributions at training time vs. serving time.
PSI > 0.2 on any of these features indicates a potential calculation bug or data pipeline issue.

