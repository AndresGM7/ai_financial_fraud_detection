# BSA / AML & Regulatory Framework for Financial Fraud Detection

---

## 1. Bank Secrecy Act (BSA) — 31 U.S.C. § 5311 et seq.

The Bank Secrecy Act requires US financial institutions to assist government agencies in
detecting and preventing money laundering. Key obligations:

### Currency Transaction Reports (CTR)
- **Trigger:** Cash transactions > $10,000 in a single business day (aggregated per customer)
- **Deadline:** Filed with FinCEN within 15 calendar days
- **Structuring violation:** Breaking up transactions to stay below the CTR threshold is a federal
  crime under 31 U.S.C. § 5324, punishable by up to 5 years imprisonment
- **Detection signal:** Multiple ATM withdrawals of $8,000–$9,999 within 24–72 hours

### Suspicious Activity Reports (SAR)
- **Trigger:** Transaction involving ≥ $5,000 where the institution knows, suspects, or has reason
  to suspect the funds involve a criminal offense or are designed to evade BSA requirements
- **Deadline:** Filed within 30 days of detection (60 days if no known subject)
- **Key elder fraud indicators that should trigger SAR consideration:**
  - Large wire transfers to overseas entities inconsistent with customer profile
  - Structuring patterns
  - Rapid movement of funds through multiple accounts (layering)
  - Customer confusion about account activity or apparent coaching

---

## 2. GLBA — Gramm-Leach-Bliley Act (Consumer Privacy)

Requires financial institutions to protect customers' nonpublic personal information (NPI).

**Relevance to fraud detection ML models:**
- Model inputs that use transaction history must comply with GLBA data-use limitations
- Any fraud score used in an adverse action (e.g., account restriction) must be explainable
  under the **Equal Credit Opportunity Act (ECOA)** adverse action requirements
- **SHAP values** provide the per-transaction feature contributions required for explainability

---

## 3. Elder Abuse Statutes

### Adult Protective Services (APS)
- All 50 US states have APS agencies that investigate elder abuse
- Financial institutions in many states have **mandatory reporting** obligations for suspected
  elder financial exploitation (EFE)
- Carefull's platform serves as an early-warning system to trigger APS referrals

### State-Specific Laws
- **California:** WIC § 15630.1 — financial institutions must report suspected elder financial abuse
- **Florida:** § 415.1034 — financial institutions may delay transactions suspected of EFE
- **New York:** GOL § 9-601 — "hold" authority for suspected exploitation

---

## 4. AML Transaction Monitoring Requirements

### Risk-Based Approach
- FFIEC guidance requires institutions to adopt a **risk-based approach** to AML
- High-risk customers (elderly, large wire activity, money-transfer-heavy) require enhanced monitoring
- Machine learning models must be **validated** periodically and their decisions documented

### Model Risk Management (OCC SR 11-7)
- Supervisory guidance on model risk management applies to fraud detection ML models
- Requires: **conceptual soundness**, appropriate data, testing, ongoing monitoring
- **Explainability requirement:** Decision engine must produce human-readable reasons for alerts
  — this is why SHAP values and LLM reasoning traces are built into this system

---

## 5. Key MCC Codes for Fraud Risk Tiers

| MCC | Description | Risk Level |
|-----|-------------|------------|
| 6099 | Money Transfer / Money Service Business | **HIGH** |
| 6012 | Wire Transfer / SWIFT | **HIGH** |
| 6011 | ATM Cash Dispensing | **HIGH** |
| 7995 | Gambling / Lottery | **HIGH** |
| 5912 | Drug Stores / Pharmacies (gift cards) | **MEDIUM** |
| 5411 | Grocery Stores | LOW |
| 5812 | Restaurants | LOW |
| 4121 | Ride-hailing (Uber/Lyft) | LOW |

---

## 6. Velocity Thresholds — Industry Benchmarks

Based on industry fraud benchmarks:

| Metric | Low Risk | Medium Risk | High Risk |
|--------|----------|-------------|-----------|
| Transactions / hour | ≤ 2 | 3–5 | > 5 |
| Transactions / day | ≤ 15 | 16–30 | > 30 |
| Amount vs. user baseline (Z-score) | < 2.0 | 2.0–4.0 | > 4.0 |
| Wire amount (single tx) | < $1,000 | $1,000–$5,000 | > $5,000 |
| New counterparty rate (30-day) | < 10% | 10–30% | > 30% |

---

## References

- FinCEN: https://www.fincen.gov/resources/statutes-and-regulations
- FFIEC BSA/AML Examination Manual: https://bsaaml.ffiec.gov/
- OCC SR 11-7: Model Risk Management Guidance
- CFPB: Elder Financial Exploitation Resources

