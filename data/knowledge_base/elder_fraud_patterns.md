# Elder Financial Fraud Patterns
**Sources: CFPB, AARP, FBI Elder Fraud Report 2023**

---

## 1. Romance / Relationship Scam

**Description**  
Fraudster builds a fictitious emotional relationship with an older adult through online platforms
(Facebook, dating sites, WhatsApp). After weeks or months of trust-building, the fraudster
invents an emergency (medical bills, business investment, travel costs) and requests wire
transfers or gift cards.

**Key Red Flags**
- Large first-time wire transfer ($3,000–$50,000) to an unknown overseas entity
- New counterparty with no prior transaction history
- WIRE or ACH transaction type
- Night or weekend hours
- Victim claims to have "met online" or "never met in person"
- Gradual escalation of transfer amounts over weeks (CUSUM signal)

**Average Loss:** $10,000–$15,000  
**FBI Ranking:** #1 fraud type by dollar loss for adults 60+

---

## 2. Tech Support Scam

**Description**  
Caller or pop-up impersonates Microsoft, Apple, or an antivirus company. Victim is told their
computer is "infected" and they must pay for remote support or buy gift cards. Scammer may
request remote access to the victim's device.

**Key Red Flags**
- Gift card merchant purchases followed by rapid ACH or wire
- Multiple transactions in short succession (velocity spike)
- Unusual hour (2–5 AM)
- New merchant with no prior transaction history
- Small to medium amounts ($200–$3,000) per transaction
- Counterparty name contains "Tech Support", "Microsoft", "Apple"

**Average Loss:** $500–$3,000

---

## 3. Lottery / Prize Scam ("You've Won!")

**Description**  
Victim receives an email, letter, or phone call claiming they have won a lottery or prize. To
claim the prize, the victim must pay taxes or processing fees — typically via Western Union,
MoneyGram, or gift cards. There is no prize.

**Key Red Flags**
- Multiple small ACH transfers ($200–$900) to money-transfer MCCs (MCC 6099)
- Transfers to Western Union, MoneyGram, or similar services
- Rapid succession: 3–8 payments within 2 hours
- New counterparty: "PRIZE CLAIM CENTER", "WINNERS FUND", etc.
- Night or weekend activity
- Counterparty based overseas

**Average Loss:** $800–$1,500

---

## 4. Account Takeover (ATO)

**Description**  
Fraudster obtains the victim's online banking credentials through phishing, data breach, or
social engineering. They then log in and initiate rapid transfers to money mule accounts via
Zelle, ACH, or wire before the victim notices.

**Key Red Flags**
- Impossible geographic velocity (geo_impossible flag) — device in a different city/country
- More than 5 Zelle or ACH transactions in under 60 minutes
- All transactions to first-time counterparties
- Unusual login time (early morning hours)
- Login from new device or IP (not directly observable in transaction data)
- Large burst of small-to-medium transfers ($50–$500 each)

**Average Loss:** $5,000–$20,000

---

## 5. Structuring / Smurfing (BSA Evasion)

**Description**  
Individual (often a family member, caregiver, or financial exploiter) makes multiple ATM cash
withdrawals just below the IRS/FinCEN $10,000 cash transaction reporting (CTR) threshold to
avoid Bank Secrecy Act (BSA) reporting requirements. This is a federal crime (31 U.S.C. § 5324).

**Key Red Flags**
- Multiple ATM withdrawals between $8,000 and $9,799
- 2–5 withdrawals within a 24–72 hour window
- Counterparty is always "CASH"
- TX type: ATM
- MCC: 6011 (Automated Cash Disbursements)
- No unusual merchant activity — the pattern is in repetition + amount band

**Average Loss:** $25,000–$100,000 (often discovered only after significant exploitation)

---

## 6. Elder Financial Exploitation (EFE) — Insider

**Description**  
A trusted person — family member, caregiver, power of attorney, or financial advisor — gradually
siphons funds from the victim's accounts. Unlike sudden-onset scams, EFE is characterised by
slow escalation that individual Z-score tests miss. CUSUM is the appropriate detector.

**Key Red Flags**
- Gradual escalation of transfer amounts over weeks/months (CUSUM alert)
- Unusual beneficiary changes or new POA activity
- Transfers to accounts with shared surname (family exploitation)
- Reduction in account balance not explained by known expenses
- Victim shows signs of isolation or cognitive decline
- Caregiver with unusual access patterns

**Average Loss:** $50,000+ (highest of all elder fraud types)

---

## 7. Card-Not-Present (CNP) Fraud

**Description**  
Stolen card number or credentials used for online purchases, typically between 1–5 AM when
the legitimate cardholder is asleep. Common at merchants the cardholder has never used.

**Key Red Flags**
- ONLINE transaction type
- Night hours (1–5 AM)
- New merchant (never transacted with before)
- Multiple transactions in short succession
- Small to medium amounts ($50–$500)
- MCC: 5999 (Miscellaneous Retail), 5734 (Computer Software), 7372 (Computer Services)

**Average Loss:** $200–$600 per incident

---

## References

- CFPB: *Protecting Older Consumers* (2023 report)
- AARP Fraud Watch Network: *Top Scams 2023*
- FBI Internet Crime Complaint Center (IC3): *Elder Fraud Report 2023*
- FinCEN: *BSA Requirements for Financial Institutions* (31 U.S.C. § 5318)
- OCC: *Bank Secrecy Act / Anti-Money Laundering Examination Procedures*

