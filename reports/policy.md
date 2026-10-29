## Operating policy
**Book.** 930 real listed Indian non-financial companies scored at 2026-09-30 (frozen universe, `data/reference/universe_frozen.csv`; a proxy for the brief's 800-borrower book, not simulated).
**Alert threshold.** Each quarter, flag the top k = min(40, floor(5% x N)) by composite score (this quarter: k = 40, 40 alerts). This is a hard cap, so the review team never receives more than 40 names.
**Quarterly cycle.**
1. T+5 days: refresh filings and prices. Use only statements public by T (FY end + 183 days). Score every borrower.
2. Flagged ranks 1-10 -> **ESCALATE**: senior credit officer within 2 weeks; covenant test, bank-statement/GST cross-check, management call.
3. Other flagged names -> **REVIEW**: analyst deep-dive this quarter using the three plain-language drivers; confirm or dismiss within 30 days.
4. Ranks up to 80 -> **WATCH**: monthly monitoring of price, ratings and pledges.
5. **Override, escalate regardless of rank**: downgrade to sub-investment grade or by 2 or more notches; auditor resignation, qualification or disclaimer; results deferred past the LODR deadline; promoter pledge above 50% or up more than 10 pp in a quarter; any lender reporting SMA-1/SMA-2 (CRILC).
6. **Escalation path**: analyst -> credit committee (watch-list tag, limit freeze, extra security) -> CRO/Board risk committee above delegated limits -> RBI June 2019 prudential framework (inter-creditor agreement) on SMA-2 or confirmed stress.

**Threshold and cost.** A missed default costs 20.0 and a false alarm costs 1.0. Flagging is worth it when (1-p) x 1.0 < p x 20.0, i.e. **p > 1/21 = 0.0476** (Polish validation cost optimum: 0.040).
Capacity binds, so the operating rule is "top-k capped at 40; within that, everyone with PD > 1/21 is reviewed first".
At an *assumed* 2.5% annual base rate, model-expected cost this quarter is **353.7** units with capped top-k, versus 451.3 with no reviews and 323.1 for an uncapped Bayes rule (118 alerts, which exceeds capacity).
