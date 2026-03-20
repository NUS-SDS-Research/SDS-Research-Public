# AI-Powered ESG Analytics for Cryptocurrency Investment Decision-Making

---

## What This Project Is About

This project aims to evaluate whether ESG (Environmental, Social, and Governance) characteristics affect the return and risk profiles of cryptocurrencies. 50 coins are scored across E, S, and G pillars, grouped into Low / Mid / High ESG categories, and then put through a range of statistical and machine learning analyses.

---

## Setup

### Requirements

Python 3.11+ is recommended. Install dependencies with:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn openpyxl
```

### Data Files

Make sure the following files are in your working directory before running anything:

- `price.csv` —> daily OHLCV prices for all 50 coins
- `Data.xlsx` —> ESG scores, read from the `Summary` sheet
- `cryptos` —> the merged prices + ESG dataset used in `Data_Exploration.ipynb`

---

## Notebooks


`Analysis.ipynb` is the main notebook. **Run blocks in order from top to bottom**, as later blocks depend on variables defined earlier. Here's what each block does:

| Block | What It Does |
|---|---|
| **Block 1: Set Up** | Imports all libraries, loads `price.csv` and `Data.xlsx`, merges into a single working dataframe |
| **Block 2: Financial Metrics** | Computes daily returns, 60-day rolling volatility, and risk-adjusted returns (return ÷ volatility) |
| **Block 3: ESG Grouping** | Classifies coins into Low / Mid / High ESG using total score cutoffs (0–4 / 4–7 / 7–12) |
| **Block 4: Return Analysis** | Plots average monthly returns by ESG group; visual check for monotonic patterns |
| **Block 5: Volatility Analysis** | Time-series volatility curves per ESG group; highlights realised risk differences |
| **Block 6: Sharpe Ratio & ANOVA** | Annualised Sharpe ratios per group; one-way ANOVA to test if differences are statistically significant |
| **Block 7: Maximum Drawdown** | Worst peak-to-trough loss per coin, averaged by ESG group; ANOVA test |
| **Block 8: Calendar Effects** | Three sub-tests: turn-of-month effect, weekend premium, holiday effect, all broken down by ESG group |
| **Block 9: Cross-Sectional Regression** | OLS regression of E, S, G pillar scores on average risk-adjusted return; includes VIF checks for multicollinearity |
| **Block 10: Machine Learning** | Random Forest and Gradient Boosting models; tests ESG predictive power with and without the `month` variable; outputs feature importance charts |
| **Block 11: Summary & Implications** | Narrative summary and investor takeaways rendered as formatted HTML in the notebook |

---

## ESG Scoring Reference

Scores are assigned per coin based on the rubric from Luo & Adelopo (2025):

| Pillar | 0 | Max | Max Score |
|---|---|---|---|
| **Environmental (E)** | Energy-intensive Proof-of-Work | Actively carbon-addressing | 3 |
| **Social (S)** | Meme coin, no utility | Social mission-focused | 6 |
| **Governance (G)** | Centralised control | Robust council/community governance | 3 |

Total scores are then binned: **Low = 0–4**, **Mid = 4–7**, **High = 7–12**.

---

## A Note on the ML Block

The machine learning section (Block 10) runs two models, Random Forest and Gradient Boosting, each in two configurations: once with only ESG features, and once with ESG + a `month` variable. Cross-validated R² is reported for each. The feature importance plots are the most informative output here and show how much of the predictive signal comes from ESG versus time.

---

## What We Found

All of the below comes directly from the output cells and summary blocks in `Analysis.ipynb`, so you can trace every number back to a specific block.

**ESG does explain risk-adjusted returns, though not all pillars pull their weight equally.** Running an OLS regression across all 50 coins, we found that E, S, and G together account for roughly 45% of the variation in risk-adjusted returns (R² = 0.45, p < 0.001). Of the three pillars, only the Social score came out positively and significantly associated with better performance (β = 0.017, p = 0.003). Coins with stronger social utility, think DeFi protocols and DApps, tended to hold up better on a risk-adjusted basis. The Governance score was actually significantly *negative* (p = 0.001), which we think reflects the fact that decentralised governance in crypto is still quite immature.

**Low ESG coins had dramatically worse Sharpe ratios, and it wasn't close.** Mean annualised Sharpe ratios came out at -2.86 for Low ESG, compared to -0.72 and -0.73 for Mid and High respectively (ANOVA: F = 4.73, p = 0.013). Meme coins and pure Proof-of-Work tokens drove most of that gap.

**High ESG coins are not the safe haven you might expect.** This was probably our most surprising finding. High ESG coins actually had the worst drawdowns, averaging around -75% peak-to-trough, versus -62% for Mid and -50% for Low (ANOVA: F = 3.95, p = 0.026). During the Crypto Crash 2025 window they also suffered the steepest daily losses on average (~-0.79%). Higher ESG does not mean lower risk, at least not in this sample.

**Calendar effects showed up clearly, particularly for High ESG coins.** The turn-of-month effect in crypto is actually *negative*, which is the opposite of what you typically see in equity markets. Mid (p = 0.049) and High ESG coins (p = 0.008) both showed significantly lower returns around month boundaries. On the weekend side, High ESG coins earned roughly +0.40% more on weekends than weekdays (t = 3.01, p = 0.003), likely reflecting retail trading flows when traditional markets are closed. The holiday effect was positive in direction (+0.42% to +0.54%) but did not reach significance for any group.

**ESG alone is a weak predictor, and the time (month) variable does most of the heavy lifting.** Our Random Forest and Gradient Boosting models both returned CV R² of around 0.01 using only ESG features. Once we added the `month` variable, that jumped to 0.46-0.50, with month accounting for 65-83% of feature importance across both models. The takeaway is that market regime and timing effects dominate, and ESG contributes relatively little predictive power on its own.

