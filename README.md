# Machine Learning Based Market Regime & Risk Prediction

A student-level quantitative research project that classifies the **next 20-trading-day market regime** for SPY and evaluates whether the predicted regime contains useful risk-management information.

The project deliberately avoids exact price prediction and avoids treating model probability as a probability of profit.

## Research questions

- What market regime are we currently experiencing?
- Which market variables help classify the regime?
- Can simple ML models classify regimes out of sample?
- Does realized risk differ across predicted regimes?
- Does a simple regime-based exposure rule add useful risk information relative to Buy & Hold?

## Data

Daily historical data are downloaded with `yfinance`:

- SPY — primary asset
- QQQ — growth/equity-market context
- TLT — long-duration Treasury context
- GLD — gold context
- VIX — volatility/stress proxy
- ^TNX — 10-year Treasury yield index
- DX-Y.NYB — U.S. Dollar Index

The default research window is 2015-01-01 through 2025-12-31. Raw downloads are stored locally under `data/raw/`.

## Target

The target is the **forward 20-trading-day SPY return**.

Thresholds are not tuned on the test set. They are derived from the training-period forward-return distribution:

- below lower training tertile → Bearish
- between the two training tertiles → Neutral
- above upper training tertile → Bullish

The final 20 observations of the training and validation windows are purged so their labels cannot depend on returns from the following evaluation period.

## Models

1. Logistic Regression — interpretable baseline
2. Random Forest — nonlinear tree-based comparison

XGBoost is intentionally omitted from the default implementation. It should only be added if there is a clear methodological reason and a meaningful out-of-sample improvement.

## Evaluation

Model selection uses validation macro-F1. The test set is kept out of model selection.

Reported classification metrics include:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Class-level precision/recall
- Confusion matrix

The risk analysis groups actual next-day market behavior by the model's **out-of-sample predicted regime**.

## Trading / risk overlay

For research purposes only:

- Bullish → 100% SPY exposure
- Neutral → 50% SPY exposure
- Bearish → 0% SPY exposure

A simple 5 bps transaction cost is applied when exposure changes.

This is an illustrative decision-support rule, not an optimized portfolio strategy.

## Project structure

```text
market-regime-ml/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── target.py
│   ├── models.py
│   ├── evaluation.py
│   ├── risk_analysis.py
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   └── run_research.py
├── notebooks/
├── results/
│   ├── charts/
│   ├── metrics/
│   └── predictions/
├── app.py
├── requirements.txt
└── README.md
```

## How to run

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download real market data

```bash
python scripts/download_data.py
```

### 4. Run the research pipeline

```bash
python scripts/run_research.py
```

This creates:

- `results/metrics/research_summary.json`
- `results/metrics/feature_importance.csv`
- `results/metrics/regime_summary.csv`
- `results/predictions/oos_predictions.csv`
- `results/predictions/strategy_returns.csv`

### 5. Launch the dashboard

```bash
streamlit run app.py
```

## Important methodological controls

- No random train/test shuffle.
- Scaling is fitted only inside the training pipeline.
- Target thresholds are derived from training data.
- Boundary labels are purged by the 20-day forecast horizon.
- Model selection is based on validation data, not the test set.
- Test performance is reported as genuinely out-of-sample.
- No synthetic market prices are included.
- No performance numbers are fabricated.

## Limitations

- Historical relationships can change.
- Regime labels depend on target construction.
- The market universe is intentionally small.
- Transaction costs are simplified.
- There is no market-impact model or live execution.
- Model probabilities are estimates, not certainty.
