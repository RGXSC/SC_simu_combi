# 📊 Supply Chain Agility Simulator

Combinatorial batch simulator to compare **Push (centralized stock, long lead times)** vs **Agile (distributed stock, short lead times)** supply chain strategies.

## 🎯 What it does

Runs **millions of scenarios** combining:
- Demand levels (weighted by log-normal probability)
- Lead times per supply chain stage
- Stock levels and distribution along the supply chain
- Planning frequencies, capacity ramp rates
- Fixed/variable cost structures

Then uses **machine learning (regression tree + random forest)** to identify which factors drive profitability.

## 🚀 Deploy on Streamlit Cloud

1. **Fork this repo** on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your forked repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done!

## 💻 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📋 Default Parameters

| Parameter | Value |
|-----------|-------|
| Price | €1,000 |
| Variable cost | €200/unit |
| Stock levels | 500, 900, 1,300 units |
| Demand center | 40% of forecast |
| Uncertainty (σ) | 0.6 |
| Demand splits A/B | 60%, 80% |
| Sim periods | 13, 26 weeks |
| Fixed costs | 20%, 30%, 40% |

All parameters are adjustable in the sidebar.

## 📦 Files

| File | Description |
|------|-------------|
| `app.py` | Main Streamlit application |
| `sim_engine.py` | Simulation engine (pure Python, no dependencies) |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Theme configuration |

## 📊 Outputs

- **Interactive dashboard** in Streamlit
- **HTML report** (standalone, shareable by email)
- **CSV export** of financial results

## 🧠 Methodology

- **Demand**: Log-normal distribution with adjustable center (mode) and spread (σ)
- **Stock cost**: Initial stock is included in variable costs (prevents "free stock" bias)
- **Comparison**: Equal stock investment across all strategies for fair comparison
- **Weighting**: All averages weighted by demand probability
- **ML**: Regression tree for interpretability, random forest for accuracy
