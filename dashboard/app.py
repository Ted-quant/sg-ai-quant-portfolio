from pathlib import Path
import pandas as pd
import streamlit as st


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="AI Quant Portfolio Dashboard",
    layout="wide",
)


# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]

PORTFOLIO_SUMMARY_PATH = ROOT_DIR / "backtest" / "results" / "final_model_portfolio_summary.csv"
ASSET_SUMMARY_PATH = ROOT_DIR / "backtest" / "results" / "final_model_asset_summary.csv"
EQUITY_CURVE_PATH = ROOT_DIR / "backtest" / "results" / "final_model_equity_curves.csv"
EQUITY_CHART_PATH = ROOT_DIR / "reports" / "charts" / "final_model_robust_assets_equity_curve.png"


# =========================
# Helper Functions
# =========================

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists."""
    if not path.exists():
        st.error(f"Missing file: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def format_percent(value: float) -> str:
    """Format decimal return values as percentages."""
    try:
        return f"{value * 100:.2f}%"
    except Exception:
        return "N/A"


def format_number(value: float) -> str:
    """Format numeric values."""
    try:
        return f"{value:.2f}"
    except Exception:
        return "N/A"


# =========================
# Title
# =========================

st.title("AI-Based Quantitative Portfolio Allocation Dashboard")

st.write(
    """
    This dashboard summarizes the final ExtraTrees-based machine learning portfolio model.
    
    The goal of the strategy is not to maximize raw return, but to improve risk-adjusted
    performance through volatility reduction, drawdown control, and downside-risk management.
    """
)


# =========================
# Final Model Configuration
# =========================

st.header("Final Model Configuration")

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    st.metric("Model", "ExtraTrees")
    st.metric("Target", "5D Return > +1%")

with config_col2:
    st.metric("Feature Set", "Full Features")
    st.metric("Position Sizing", "Binary")

with config_col3:
    st.metric("Portfolio", "Robust Assets")
    st.metric("Assets", "D05_SI, NVDA, SPY")


# =========================
# Load Results
# =========================

portfolio_df = load_csv(PORTFOLIO_SUMMARY_PATH)
asset_df = load_csv(ASSET_SUMMARY_PATH)
equity_df = load_csv(EQUITY_CURVE_PATH)


# =========================
# Portfolio-Level Results
# =========================

st.header("Portfolio-Level Final Backtest Result")

if not portfolio_df.empty:
    display_df = portfolio_df.copy()

    percent_cols = ["total_return", "volatility", "mdd", "var_95", "es_95"]
    number_cols = ["sharpe", "calmar"]

    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_percent)

    for col in number_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(format_number)

    st.dataframe(display_df, use_container_width=True)

    try:
        buy_hold = portfolio_df[portfolio_df["strategy"] == "Buy & Hold"].iloc[0]
        final_ml = portfolio_df[portfolio_df["strategy"] == "Final ExtraTrees ML"].iloc[0]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric(
                "Sharpe Improvement",
                format_number(final_ml["sharpe"]),
                delta=f"{final_ml['sharpe'] - buy_hold['sharpe']:.2f}",
            )

        with metric_col2:
            st.metric(
                "MDD Reduction",
                format_percent(final_ml["mdd"]),
                delta=f"{(final_ml['mdd'] - buy_hold['mdd']) * 100:.2f} pp",
            )

        with metric_col3:
            st.metric(
                "Volatility Reduction",
                format_percent(final_ml["volatility"]),
                delta=f"{(final_ml['volatility'] - buy_hold['volatility']) * 100:.2f} pp",
            )

        with metric_col4:
            st.metric(
                "Calmar Improvement",
                format_number(final_ml["calmar"]),
                delta=f"{final_ml['calmar'] - buy_hold['calmar']:.2f}",
            )

    except Exception:
        st.warning("Could not calculate comparison metrics.")

else:
    st.warning("Portfolio summary file is not available.")


# =========================
# Equity Curve
# =========================

st.header("Equity Curve")

if EQUITY_CHART_PATH.exists():
    st.image(str(EQUITY_CHART_PATH), caption="Buy & Hold vs Final ExtraTrees ML")
elif not equity_df.empty:
    st.line_chart(equity_df)
else:
    st.warning("Equity curve data is not available.")


# =========================
# Asset-Level Results
# =========================

st.header("Asset-Level Final Backtest Result")

if not asset_df.empty:
    display_asset_df = asset_df.copy()

    percent_cols = [
        "model_return",
        "benchmark_return",
        "model_volatility",
        "benchmark_volatility",
        "model_mdd",
        "benchmark_mdd",
        "model_var_95",
        "model_es_95",
        "market_exposure",
        "cost_drag",
    ]

    number_cols = [
        "model_sharpe",
        "benchmark_sharpe",
        "model_calmar",
        "benchmark_calmar",
    ]

    for col in percent_cols:
        if col in display_asset_df.columns:
            display_asset_df[col] = display_asset_df[col].apply(format_percent)

    for col in number_cols:
        if col in display_asset_df.columns:
            display_asset_df[col] = display_asset_df[col].apply(format_number)

    st.dataframe(display_asset_df, use_container_width=True)
else:
    st.warning("Asset summary file is not available.")


# =========================
# Interpretation
# =========================

st.header("Interpretation")

st.write(
    """
    The final ML strategy produced a lower raw return than Buy & Hold, but it improved
    several important risk-adjusted performance metrics.

    The key result is that the model reduced volatility and maximum drawdown while improving
    Sharpe and Calmar ratios. This supports the interpretation that the final strategy is a
    risk-controlled portfolio allocation model rather than a pure return-maximization model.
    """
)

st.subheader("Key Portfolio Insight")

st.code(
    """
Buy & Hold produced higher raw return.
Final ExtraTrees ML produced stronger risk-adjusted performance.
    """,
    language="text",
)


# =========================
# Future Extensions
# =========================

st.header("Future Extensions")

st.write(
    """
    Potential future extensions include:
    
    - QuantConnect / LEAN implementation
    - Intraday data testing using hourly or 15-minute bars
    - More detailed transaction cost and slippage modeling
    - Alpaca paper-trading signal pipeline
    - Expanded asset universe including SGX, US equities, ETFs, and FX
    """
)
