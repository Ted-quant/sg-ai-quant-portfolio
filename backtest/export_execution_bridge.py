"""Export final ML signals and position changes for execution-cost analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from final_model_backtest import (
    ASSETS,
    FEATURE_COLS,
    REGIME_FILTERS,
    SIGNAL_THRESHOLDS,
    START_DATE,
    TEST_START,
    TOTAL_COST_PER_POSITION_CHANGE,
    TRAIN_END,
    VALID_END,
    VALID_START,
    backtest_signal,
    build_model,
    build_signal,
    calculate_metrics,
    create_features,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

BRIDGE_TEST_END = "2026-07-06"

# yfinance treats the end date as exclusive.
# Data through 2026-07-13 provides the five future trading days
# needed to calculate the target for 2026-07-06.
DOWNLOAD_END_EXCLUSIVE = "2026-07-14"

SNAPSHOT_PATH = (
    REPO_ROOT
    / "data"
    / "snapshots"
    / "final_model_bridge_prices_2026-07-06.csv"
)

DAILY_BRIDGE_PATH = (
    REPO_ROOT
    / "backtest"
    / "results"
    / "final_model_execution_bridge.csv"
)

ORDER_BRIDGE_PATH = (
    REPO_ROOT
    / "backtest"
    / "results"
    / "final_model_execution_orders.csv"
)


def download_fixed_snapshot() -> pd.DataFrame:
    """Download and combine a fixed adjusted-close price snapshot."""
    frames = []

    for asset_name, ticker in ASSETS.items():
        print(
            f"Downloading fixed price snapshot for "
            f"{asset_name} ({ticker})..."
        )

        dataframe = yf.download(
            ticker,
            start=START_DATE,
            end=DOWNLOAD_END_EXCLUSIVE,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if dataframe.empty:
            raise ValueError(
                f"No price data downloaded for {ticker}."
            )

        if isinstance(dataframe.columns, pd.MultiIndex):
            dataframe.columns = dataframe.columns.get_level_values(0)

        if "Close" not in dataframe.columns:
            raise ValueError(
                f"Close column was not found for {ticker}."
            )

        price_frame = dataframe[["Close"]].copy()
        price_frame.columns = ["close"]
        price_frame = price_frame.reset_index()

        date_column = price_frame.columns[0]
        price_frame = price_frame.rename(
            columns={date_column: "date"}
        )

        price_frame["date"] = (
            pd.to_datetime(price_frame["date"])
            .dt.tz_localize(None)
        )

        price_frame["asset"] = asset_name
        price_frame["ticker"] = ticker

        frames.append(
            price_frame[
                [
                    "date",
                    "asset",
                    "ticker",
                    "close",
                ]
            ]
        )

    snapshot = pd.concat(
        frames,
        ignore_index=True,
    )

    snapshot = snapshot.sort_values(
        [
            "asset",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    return snapshot


def validate_snapshot(snapshot: pd.DataFrame) -> None:
    """Validate the stored bridge price snapshot."""
    required_columns = {
        "date",
        "asset",
        "ticker",
        "close",
    }

    missing_columns = (
        required_columns
        - set(snapshot.columns)
    )

    if missing_columns:
        raise ValueError(
            "Snapshot is missing columns: "
            f"{sorted(missing_columns)}"
        )

    missing_assets = (
        set(ASSETS)
        - set(snapshot["asset"].unique())
    )

    if missing_assets:
        raise ValueError(
            "Snapshot is missing assets: "
            f"{sorted(missing_assets)}"
        )

    duplicate_rows = snapshot.duplicated(
        subset=[
            "asset",
            "date",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            "Snapshot contains duplicate asset-date rows."
        )

    if snapshot["close"].isna().any():
        raise ValueError(
            "Snapshot contains missing close prices."
        )


def load_or_create_snapshot() -> pd.DataFrame:
    """Load the fixed snapshot or create it on the first run."""
    if SNAPSHOT_PATH.exists():
        print(
            f"Loading existing snapshot:\n"
            f"{SNAPSHOT_PATH}"
        )

        snapshot = pd.read_csv(
            SNAPSHOT_PATH,
            parse_dates=["date"],
        )
    else:
        print(
            "No bridge snapshot found. "
            "Creating a fixed snapshot..."
        )

        snapshot = download_fixed_snapshot()

        SNAPSHOT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        snapshot.to_csv(
            SNAPSHOT_PATH,
            index=False,
            date_format="%Y-%m-%d",
        )

        print(
            f"Saved fixed snapshot:\n"
            f"{SNAPSHOT_PATH}"
        )

    validate_snapshot(snapshot)

    return snapshot.sort_values(
        [
            "asset",
            "date",
        ]
    ).reset_index(
        drop=True
    )


def select_best_configuration(
    valid_df: pd.DataFrame,
    valid_probability: pd.Series,
) -> tuple[float, str, float]:
    """Select threshold and regime filter using validation Calmar."""
    best_score = -np.inf
    best_threshold = None
    best_filter = None

    for threshold in SIGNAL_THRESHOLDS:
        for regime_filter in REGIME_FILTERS:
            valid_signal = build_signal(
                valid_df,
                valid_probability,
                threshold,
                regime_filter,
            )

            (
                valid_returns,
                _,
                _,
                _,
            ) = backtest_signal(
                valid_df,
                valid_signal,
            )

            valid_metrics = calculate_metrics(
                valid_returns
            )

            score = valid_metrics["calmar"]

            if pd.isna(score):
                score = -np.inf

            if score > best_score:
                best_score = float(score)
                best_threshold = float(threshold)
                best_filter = regime_filter

    if best_threshold is None or best_filter is None:
        raise RuntimeError(
            "No valid signal configuration was selected."
        )

    return (
        best_threshold,
        best_filter,
        best_score,
    )


def build_asset_bridge(
    snapshot: pd.DataFrame,
    asset_name: str,
    ticker: str,
) -> pd.DataFrame:
    """Create daily signal and execution-bridge rows for one asset."""
    asset_snapshot = (
        snapshot.loc[
            snapshot["asset"] == asset_name
        ]
        .sort_values("date")
        .copy()
    )

    price_df = (
        asset_snapshot
        .set_index("date")[["close"]]
    )

    dataframe = create_features(
        price_df
    )

    train_df = dataframe.loc[
        :TRAIN_END
    ].copy()

    valid_df = dataframe.loc[
        VALID_START:VALID_END
    ].copy()

    test_df = dataframe.loc[
        TEST_START:BRIDGE_TEST_END
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"Training data is empty for {asset_name}."
        )

    if valid_df.empty:
        raise ValueError(
            f"Validation data is empty for {asset_name}."
        )

    if test_df.empty:
        raise ValueError(
            f"Test data is empty for {asset_name}."
        )

    model = build_model()

    model.fit(
        train_df[FEATURE_COLS],
        train_df["target"],
    )

    valid_probability = pd.Series(
        model.predict_proba(
            valid_df[FEATURE_COLS]
        )[:, 1],
        index=valid_df.index,
    )

    test_probability = pd.Series(
        model.predict_proba(
            test_df[FEATURE_COLS]
        )[:, 1],
        index=test_df.index,
    )

    (
        selected_threshold,
        selected_filter,
        validation_calmar,
    ) = select_best_configuration(
        valid_df,
        valid_probability,
    )

    test_signal = build_signal(
        test_df,
        test_probability,
        selected_threshold,
        selected_filter,
    ).astype(float)

    target_position = test_signal
    previous_position = (
        target_position.shift(1).fillna(0.0)
    )

    position_change = (
        target_position
        - previous_position
    )

    position_change_abs = (
        position_change.abs()
    )

    gross_strategy_return = (
        previous_position
        * test_df["return_1d"]
    )

    fixed_cost_return = (
        position_change_abs
        * TOTAL_COST_PER_POSITION_CHANGE
    )

    fixed_cost_net_return = (
        gross_strategy_return
        - fixed_cost_return
    )

    side = np.select(
        [
            position_change > 0,
            position_change < 0,
        ],
        [
            "BUY",
            "SELL",
        ],
        default="NONE",
    )

    raw_price_dates = price_df.index.sort_values()

    next_date_map = pd.Series(
        raw_price_dates[1:],
        index=raw_price_dates[:-1],
    )

    effective_return_date = (
        test_df.index.to_series()
        .map(next_date_map)
    )

    bridge = pd.DataFrame(
        {
            "signal_date": test_df.index,
            "effective_return_date": (
                effective_return_date.values
            ),
            "asset": asset_name,
            "ticker": ticker,
            "portfolio_sleeve_weight": (
                1.0 / len(ASSETS)
            ),
            "decision_price": (
                test_df["close"].values
            ),
            "predicted_probability": (
                test_probability.values
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "selected_filter": (
                selected_filter
            ),
            "validation_calmar": (
                validation_calmar
            ),
            "signal": (
                test_signal.astype(int).values
            ),
            "previous_position": (
                previous_position.values
            ),
            "target_position": (
                target_position.values
            ),
            "position_change": (
                position_change.values
            ),
            "position_change_abs": (
                position_change_abs.values
            ),
            "side": side,
            "daily_asset_return": (
                test_df["return_1d"].values
            ),
            "gross_strategy_return": (
                gross_strategy_return.values
            ),
            "fixed_cost_bps": (
                TOTAL_COST_PER_POSITION_CHANGE
                * 10_000
            ),
            "fixed_cost_return": (
                fixed_cost_return.values
            ),
            "fixed_cost_net_return": (
                fixed_cost_net_return.values
            ),
            "data_as_of_date": (
                asset_snapshot["date"]
                .max()
                .strftime("%Y-%m-%d")
            ),
            "data_snapshot": (
                SNAPSHOT_PATH.name
            ),
        }
    )

    (
        reference_net_return,
        reference_turnover,
        reference_cost_drag,
        _,
    ) = backtest_signal(
        test_df,
        test_signal,
    )

    if not np.allclose(
        bridge["fixed_cost_net_return"],
        reference_net_return.values,
        equal_nan=True,
    ):
        raise AssertionError(
            f"Net-return validation failed for {asset_name}."
        )

    if not np.isclose(
        bridge["position_change_abs"].sum(),
        reference_turnover,
    ):
        raise AssertionError(
            f"Turnover validation failed for {asset_name}."
        )

    if not np.isclose(
        bridge["fixed_cost_return"].sum(),
        reference_cost_drag,
    ):
        raise AssertionError(
            f"Cost-drag validation failed for {asset_name}."
        )

    return bridge


def print_summary(
    daily_bridge: pd.DataFrame,
) -> None:
    """Print a concise bridge summary by asset."""
    print(
        "\n=============================="
    )
    print(
        "Execution Bridge Summary"
    )
    print(
        "=============================="
    )

    for asset_name in ASSETS:
        asset_rows = daily_bridge.loc[
            daily_bridge["asset"] == asset_name
        ]

        trade_rows = asset_rows.loc[
            asset_rows["position_change_abs"] > 0
        ]

        buys = int(
            (trade_rows["side"] == "BUY").sum()
        )

        sells = int(
            (trade_rows["side"] == "SELL").sum()
        )

        turnover = float(
            trade_rows["position_change_abs"].sum()
        )

        threshold = float(
            asset_rows["selected_threshold"].iloc[0]
        )

        regime_filter = (
            asset_rows["selected_filter"].iloc[0]
        )

        print(
            f"{asset_name}: "
            f"daily rows={len(asset_rows)}, "
            f"orders={len(trade_rows)}, "
            f"buys={buys}, "
            f"sells={sells}, "
            f"turnover={turnover:.0f}, "
            f"threshold={threshold:.2f}, "
            f"filter={regime_filter}"
        )


def main() -> None:
    """Create the fixed snapshot and execution bridge CSV files."""
    snapshot = load_or_create_snapshot()

    bridge_frames = []

    for asset_name, ticker in ASSETS.items():
        print(
            f"\nBuilding bridge for "
            f"{asset_name} ({ticker})..."
        )

        bridge_frames.append(
            build_asset_bridge(
                snapshot,
                asset_name,
                ticker,
            )
        )

    daily_bridge = pd.concat(
        bridge_frames,
        ignore_index=True,
    )

    daily_bridge = daily_bridge.sort_values(
        [
            "signal_date",
            "asset",
        ]
    ).reset_index(
        drop=True
    )

    execution_orders = daily_bridge.loc[
        daily_bridge["position_change_abs"] > 0
    ].copy()

    DAILY_BRIDGE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_bridge.to_csv(
        DAILY_BRIDGE_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    execution_orders.to_csv(
        ORDER_BRIDGE_PATH,
        index=False,
        date_format="%Y-%m-%d",
    )

    print_summary(
        daily_bridge
    )

    print(
        "\nSaved files:"
    )
    print(
        f"- {SNAPSHOT_PATH}"
    )
    print(
        f"- {DAILY_BRIDGE_PATH}"
    )
    print(
        f"- {ORDER_BRIDGE_PATH}"
    )

    print(
        "\nThe order CSV contains only "
        "BUY and SELL position changes."
    )


if __name__ == "__main__":
    main()
