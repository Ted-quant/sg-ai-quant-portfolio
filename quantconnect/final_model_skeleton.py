from AlgorithmImports import *


class FinalExtraTreesMLSkeleton(QCAlgorithm):
    """
    QuantConnect / LEAN-style skeleton for the final ExtraTrees ML portfolio strategy.

    This file is a platform implementation prototype.

    The local Python research model selected:
    - Model: ExtraTreesClassifier
    - Feature Set: Full Features
    - Target: 5-day forward return > +1%
    - Position Sizing: Binary
    - Portfolio: Robust Assets
    - Assets: D05_SI, NVDA, SPY

    This skeleton focuses on translating the research logic into an event-driven
    algorithmic trading structure. It does not fully reproduce the trained
    sklearn model yet.
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2026, 7, 13)
        self.SetCash(100000)

        self.symbols = []

        # Note:
        # QuantConnect data availability for D05.SI should be confirmed.
        # SPY and NVDA are included as standard US-listed securities.
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.nvda = self.AddEquity("NVDA", Resolution.Daily).Symbol

        self.symbols = [self.spy, self.nvda]

        # D05.SI is kept as a design placeholder because it was part of the
        # final local robust portfolio. Actual availability may require
        # QuantConnect data mapping or an alternative Singapore-listed dataset.
        self.d05_placeholder = "D05_SI"

        self.lookback = 220
        self.price_windows = {}

        for symbol in self.symbols:
            self.price_windows[symbol] = RollingWindow[float](self.lookback)

        # Rebalance once per day after market open.
        self.Schedule.On(
            self.DateRules.EveryDay(self.spy),
            self.TimeRules.AfterMarketOpen(self.spy, 30),
            self.Rebalance
        )

        self.SetWarmUp(self.lookback, Resolution.Daily)

    def OnData(self, data):
        """
        Store daily close prices in rolling windows.
        """
        for symbol in self.symbols:
            if data.ContainsKey(symbol) and data[symbol] is not None:
                self.price_windows[symbol].Add(float(data[symbol].Close))

    def Rebalance(self):
        """
        Daily rebalance function.

        In the local Python model:
        - Features were calculated from daily price data.
        - ExtraTrees generated a probability signal.
        - Binary position sizing converted the signal into buy-or-cash.
        - Active robust assets were equal-weighted.

        In this skeleton:
        - We calculate the same style of daily features.
        - We use a simplified proxy signal to represent the ML signal.
        - Active assets are equal-weighted using SetHoldings().
        """

        if self.IsWarmingUp:
            return

        active_symbols = []

        for symbol in self.symbols:
            if not self.price_windows[symbol].IsReady:
                continue

            features = self.CalculateFeatures(symbol)

            if features is None:
                continue

            signal = self.GenerateProxySignal(features)

            if signal == 1:
                active_symbols.append(symbol)

        if len(active_symbols) == 0:
            for symbol in self.symbols:
                self.Liquidate(symbol)
            return

        target_weight = 1.0 / len(active_symbols)

        for symbol in self.symbols:
            if symbol in active_symbols:
                self.SetHoldings(symbol, target_weight)
            else:
                self.Liquidate(symbol)

    def CalculateFeatures(self, symbol):
        """
        Calculate local-model-style daily features.

        Local final model features:
        - return_5d
        - return_20d
        - ma10_ratio
        - ma20_ratio
        - ma60_ratio
        - volatility_20d
        - rsi_14
        """

        prices = list(self.price_windows[symbol])

        if len(prices) < 200:
            return None

        # RollingWindow stores newest value first.
        prices = list(reversed(prices))

        current_price = prices[-1]

        return_5d = current_price / prices[-6] - 1
        return_20d = current_price / prices[-21] - 1

        ma10 = sum(prices[-10:]) / 10
        ma20 = sum(prices[-20:]) / 20
        ma60 = sum(prices[-60:]) / 60

        ma10_ratio = current_price / ma10 - 1
        ma20_ratio = current_price / ma20 - 1
        ma60_ratio = current_price / ma60 - 1

        returns_20d = []
        for i in range(-20, 0):
            daily_return = prices[i] / prices[i - 1] - 1
            returns_20d.append(daily_return)

        mean_return = sum(returns_20d) / len(returns_20d)
        variance = sum((r - mean_return) ** 2 for r in returns_20d) / len(returns_20d)
        volatility_20d = variance ** 0.5

        rsi_14 = self.CalculateRSI(prices, 14)

        return {
            "return_5d": return_5d,
            "return_20d": return_20d,
            "ma10_ratio": ma10_ratio,
            "ma20_ratio": ma20_ratio,
            "ma60_ratio": ma60_ratio,
            "volatility_20d": volatility_20d,
            "rsi_14": rsi_14,
        }

    def CalculateRSI(self, prices, period):
        """
        Calculate a simple RSI value from close prices.
        """

        gains = []
        losses = []

        recent_prices = prices[-(period + 1):]

        for i in range(1, len(recent_prices)):
            change = recent_prices[i] - recent_prices[i - 1]

            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def GenerateProxySignal(self, features):
        """
        Simplified proxy for the final local ML signal.

        The final local ExtraTrees model found that the most important features were:
        - ma60_ratio
        - volatility_20d
        - rsi_14

        This proxy signal uses the same financial intuition:
        - Favor positive medium-term trend
        - Avoid extremely overbought conditions
        - Avoid unusually unstable price action

        Future extension:
        Replace this proxy with a serialized sklearn model or model retraining
        inside the LEAN algorithm.
        """

        trend_ok = features["ma60_ratio"] > 0
        rsi_ok = 35 <= features["rsi_14"] <= 75
        volatility_ok = features["volatility_20d"] < 0.05

        if trend_ok and rsi_ok and volatility_ok:
            return 1

        return 0
