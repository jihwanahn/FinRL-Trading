"""
Intraday Backtester Module (Korean Stocks)
==========================================

일중(intraday) 전략 전용 백테스터.
bt 라이브러리 확장 및 커스텀 시뮬레이션을 결합한다.

사용법:
    bt_engine = IntradayBacktest(config)
    result = bt_engine.run(daily_data, intraday_data)
    bt_engine.print_summary(result)
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.strategies.intraday.intraday_signal_engine import IntradaySignalEngine

logger = logging.getLogger(__name__)


@dataclass
class IntradayBacktestConfig:
    """단타 백테스트 설정."""
    start_date: str
    end_date: str
    mode: str = 'daily_sim'           # 'daily_sim' | 'intraday_15m'
    initial_capital: float = 10_000_000.0
    transaction_cost: float = 0.00025  # 왕복 0.025% (수수료 0.015% × 2 + 증권거래세)
    max_positions: int = 5
    position_size: float = 0.15
    min_gap_pct: float = 0.02
    min_vol_ratio: float = 1.5
    benchmark_ticker: str = "069500"  # KODEX 200


@dataclass
class IntradayBacktestResult:
    """단타 백테스트 결과."""
    strategy_name: str
    portfolio_values: pd.Series
    daily_returns: pd.Series
    trades_df: pd.DataFrame
    metrics: Dict[str, float]
    config: IntradayBacktestConfig

    def to_summary(self) -> str:
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Period: {self.config.start_date} ~ {self.config.end_date}",
            f"Mode: {self.config.mode}",
            "=" * 50,
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:30s}: {v:.4f}")
            else:
                lines.append(f"  {k:30s}: {v}")
        return "\n".join(lines)


class IntradayBacktest:
    """단타 전략 백테스터."""

    def __init__(self, config: IntradayBacktestConfig):
        self.config = config
        self.engine = IntradaySignalEngine(
            mode=config.mode,
            min_gap_pct=config.min_gap_pct,
            min_vol_ratio=config.min_vol_ratio,
            max_positions=config.max_positions,
            position_size=config.position_size,
            transaction_cost=config.transaction_cost,
        )

    def run(
        self,
        daily_data: Dict[str, pd.DataFrame],
        intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
        strategy_name: str = "KR-Intraday",
    ) -> IntradayBacktestResult:
        """백테스트 실행."""
        logger.info(f"Running intraday backtest [{self.config.mode}] "
                    f"{self.config.start_date} ~ {self.config.end_date}")

        sim_result = self.engine.simulate_daily(
            daily_data=daily_data,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            initial_capital=self.config.initial_capital,
        )

        portfolio_values = sim_result['portfolio_value']
        daily_returns = sim_result['daily_return']
        metrics = self._compute_metrics(daily_returns, portfolio_values)

        return IntradayBacktestResult(
            strategy_name=strategy_name,
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
            trades_df=sim_result,
            metrics=metrics,
            config=self.config,
        )

    def _compute_metrics(self, returns: pd.Series,
                          values: pd.Series) -> Dict[str, float]:
        """성과 지표 계산."""
        n = len(returns)
        if n == 0:
            return {}
        ann_factor = 252  # 한국 주식시장 연간 거래일

        total_return = float(values.iloc[-1] / values.iloc[0] - 1.0) if values.iloc[0] > 0 else 0.0
        years = n / ann_factor
        cagr = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0

        vol = float(returns.std() * np.sqrt(ann_factor))
        sharpe = float(cagr / vol) if vol > 0 else 0.0

        # MDD
        cum_max = values.cummax()
        drawdowns = (values - cum_max) / cum_max
        mdd = float(drawdowns.min())

        # 승률
        positive_days = int((returns > 0).sum())
        win_rate = positive_days / n if n > 0 else 0.0

        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd,
            'win_rate': win_rate,
            'total_trading_days': n,
            'positive_days': positive_days,
        }

    def print_summary(self, result: IntradayBacktestResult) -> None:
        print(result.to_summary())


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Intraday Backtester')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--mode', default='daily_sim', choices=['daily_sim', 'intraday_15m'])
    parser.add_argument('--capital', type=float, default=10_000_000)
    args = parser.parse_args()

    config = IntradayBacktestConfig(
        start_date=args.start,
        end_date=args.end,
        mode=args.mode,
        initial_capital=args.capital,
    )
    print(f"Intraday backtest config: {config}")
    print("Load daily_data and call IntradayBacktest(config).run(daily_data) to execute.")
