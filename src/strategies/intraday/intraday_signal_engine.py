"""
Intraday Signal Engine Module (Korean Stocks)
=============================================

단타 신호 오케스트레이터.
두 가지 모드를 지원한다:
    모드 1 - 일봉 OHLC 시뮬레이션 (10년 백테스트용)
    모드 2 - 15분봉 (최근 구간 정밀 백테스트용)
"""

import logging
import os
import sys
from typing import Dict, Optional, List

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.strategies.intraday.aziz_strategies import (
    AzizStrategyOrchestrator, add_vwap, MomentumScanner
)

logger = logging.getLogger(__name__)


class IntradaySignalEngine:
    """
    단타 신호 오케스트레이터.

    모드 1 (daily_sim): 일봉 기반 갭/모멘텀 시뮬레이션
        - 10년 역사 데이터 백테스트 가능
        - 당일 시가 매수 → 종가 청산 (단순 모델)

    모드 2 (intraday_15m): Andrew Aziz 전략 기반
        - 15분봉 데이터 필요 (최근 구간만 가능)
        - ORB, GapAndGo, BullFlag, ABCD, VWAPReversal
    """

    MODES = ('daily_sim', 'intraday_15m')

    def __init__(
        self,
        mode: str = 'daily_sim',
        min_gap_pct: float = 0.02,
        min_vol_ratio: float = 1.5,
        max_positions: int = 5,
        position_size: float = 0.15,   # 포지션당 최대 비중
        transaction_cost: float = 0.00025,  # 왕복 0.025%
        daily_stop_loss: float = -0.02,    # 일중 손절: -2% (시가 기준)
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.min_gap_pct = min_gap_pct
        self.min_vol_ratio = min_vol_ratio
        self.max_positions = max_positions
        self.position_size = position_size
        self.transaction_cost = transaction_cost
        self.daily_stop_loss = daily_stop_loss
        self._orchestrator = AzizStrategyOrchestrator() if mode == 'intraday_15m' else None
        self._scanner = MomentumScanner()

    # ------------------------------------------------------------------
    # 공통 인터페이스
    # ------------------------------------------------------------------

    def generate_weights(
        self,
        daily_data: Dict[str, pd.DataFrame],
        intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
        target_date: Optional[pd.Timestamp] = None,
        portfolio_value: float = 1_000_000,
    ) -> Dict[str, float]:
        """
        당일 포트폴리오 비중 생성.
        모든 포지션은 당일 청산을 원칙으로 한다.

        Args:
            daily_data: {ticker: daily_df}
            intraday_data: {ticker: 15min_df} (mode='intraday_15m'일 때 필요)
            target_date: 신호 생성 기준일
            portfolio_value: 총 포트폴리오 가치

        Returns:
            {ticker: weight}
        """
        if self.mode == 'daily_sim':
            return self._daily_sim_weights(daily_data)
        elif self.mode == 'intraday_15m':
            if intraday_data is None:
                logger.warning("intraday_15m mode requires intraday_data; falling back to daily_sim")
                return self._daily_sim_weights(daily_data)
            return self._intraday_15m_weights(daily_data, intraday_data, portfolio_value)
        return {}

    # ------------------------------------------------------------------
    # 모드 1: 일봉 시뮬레이션
    # ------------------------------------------------------------------

    def _daily_sim_weights(self, daily_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        일봉 OHLC 기반 당일 매매 시뮬레이션.
        갭 + 거래량 필터로 종목 선정 후 균등 비중 배분.
        """
        candidates = self._scanner.scan(
            daily_data,
            min_gap_pct=self.min_gap_pct,
            min_vol_ratio=self.min_vol_ratio,
            top_n=self.max_positions,
        )
        if not candidates:
            return {}
        weight = min(self.position_size, 1.0 / len(candidates))
        return {t: weight for t in candidates}

    # ------------------------------------------------------------------
    # 모드 2: 15분봉 정밀 전략
    # ------------------------------------------------------------------

    def _intraday_15m_weights(
        self,
        daily_data: Dict[str, pd.DataFrame],
        intraday_data: Dict[str, pd.DataFrame],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Andrew Aziz 전략 기반 신호 생성 및 비중 변환."""
        signals = self._orchestrator.generate_signals(
            intraday_data, daily_data, portfolio_value
        )
        return self._orchestrator.signals_to_weights(signals, portfolio_value)

    # ------------------------------------------------------------------
    # 성과 시뮬레이션 (일봉 기반)
    # ------------------------------------------------------------------

    def simulate_daily(
        self,
        daily_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        initial_capital: float = 10_000_000,
    ) -> pd.DataFrame:
        """
        일봉 기반 단타 전략 백테스트 시뮬레이션.
        당일 시가 매수 → 종가 청산 가정.

        Returns:
            DataFrame: 날짜별 포트폴리오 가치 및 수익률
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # 공통 거래일 목록 생성
        all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        for df in daily_data.values():
            dates = pd.to_datetime(df['date'] if 'date' in df.columns else df.index)
            all_dates = all_dates.union(dates)
        all_dates = all_dates[(all_dates >= start) & (all_dates <= end)].sort_values()

        portfolio_value = initial_capital
        records = []

        for i, date in enumerate(all_dates):
            # 해당 날짜까지의 데이터로 슬라이스
            data_slice: Dict[str, pd.DataFrame] = {}
            for ticker, df in daily_data.items():
                date_col = 'date' if 'date' in df.columns else df.index.name or 'index'
                if date_col in df.columns:
                    sub = df[pd.to_datetime(df[date_col]) <= date]
                else:
                    sub = df[df.index <= date]
                if len(sub) >= 2:
                    data_slice[ticker] = sub

            weights = self._daily_sim_weights(data_slice)
            daily_return = 0.0

            for ticker, weight in weights.items():
                df_t = data_slice.get(ticker)
                if df_t is None or len(df_t) < 1:
                    continue
                today_row = df_t.iloc[-1]
                day_open = float(today_row.get('prcod', today_row.get('open', 0)))
                day_close = float(today_row.get('prccd', today_row.get('close', 0)))
                day_low  = float(today_row.get('prcld', today_row.get('low', day_open)))
                if day_open > 0 and day_close > 0:
                    # 일중 손절 시뮬레이션: 저가가 stop_loss 이하면 stop_loss 가격에 청산
                    open_to_low = (day_low - day_open) / day_open if day_open > 0 else 0
                    if open_to_low <= self.daily_stop_loss:
                        gross_return = self.daily_stop_loss  # stop hit
                    else:
                        gross_return = (day_close - day_open) / day_open
                    net_return = gross_return - self.transaction_cost * 2
                    daily_return += weight * net_return

            portfolio_value *= (1 + daily_return)
            records.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'n_positions': len(weights),
            })

        result_df = pd.DataFrame(records).set_index('date')
        result_df['cumulative_return'] = (result_df['portfolio_value'] / initial_capital) - 1.0
        return result_df
