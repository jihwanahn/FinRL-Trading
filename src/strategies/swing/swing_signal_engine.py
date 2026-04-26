"""
Swing Signal Engine Module (Korean Stocks)
==========================================

기술적 지표 기반 스윙 트레이딩 신호 생성기.
일봉 OHLCV 데이터로 주 1~2회 리밸런싱 신호를 생성한다.

신호 구성:
    - MA5/MA20/MA60 크로스오버
    - RSI(14)
    - MACD (12-26-9)
    - 볼린저 밴드 (20일, 2σ)
"""

import logging
import os
import sys
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.strategies.base_signal import BaseSignalEngine

logger = logging.getLogger(__name__)


class SwingSignalEngine(BaseSignalEngine):
    """
    스윙 트레이딩 신호 엔진.

    BaseSignalEngine을 상속하며, 주간 리밸런싱 빈도로 동작한다.
    """

    def __init__(
        self,
        ma_short: int = 5,
        ma_mid: int = 20,
        ma_long: int = 60,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rebalance_freq: str = 'W',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.ma_short = ma_short
        self.ma_mid = ma_mid
        self.ma_long = ma_long
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rebalance_freq = rebalance_freq

    def get_signal_frequency(self) -> str:
        """주간 신호 생성."""
        return "W"

    def generate_signal_one_ticker(self, df: pd.DataFrame) -> pd.Series:
        """
        단일 종목의 스윙 신호를 생성한다.

        Args:
            df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']

        Returns:
            pd.Series: 날짜 인덱스의 신호값 (-1.0 ~ +1.0)
        """
        df = df.copy().set_index('date') if 'date' in df.columns else df.copy()
        close = df['close'].astype(float)
        n = len(close)
        if n < max(self.ma_long, self.rsi_period, self.macd_slow + self.macd_signal_period, self.bb_period) + 5:
            return pd.Series(0.0, index=df.index)

        # --- MA 크로스오버 신호 ---
        ma_short = close.rolling(self.ma_short).mean()
        ma_mid = close.rolling(self.ma_mid).mean()
        ma_long = close.rolling(self.ma_long).mean()

        # 각 조건을 독립 점수로 계산 후 합산 (set 후 +=/- 으로 인한 override 방지)
        score_mid  = (ma_short > ma_mid).astype(float) * 2 - 1   # +1 or -1
        score_long = (ma_short > ma_long).astype(float) * 2 - 1  # +1 or -1
        ma_signal = (0.5 * score_mid + 0.5 * score_long).clip(-1.0, 1.0)

        # --- RSI 신호 ---
        rsi = self._compute_rsi(close, self.rsi_period)
        rsi_signal = pd.Series(0.0, index=df.index)
        rsi_signal[rsi <= self.rsi_oversold] = 1.0   # 과매도 → 매수
        rsi_signal[rsi >= self.rsi_overbought] = -1.0  # 과매수 → 매도

        # --- MACD 신호 ---
        macd_line, signal_line, _ = self._compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal_period
        )
        macd_signal_series = pd.Series(0.0, index=df.index)
        macd_signal_series[macd_line > signal_line] = 1.0
        macd_signal_series[macd_line < signal_line] = -1.0

        # --- 볼린저 밴드 신호 ---
        bb_mid = close.rolling(self.bb_period).mean()
        bb_std_series = close.rolling(self.bb_period).std()
        bb_upper = bb_mid + self.bb_std * bb_std_series
        bb_lower = bb_mid - self.bb_std * bb_std_series
        bb_signal = pd.Series(0.0, index=df.index)
        bb_signal[close <= bb_lower] = 1.0   # 하단 밴드 이탈 → 반등 기대
        bb_signal[close >= bb_upper] = -1.0  # 상단 밴드 이탈 → 되돌림 기대

        # --- 신호 합산 (가중 평균) ---
        weights = {'ma': 0.35, 'rsi': 0.25, 'macd': 0.25, 'bb': 0.15}
        combined = (
            weights['ma'] * ma_signal +
            weights['rsi'] * rsi_signal +
            weights['macd'] * macd_signal_series +
            weights['bb'] * bb_signal
        )
        # 리밸런싱 주기에 맞게 resample
        # '2W'는 backtest runner에서 서브샘플링하므로 'W-FRI'로 생성
        freq = self.rebalance_freq
        if freq in ('W', '2W'):
            rule = 'W-FRI'
        elif freq in ('ME', 'M'):
            rule = 'ME'
        elif freq in ('QS', 'Q'):
            rule = 'QS'
        else:
            rule = freq  # pass through
        resampled = combined.resample(rule).last()
        return resampled.fillna(0.0)

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산."""
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                      signal: int = 9):
        """MACD 계산. Returns (macd_line, signal_line, histogram)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def generate_weights_from_signals(self, signals: pd.DataFrame,
                                       max_positions: int = 10,
                                       min_signal: float = 0.3) -> pd.DataFrame:
        """
        신호 DataFrame을 포트폴리오 비중 DataFrame으로 변환한다.

        Args:
            signals: DataFrame (날짜 × 종목) 신호값
            max_positions: 최대 보유 종목 수
            min_signal: 최소 신호 강도 (이 이상인 종목만 포함)

        Returns:
            DataFrame: 날짜별 종목 비중
        """
        weights_list = []
        for date, row in signals.iterrows():
            positive = row[row >= min_signal].nlargest(max_positions)
            if positive.sum() > 0:
                w = positive / positive.sum()
            else:
                w = pd.Series(dtype=float)
            weights_list.append(w)

        weights_df = pd.DataFrame(weights_list, index=signals.index).fillna(0.0)
        return weights_df
