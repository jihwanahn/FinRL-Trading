"""
Andrew Aziz Intraday Strategies Module (Korean Market)
=======================================================

Andrew Aziz "How to Day Trade for a Living" /
"Advanced Techniques in Day Trading" 전략 구현.

각 전략은 15분봉 OHLCV + VWAP + 거래량 데이터를 사용한다.
한국 시장 특성:
    - 롱 온리 (공매도 제한)
    - 동시호가 09:00 반영
    - 상한가/하한가 ±30%
    - 증권거래세 0.20% (매도)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """당일 VWAP 계산."""
    cum_vol = df['volume'].cumsum()
    cum_tp_vol = (df['typical_price'] * df['volume']).cumsum()
    return (cum_tp_vol / cum_vol.replace(0, np.nan)).fillna(method='ffill')


def add_typical_price(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
    return df


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = add_typical_price(df)
    df['vwap'] = compute_vwap(df)
    return df


@dataclass
class TradeSignal:
    """단일 매매 신호."""
    strategy: str
    ticker: str
    side: str              # 'buy' | 'sell'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    target_price: float
    risk: float            # entry - stop_loss
    reward: float          # target - entry
    signal_strength: float = 1.0  # 0~1

    @property
    def rr_ratio(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0.0


# ---------------------------------------------------------------------------
# Strategy 1: Gap and Go
# ---------------------------------------------------------------------------

class GapAndGo:
    """
    갭 앤 고 전략.
    - 전날 종가 대비 +2% 이상 갭업 + 장전 거래량 존재
    - 시가 직후 첫 15분봉 고점 돌파 시 매수
    - 손절: 당일 시가 또는 VWAP 이탈
    - 목표: Risk:Reward = 1:2
    """
    MIN_GAP_PCT = 0.02       # 최소 갭 %
    MIN_RR_RATIO = 1.5       # 최소 리스크/리워드

    def generate_signal(self, df_intraday: pd.DataFrame,
                         prev_close: float,
                         ticker: str) -> Optional[TradeSignal]:
        """
        Args:
            df_intraday: 당일 15분봉 DataFrame (vwap 컬럼 포함)
            prev_close: 전일 종가
            ticker: 종목 코드
        """
        if df_intraday.empty or prev_close <= 0:
            return None
        if 'vwap' not in df_intraday.columns:
            df_intraday = add_vwap(df_intraday)

        day_open = float(df_intraday.iloc[0]['open'])
        gap_pct = (day_open - prev_close) / prev_close

        if gap_pct < self.MIN_GAP_PCT:
            return None  # 갭 부족

        # 첫 15분봉
        first_bar = df_intraday.iloc[0]
        breakout_level = float(first_bar['high'])
        stop_price = min(float(first_bar['low']), float(first_bar.get('vwap', day_open)))
        risk = breakout_level - stop_price
        if risk <= 0:
            return None
        target = breakout_level + risk * 2.0  # 1:2 R:R

        # 돌파 확인 (2번째 봉 이후)
        for _, bar in df_intraday.iloc[1:].iterrows():
            if float(bar['high']) >= breakout_level and float(bar['volume']) > 0:
                return TradeSignal(
                    strategy='GapAndGo',
                    ticker=ticker,
                    side='buy',
                    entry_time=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(bar.name),
                    entry_price=breakout_level,
                    stop_loss=stop_price,
                    target_price=target,
                    risk=risk,
                    reward=target - breakout_level,
                    signal_strength=min(1.0, gap_pct / 0.05),
                )
        return None


# ---------------------------------------------------------------------------
# Strategy 2: Bull Flag Momentum
# ---------------------------------------------------------------------------

class BullFlagMomentum:
    """
    불 플래그 모멘텀 전략.
    - 강한 상승 후 좁은 횡보(깃대+깃발) 패턴 감지
    - 깃발 상단 돌파 + 거래량 급증 시 매수
    """
    MIN_POLE_GAIN = 0.03    # 깃대 최소 상승률 3%
    MAX_FLAG_BARS = 5       # 깃발 최대 봉 수
    MAX_FLAG_RANGE_PCT = 0.02  # 깃발 최대 가격 범위 2%

    def generate_signal(self, df_intraday: pd.DataFrame,
                         ticker: str) -> Optional[TradeSignal]:
        if len(df_intraday) < self.MAX_FLAG_BARS + 3:
            return None
        if 'vwap' not in df_intraday.columns:
            df_intraday = add_vwap(df_intraday)

        closes = df_intraday['close'].astype(float).values
        highs = df_intraday['high'].astype(float).values
        lows = df_intraday['low'].astype(float).values
        volumes = df_intraday['volume'].astype(float).values

        for pole_end in range(2, len(df_intraday) - self.MAX_FLAG_BARS - 1):
            pole_low = lows[:pole_end].min()
            pole_high = highs[pole_end]
            pole_gain = (pole_high - pole_low) / pole_low if pole_low > 0 else 0
            if pole_gain < self.MIN_POLE_GAIN:
                continue

            # 깃발 구간 확인
            flag_bars = df_intraday.iloc[pole_end: pole_end + self.MAX_FLAG_BARS]
            flag_high = flag_bars['high'].max()
            flag_low = flag_bars['low'].min()
            flag_range = (flag_high - flag_low) / flag_high if flag_high > 0 else 1.0
            if flag_range > self.MAX_FLAG_RANGE_PCT:
                continue  # 깃발 범위 초과

            # 돌파 시그널
            breakout_level = float(flag_high)
            next_idx = pole_end + self.MAX_FLAG_BARS
            if next_idx >= len(df_intraday):
                continue
            next_bar = df_intraday.iloc[next_idx]
            avg_vol = volumes[:pole_end].mean()
            if float(next_bar['high']) > breakout_level and float(next_bar['volume']) > avg_vol * 1.5:
                risk = breakout_level - float(flag_low)
                if risk <= 0:
                    continue
                return TradeSignal(
                    strategy='BullFlag',
                    ticker=ticker,
                    side='buy',
                    entry_time=next_bar.name if isinstance(next_bar.name, pd.Timestamp) else pd.Timestamp(str(next_bar.name)),
                    entry_price=breakout_level,
                    stop_loss=float(flag_low),
                    target_price=breakout_level + risk * 2.0,
                    risk=risk,
                    reward=risk * 2.0,
                    signal_strength=min(1.0, pole_gain / 0.10),
                )
        return None


# ---------------------------------------------------------------------------
# Strategy 3: ABCD Pattern
# ---------------------------------------------------------------------------

class ABCDPattern:
    """
    ABCD 패턴 전략.
    A: 장초반 고점 → B: 조정 → C: B에서 반등 → D: A 돌파
    피보나치 되돌림 0.618~0.786로 B 포인트 검증
    """
    FIB_MIN = 0.618
    FIB_MAX = 0.786

    def generate_signal(self, df_intraday: pd.DataFrame,
                         ticker: str) -> Optional[TradeSignal]:
        if len(df_intraday) < 8:
            return None
        if 'vwap' not in df_intraday.columns:
            df_intraday = add_vwap(df_intraday)

        highs = df_intraday['high'].astype(float).values
        lows = df_intraday['low'].astype(float).values

        # A: 장초반(처음 4봉 내) 고점
        a_idx = int(np.argmax(highs[:4]))
        a_price = highs[a_idx]
        if a_idx == 0:
            base_low = lows[0]
        else:
            base_low = lows[:a_idx].min()

        # B: A 이후 조정 저점 (피보나치 검증)
        search_range = df_intraday.iloc[a_idx + 1: a_idx + 6]
        if len(search_range) < 2:
            return None
        b_idx_rel = int(search_range['low'].astype(float).values.argmin())
        b_price = float(search_range['low'].iloc[b_idx_rel])
        b_idx_abs = a_idx + 1 + b_idx_rel

        retracement = (a_price - b_price) / (a_price - base_low) if (a_price - base_low) > 0 else 0
        if not (self.FIB_MIN <= retracement <= self.FIB_MAX):
            return None

        # C: B에서 반등 (A 미달)
        c_range = df_intraday.iloc[b_idx_abs + 1: b_idx_abs + 4]
        if c_range.empty:
            return None
        c_price = float(c_range['high'].max())
        if c_price >= a_price:
            return None  # A 돌파 → ABCD 무효

        # D: C 이후 A 재돌파
        d_range = df_intraday.iloc[b_idx_abs + 4:]
        for _, bar in d_range.iterrows():
            if float(bar['high']) > a_price:
                risk = a_price - b_price
                if risk <= 0:
                    return None
                return TradeSignal(
                    strategy='ABCD',
                    ticker=ticker,
                    side='buy',
                    entry_time=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(str(bar.name)),
                    entry_price=a_price,
                    stop_loss=b_price,
                    target_price=a_price + risk,
                    risk=risk,
                    reward=risk,
                    signal_strength=0.7,
                )
        return None


# ---------------------------------------------------------------------------
# Strategy 4: Opening Range Breakout (ORB)
# ---------------------------------------------------------------------------

class OpeningRangeBreakout:
    """
    오프닝 레인지 브레이크아웃 (ORB).
    한국 장: 09:00~09:15 (15분) 고/저 범위 설정 후 돌파 매수.
    """
    ORB_BARS = 1   # 첫 1개 15분봉 = 09:00~09:15

    def generate_signal(self, df_intraday: pd.DataFrame,
                         ticker: str) -> Optional[TradeSignal]:
        if len(df_intraday) <= self.ORB_BARS:
            return None
        if 'vwap' not in df_intraday.columns:
            df_intraday = add_vwap(df_intraday)

        # ORB 범위
        orb = df_intraday.iloc[:self.ORB_BARS]
        orb_high = float(orb['high'].max())
        orb_low = float(orb['low'].min())
        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return None

        avg_vol = float(orb['volume'].mean())

        # 돌파 확인
        for _, bar in df_intraday.iloc[self.ORB_BARS:].iterrows():
            if float(bar['high']) > orb_high and float(bar['volume']) > avg_vol * 1.2:
                risk = orb_high - orb_low
                return TradeSignal(
                    strategy='ORB',
                    ticker=ticker,
                    side='buy',
                    entry_time=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(str(bar.name)),
                    entry_price=orb_high,
                    stop_loss=orb_low,
                    target_price=orb_high + orb_range * 2.0,
                    risk=risk,
                    reward=orb_range * 2.0,
                    signal_strength=min(1.0, float(bar['volume']) / (avg_vol * 2)),
                )
        return None


# ---------------------------------------------------------------------------
# Strategy 5: VWAP Reversal (Fallen Angel)
# ---------------------------------------------------------------------------

class VWAPReversal:
    """
    VWAP 리버설 전략.
    갭다운 후 VWAP 회복 시도 포착.
    """
    MIN_GAPDOWN_PCT = 0.03  # 최소 -3% 갭다운

    def generate_signal(self, df_intraday: pd.DataFrame,
                         prev_close: float,
                         ticker: str) -> Optional[TradeSignal]:
        if df_intraday.empty or prev_close <= 0:
            return None
        if 'vwap' not in df_intraday.columns:
            df_intraday = add_vwap(df_intraday)

        day_open = float(df_intraday.iloc[0]['open'])
        gap_pct = (day_open - prev_close) / prev_close
        if gap_pct > -self.MIN_GAPDOWN_PCT:
            return None  # 갭다운 부족

        day_low = float(df_intraday['low'].min())

        # VWAP 회복 시도
        for i, (_, bar) in enumerate(df_intraday.iterrows()):
            if i == 0:
                continue
            vwap = float(bar.get('vwap', day_open))
            if float(bar['close']) > vwap and float(df_intraday.iloc[i - 1]['close']) <= vwap:
                risk = vwap - day_low
                if risk <= 0:
                    continue
                return TradeSignal(
                    strategy='VWAPReversal',
                    ticker=ticker,
                    side='buy',
                    entry_time=bar.name if isinstance(bar.name, pd.Timestamp) else pd.Timestamp(str(bar.name)),
                    entry_price=vwap,
                    stop_loss=day_low,
                    target_price=vwap + risk * 1.5,
                    risk=risk,
                    reward=risk * 1.5,
                    signal_strength=0.6,
                )
        return None


# ---------------------------------------------------------------------------
# Momentum Scanner
# ---------------------------------------------------------------------------

class MomentumScanner:
    """
    당일 유니버스 스캐닝.
    조건: 갭 +2% 이상, 거래량 전일 2배 이상, 시총 300억 이상
    """

    def scan(self, price_data: Dict[str, pd.DataFrame],
              min_gap_pct: float = 0.02,
              min_vol_ratio: float = 2.0,
              top_n: int = 10) -> List[str]:
        """
        Args:
            price_data: {ticker: daily_df} (최소 2일치)
            min_gap_pct: 최소 갭 비율
            min_vol_ratio: 전일 대비 최소 거래량 배율
            top_n: 상위 N개 반환

        Returns:
            List of ticker codes
        """
        candidates = []
        for ticker, df in price_data.items():
            if len(df) < 5:
                continue
            # Look-ahead bias 방지: today_vol 사용 금지 → prev 5일 평균 거래량 사용
            prev = df.iloc[-2]
            today = df.iloc[-1]
            prev_close = float(prev.get('close', prev.get('prccd', 0)))
            today_open = float(today.get('open', today.get('prcod', 0)))
            # 5일 평균 거래량 (전일까지)
            prev_5d = df.iloc[-6:-1]
            avg_vol_5d = prev_5d.apply(
                lambda r: float(r.get('volume', r.get('cshtrd', 0))), axis=1
            ).mean()
            prev_vol = avg_vol_5d if avg_vol_5d > 0 else 1.0

            if prev_close <= 0 or today_open <= 0:
                continue
            gap = (today_open - prev_close) / prev_close

            # 전일 종가 대비 갭 + 전일 이전 추세 필터 (3일 연속 양봉)
            prev3_closes = [
                float(df.iloc[-i].get('close', df.iloc[-i].get('prccd', 0)))
                for i in range(2, 5)
            ]
            prev_opens = [
                float(df.iloc[-i].get('open', df.iloc[-i].get('prcod', 0)))
                for i in range(2, 5)
            ]
            # 최소 2/3 봉이 양봉이어야 추세 확인 (없으면 패스)
            bullish_days = sum(c > o for c, o in zip(prev3_closes, prev_opens))

            if gap >= min_gap_pct and bullish_days >= 2:
                candidates.append((ticker, gap, bullish_days))

        # 갭 크기 기준 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _, _ in candidates[:top_n]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class AzizStrategyOrchestrator:
    """
    Andrew Aziz 전략 통합 오케스트레이터.
    5개 전략의 신호를 우선순위 기반으로 결합한다.
    """
    PRIORITY = ['ORB', 'GapAndGo', 'BullFlag', 'ABCD', 'VWAPReversal']
    MAX_POSITIONS = 3
    POSITION_SIZE = 0.20  # 포지션당 총자산의 20%
    RISK_PER_TRADE = 0.01  # 총자산의 1% 손절

    def __init__(self):
        self.orb = OpeningRangeBreakout()
        self.gap_and_go = GapAndGo()
        self.bull_flag = BullFlagMomentum()
        self.abcd = ABCDPattern()
        self.vwap_reversal = VWAPReversal()
        self.scanner = MomentumScanner()

    def generate_signals(
        self,
        intraday_data: Dict[str, pd.DataFrame],
        daily_data: Dict[str, pd.DataFrame],
        portfolio_value: float = 1_000_000,
    ) -> List[TradeSignal]:
        """
        당일 매매 신호 목록 생성.

        Args:
            intraday_data: {ticker: 15분봉 DataFrame (vwap 포함)}
            daily_data: {ticker: 일봉 DataFrame (최소 2일치)}
            portfolio_value: 총 포트폴리오 가치

        Returns:
            List[TradeSignal] (우선순위 정렬)
        """
        # 1. 유니버스 스캐닝
        candidates = self.scanner.scan(daily_data)
        logger.info(f"Momentum scanner found {len(candidates)} candidates: {candidates}")

        signals: List[TradeSignal] = []
        for ticker in candidates:
            if ticker not in intraday_data:
                continue
            df = add_vwap(intraday_data[ticker])
            prev_closes = daily_data.get(ticker)
            prev_close = float(prev_closes.iloc[-2].get('close', prev_closes.iloc[-2].get('prccd', 0))) if prev_closes is not None and len(prev_closes) >= 2 else 0.0

            # 각 전략 신호 시도 (우선순위 순)
            sig = (
                self.orb.generate_signal(df, ticker)
                or self.gap_and_go.generate_signal(df, prev_close, ticker)
                or self.bull_flag.generate_signal(df, ticker)
                or self.abcd.generate_signal(df, ticker)
                or self.vwap_reversal.generate_signal(df, prev_close, ticker)
            )
            if sig is not None:
                signals.append(sig)

        # 최대 포지션 수 제한 (신호 강도 기준 정렬)
        signals.sort(key=lambda s: s.signal_strength, reverse=True)
        return signals[:self.MAX_POSITIONS]

    def signals_to_weights(self, signals: List[TradeSignal],
                            portfolio_value: float) -> Dict[str, float]:
        """
        신호를 포트폴리오 비중으로 변환.
        리스크 기반 포지션 사이징:
            수량 = (portfolio_value * RISK_PER_TRADE) / risk_per_share
        """
        weights: Dict[str, float] = {}
        for sig in signals:
            if sig.risk <= 0 or sig.entry_price <= 0:
                continue
            risk_amount = portfolio_value * self.RISK_PER_TRADE
            qty = risk_amount / sig.risk
            position_value = qty * sig.entry_price
            weight = min(position_value / portfolio_value, self.POSITION_SIZE)
            weights[sig.ticker] = weight
        return weights
