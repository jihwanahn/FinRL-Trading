"""
Intraday Backtest Runner (Korean Market - OHLC Simulation)
===========================================================

Module C: 단타 전략 백테스트 (일봉 OHLC 시뮬레이션)

역사 15분봉 데이터 한계로 일봉 OHLC 기반 시뮬레이션:
- 장 시작 시 갭 방향 추종 진입
- 목표가/손절가 도달 시 청산 (시뮬레이션)
- 미청산 포지션은 종가 전 전량 청산

사용법:
    python src/backtest/run_intraday_backtest.py \
        --start 2015-01-01 --end 2024-12-31 \
        --strategy gap_and_go --max-positions 3

출력:
    data/results/intraday_backtest_YYYYMMDD.csv
"""

import logging
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data.data_store import get_data_store

logger = logging.getLogger(__name__)


def load_ohlcv_long(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """DataStore에서 OHLCV 데이터를 long 형태로 반환."""
    ds = get_data_store()
    df = ds.get_price_data(tickers, start_date, end_date)
    if df.empty:
        raise RuntimeError("가격 데이터가 없습니다.")

    tic_col = next((c for c in ['tic', 'ticker', 'gvkey'] if c in df.columns), None)
    date_col = next((c for c in ['datadate', 'date'] if c in df.columns), None)

    col_map = {
        tic_col: 'ticker',
        date_col: 'date',
        'prcod': 'open',
        'prchd': 'high',
        'prcld': 'low',
        'prccd': 'close',
        'cshtrd': 'volume',
        'adj_close': 'adj_close',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    return df


class IntradayOHLCSimulator:
    """
    일봉 OHLC 기반 단타 시뮬레이터.

    전략: 갭 앤 고 + ORB 시뮬레이션
    - 전일 대비 갭업 +2% 이상 → 장 시작 시 진입
    - 목표가: 진입가 + gap_size (R:R = 1:2)
    - 손절가: 전일 종가 (갭 하단)
    - EOD 청산: 당일 종가로 강제 청산 (30분 전 기준 close 사용)
    """

    def __init__(
        self,
        gap_threshold: float = 0.02,    # 최소 갭 +2%
        rr_ratio: float = 1.5,           # Risk:Reward 비율 (기본 1.5로 낮춤)
        max_positions: int = 3,          # 최대 동시 포지션
        position_size_pct: float = 0.20, # 포지션당 자본 비율
        commission: float = 0.00015,     # 편도 수수료
        tax: float = 0.002,              # 매도 시 증권거래세
        max_gap_pct: float = 0.15,       # 최대 갭 % (15% 이상은 이상치 제외)
        stop_offset: float = 0.015,      # 진입가 기준 손절 거리 (1.5%)
    ):
        self.gap_threshold = gap_threshold
        self.rr_ratio = rr_ratio
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.tax = tax
        self.max_gap_pct = max_gap_pct
        self.stop_offset = stop_offset

    def simulate(self, ohlcv_df: pd.DataFrame,
                 initial_capital: float = 100_000_000) -> pd.DataFrame:
        """
        일봉 OHLC 기반 단타 시뮬레이션 실행.

        Args:
            ohlcv_df: long 형태 OHLCV (ticker, date, open, high, low, close, volume)
            initial_capital: 초기 자본 (원)

        Returns:
            DataFrame: 일별 포트폴리오 가치 및 거래 내역
        """
        df = ohlcv_df.copy()
        df['prev_close'] = df.groupby('ticker')['close'].shift(1)
        df['gap_pct'] = (df['open'] - df['prev_close']) / df['prev_close']
        df = df.dropna(subset=['prev_close'])

        dates = sorted(df['date'].unique())
        capital = initial_capital
        portfolio_values = []
        trades_log = []

        for date in dates:
            day_df = df[df['date'] == date].copy()

            # 갭 조건 충족 종목 선정 (상위 max_positions개)
            # max_gap_pct 이상 갭은 이상치(상한가 인접 등)로 제외
            candidates = day_df[
                (day_df['gap_pct'] >= self.gap_threshold) &
                (day_df['gap_pct'] <= self.max_gap_pct)
            ].copy()
            candidates = candidates.nlargest(self.max_positions, 'gap_pct')

            day_pnl = 0.0
            for _, row in candidates.iterrows():
                entry_price = row['open']
                prev_close = row['prev_close']
                gap = entry_price - prev_close
                # 손절: 진입가 기준 고정 offset (너무 넓은 갭으로 손절이 커지는 것 방지)
                stop = entry_price * (1 - self.stop_offset)
                risk = entry_price - stop
                target = entry_price + risk * self.rr_ratio  # 목표가

                # 포지션 크기 계산
                pos_value = capital * self.position_size_pct / max(1, len(candidates))
                shares = int(pos_value / entry_price)
                if shares <= 0:
                    continue

                # 시뮬레이션: 당일 high/low 기준 청산 시뮬
                if row['high'] >= target:
                    exit_price = target
                    exit_type = 'target'
                elif row['low'] <= stop:
                    exit_price = stop
                    exit_type = 'stop'
                else:
                    exit_price = row['close']
                    exit_type = 'eod'

                cost_buy = shares * entry_price * self.commission
                cost_sell = shares * exit_price * (self.commission + self.tax)
                trade_pnl = shares * (exit_price - entry_price) - cost_buy - cost_sell

                day_pnl += trade_pnl
                trades_log.append({
                    'date': date,
                    'ticker': row['ticker'],
                    'entry': entry_price,
                    'exit': exit_price,
                    'shares': shares,
                    'gap_pct': row['gap_pct'],
                    'exit_type': exit_type,
                    'pnl': trade_pnl,
                })

            capital += day_pnl
            portfolio_values.append({'date': date, 'portfolio_value': capital})

        results_df = pd.DataFrame(portfolio_values).set_index('date')
        results_df.index = pd.to_datetime(results_df.index)

        trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
        return results_df, trades_df


def compute_metrics(portfolio_df: pd.DataFrame,
                    initial_capital: float) -> dict:
    """기본 성과 지표 계산."""
    pv = portfolio_df['portfolio_value']
    total_return = (pv.iloc[-1] / initial_capital) - 1
    n_years = (pv.index[-1] - pv.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    daily_ret = pv.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    rolling_max = pv.cummax()
    drawdown = (pv - rolling_max) / rolling_max
    mdd = drawdown.min()
    return {
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'mdd': mdd,
        'n_trades': 0,  # filled later
    }


def get_all_tickers_from_db() -> list:
    """DB에서 모든 종목 코드 반환."""
    import sqlite3
    ds = get_data_store()
    try:
        conn = sqlite3.connect(ds.db_path)
        df = pd.read_sql("SELECT DISTINCT ticker FROM price_data", conn)
        conn.close()
        return df['ticker'].tolist()
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description='Korean Intraday Backtest (OHLC Simulation)')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--capital', type=float, default=100_000_000,
                        help='초기 자본 (원)')
    parser.add_argument('--gap-threshold', type=float, default=0.02,
                        help='갭 최소 비율 (기본: 2%%)')
    parser.add_argument('--max-positions', type=int, default=3,
                        help='최대 동시 포지션 (기본: 3)')
    parser.add_argument('--position-pct', type=float, default=0.20,
                        help='포지션당 자본 비율 (기본: 20%%)')
    parser.add_argument('--rr-ratio', type=float, default=2.0,
                        help='Risk:Reward 비율 (기본: 2.0)')
    parser.add_argument('--output-dir', default='data/results')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1) 종목 리스트
    logger.info("DB에서 종목 리스트 로드 중...")
    tickers = get_all_tickers_from_db()
    if not tickers:
        logger.error("종목 리스트를 가져올 수 없습니다.")
        sys.exit(1)
    logger.info(f"  {len(tickers)}개 종목")

    # 2) OHLCV 로드
    logger.info("OHLCV 데이터 로드 중...")
    ohlcv = load_ohlcv_long(tickers, args.start, args.end)
    logger.info(f"  {len(ohlcv):,}행 로드됨")

    # 3) 시뮬레이션
    simulator = IntradayOHLCSimulator(
        gap_threshold=args.gap_threshold,
        rr_ratio=args.rr_ratio,
        max_positions=args.max_positions,
        position_size_pct=args.position_pct,
    )
    logger.info("일봉 OHLC 시뮬레이션 실행 중...")
    portfolio_df, trades_df = simulator.simulate(ohlcv, initial_capital=args.capital)

    # 4) 성과 지표
    metrics = compute_metrics(portfolio_df, args.capital)
    metrics['n_trades'] = len(trades_df)

    print("\n" + "=" * 60)
    print("Korean Intraday Backtest (Gap-and-Go, OHLC Simulation)")
    print("=" * 60)
    print(f"기간: {args.start} ~ {args.end}")
    print(f"초기 자본: {args.capital:,.0f}원")
    print(f"갭 최소 비율: {args.gap_threshold:.0%}")
    print(f"R:R 비율: {args.rr_ratio:.1f}")
    print()
    print(f"연환산 수익률 (CAGR): {metrics['cagr']:.2%}")
    print(f"누적 수익률:          {metrics['total_return']:.2%}")
    print(f"샤프 지수:            {metrics['sharpe']:.2f}")
    print(f"최대 낙폭 (MDD):      {metrics['mdd']:.2%}")
    print(f"총 거래 횟수:         {metrics['n_trades']:,}건")

    if not trades_df.empty:
        print(f"\n거래 유형별 분포:")
        print(trades_df['exit_type'].value_counts().to_string())
        win_rate = (trades_df['pnl'] > 0).mean()
        print(f"승률: {win_rate:.2%}")

    # 5) 저장
    out_dir = Path(project_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

    pv_path = out_dir / f'intraday_portfolio_{suffix}.csv'
    portfolio_df.to_csv(pv_path)
    logger.info(f"포트폴리오 가치 저장: {pv_path}")

    if not trades_df.empty:
        trades_path = out_dir / f'intraday_trades_{suffix}.csv'
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"거래 내역 저장: {trades_path}")

    print(f"\n결과 저장: {out_dir}")


if __name__ == '__main__':
    main()
