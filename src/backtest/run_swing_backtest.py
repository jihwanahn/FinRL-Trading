"""
Swing Trading Backtest Runner (Korean Market)
=============================================

Module B: 스윙 트레이딩 백테스트
SwingSignalEngine + BacktestEngine 연결 스크립트

사용법:
    python src/backtest/run_swing_backtest.py \
        --start 2015-01-01 --end 2024-12-31 \
        --max-positions 10 --stop-loss -0.05

출력:
    data/results/swing_backtest_YYYYMMDD.xlsx
    data/results/swing_backtest_YYYYMMDD_performance.html
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
from src.backtest.backtest_engine import BacktestConfig, BacktestEngine
from src.strategies.swing.swing_signal_engine import SwingSignalEngine

logger = logging.getLogger(__name__)


def load_price_data_wide(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """DataStore에서 종가 데이터를 wide 형태로 반환 (날짜 × 종목)."""
    ds = get_data_store()
    long_df = ds.get_price_data(tickers, start_date, end_date)
    if long_df.empty:
        raise RuntimeError("가격 데이터가 없습니다. 먼저 krx_fetcher.py --fetch-kospi200 를 실행하세요.")

    # 종목코드 컬럼 확인
    tic_col = next((c for c in ['tic', 'ticker', 'gvkey'] if c in long_df.columns), None)
    date_col = next((c for c in ['datadate', 'date'] if c in long_df.columns), None)
    price_col = next((c for c in ['adj_close', 'prccd', 'close'] if c in long_df.columns), None)

    if not (tic_col and date_col and price_col):
        raise RuntimeError(f"가격 데이터 컬럼 확인 실패. Columns: {long_df.columns.tolist()}")

    wide_df = (
        long_df[[date_col, tic_col, price_col]]
        .rename(columns={date_col: 'date', tic_col: 'ticker', price_col: 'close'})
        .pivot_table(index='date', columns='ticker', values='close', aggfunc='last')
    )
    wide_df.index = pd.to_datetime(wide_df.index)
    wide_df = wide_df.sort_index()
    logger.info(f"가격 데이터 로드: {wide_df.shape[0]}일 × {wide_df.shape[1]}종목")
    return wide_df


def get_kospi200_tickers() -> list:
    """DataStore에서 KOSPI200 구성 종목 리스트 반환."""
    ds = get_data_store()
    import sqlite3
    conn = sqlite3.connect(ds.db_path)
    try:
        df = pd.read_sql("SELECT DISTINCT tickers FROM kospi200_components", conn)
        all_tickers = set()
        for row in df['tickers']:
            if row:
                all_tickers.update(str(row).split(','))
        return [t.strip().zfill(6) for t in all_tickers if t.strip()]
    except Exception:
        pass
    finally:
        conn.close()

    # DataStore가 없으면 가격 데이터에서 추출
    try:
        conn2 = sqlite3.connect(ds.db_path)
        df2 = pd.read_sql("SELECT DISTINCT ticker FROM price_data", conn2)
        conn2.close()
        return df2['ticker'].tolist()
    except Exception:
        return []


def generate_swing_weights(price_wide: pd.DataFrame,
                           engine: SwingSignalEngine,
                           max_positions: int = 10,
                           min_signal: float = 0.30) -> pd.DataFrame:
    """
    각 종목별 스윙 신호를 계산하고 주간 포트폴리오 비중을 반환한다.

    Returns:
        DataFrame: 날짜(주별) × 종목 비중
    """
    logger.info("스윙 신호 계산 중...")
    all_signals: dict = {}

    for ticker in price_wide.columns:
        series = price_wide[ticker].dropna()
        if len(series) < 70:  # MA60 + 여유
            continue
        df_ticker = pd.DataFrame({
            'date': series.index,
            'close': series.values,
            'open': series.values,
            'high': series.values,
            'low': series.values,
            'volume': np.ones(len(series)),
        }).reset_index(drop=True)
        try:
            sig = engine.generate_signal_one_ticker(df_ticker)
            all_signals[ticker] = sig
        except Exception as e:
            logger.debug(f"Signal failed for {ticker}: {e}")

    if not all_signals:
        raise RuntimeError("신호 생성 실패: 모든 종목에서 신호를 계산할 수 없습니다.")

    signals_df = pd.DataFrame(all_signals)
    signals_df.index = pd.to_datetime(signals_df.index)

    # 비중 계산
    weights_df = engine.generate_weights_from_signals(
        signals_df, max_positions=max_positions, min_signal=min_signal
    )
    logger.info(f"스윙 신호 완료: {len(weights_df)}주 × {len(weights_df.columns)}종목")
    return weights_df


def main():
    parser = argparse.ArgumentParser(description='Korean Swing Trading Backtest')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--capital', type=float, default=100_000_000,
                        help='초기 자본 (원, 기본: 1억원)')
    parser.add_argument('--max-positions', type=int, default=10)
    parser.add_argument('--min-signal', type=float, default=0.40,
                        help='최소 신호 강도 (0~1, 기본: 0.40)')
    parser.add_argument('--rebalance-freq', default='2W',
                        choices=['W', '2W', 'ME', 'QS'],
                        help='리밸런싱 주기 (기본: 2W=격주)')
    parser.add_argument('--stop-loss', type=float, default=-0.05,
                        help='절대 손절 비율 (기본: -5%)')
    parser.add_argument('--transaction-cost', type=float, default=0.003,
                        help='왕복 거래비용 (기본: 0.3%)')
    parser.add_argument('--output-dir', default='data/results')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1) 종목 리스트
    logger.info("KOSPI200 종목 로드 중...")
    tickers = get_kospi200_tickers()
    if not tickers:
        logger.error("종목 리스트를 가져올 수 없습니다.")
        sys.exit(1)
    logger.info(f"  {len(tickers)}개 종목 로드됨")

    # 2) 가격 데이터 로드
    price_wide = load_price_data_wide(tickers, args.start, args.end)

    # 3) 스윙 신호 생성
    rebalance_freq = getattr(args, 'rebalance_freq', '2W')
    engine = SwingSignalEngine(rebalance_freq=rebalance_freq)
    weight_signals = generate_swing_weights(
        price_wide, engine,
        max_positions=args.max_positions,
        min_signal=args.min_signal,
    )

    # 3b) 비중 날짜를 가격 데이터의 실제 거래일로 스냅
    #     (주말/공휴일이 포함된 경우 직전 거래일로 이동)
    trading_days = price_wide.index
    weight_signals_aligned = weight_signals.copy()
    new_index = []
    for dt in weight_signals.index:
        # 해당 날짜 이전 (또는 당일)의 가장 가까운 거래일 찾기
        candidates = trading_days[trading_days <= dt]
        if len(candidates) > 0:
            new_index.append(candidates[-1])
        else:
            new_index.append(dt)
    weight_signals_aligned.index = pd.DatetimeIndex(new_index)
    # 같은 날짜로 겹친 경우 마지막 값 유지
    weight_signals_aligned = weight_signals_aligned.groupby(level=0).last()
    weight_signals = weight_signals_aligned

    # 4) 백테스트 실행
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        rebalance_freq=rebalance_freq,
        initial_capital=args.capital,
        transaction_cost=args.transaction_cost,
        exchange='XKRX',
        benchmark_tickers=['069500'],  # KODEX 200
    )
    bt_engine = BacktestEngine(config)
    logger.info("백테스트 실행 중...")
    result = bt_engine.run_backtest('KR-Swing', price_wide, weight_signals)

    # 5) 결과 출력
    print("\n" + "=" * 60)
    print("Korean Swing Trading Backtest Results")
    print("=" * 60)
    print(f"기간: {args.start} ~ {args.end}")
    print(f"초기 자본: {args.capital:,.0f}원")
    print()

    metrics = result.metrics
    print(f"연환산 수익률 (CAGR): {result.annualized_return:.2%}")
    print(f"누적 수익률:          {metrics.get('total_return', 0):.2%}")
    print(f"샤프 지수:            {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"최대 낙폭 (MDD):      {metrics.get('max_drawdown', 0):.2%}")
    print(f"변동성 (연환산):      {metrics.get('annual_volatility', 0):.2%}")
    # win_rate: compute from daily portfolio returns
    pv = result.portfolio_values if isinstance(result.portfolio_values, pd.Series) else result.portfolio_values.iloc[:, 0]
    daily_ret = pv.pct_change().dropna()
    win_rate = (daily_ret > 0).mean() if len(daily_ret) > 0 else 0.0
    print(f"일 기준 수익 비율:    {win_rate:.2%}")
    print()

    if result.benchmark_annualized:
        print("벤치마크 대비:")
        for bmk, bret in result.benchmark_annualized.items():
            print(f"  {bmk}: {bret:.2%} p.a.")

    # 6) 파일 저장
    out_dir = Path(project_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = out_dir / f'swing_backtest_{suffix}.csv'
    result.portfolio_values.to_csv(csv_path)
    logger.info(f"포트폴리오 가치 저장: {csv_path}")

    weights_path = out_dir / f'swing_weights_{suffix}.csv'
    weight_signals.to_csv(weights_path)
    logger.info(f"비중 히스토리 저장: {weights_path}")

    print(f"\n결과 저장: {out_dir}")


if __name__ == '__main__':
    main()
