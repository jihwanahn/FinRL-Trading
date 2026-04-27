"""
Quarterly Factor Backtest Runner (Korean Market)
=================================================

Module A: ML 분기 팩터 백테스트
ml_bucket_selection.py의 예측 CSV를 포트폴리오 비중으로 변환 후 BacktestEngine 실행.

사용법:
    python src/backtest/run_quarterly_backtest.py \
        --predictions data/kospi200_ml_bucket_predictions_*.csv \
        --start 2023-01-01 --end 2024-12-31

출력:
    data/results/quarterly_backtest_YYYYMMDD.csv
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

logger = logging.getLogger(__name__)


def load_predictions(pred_path: str) -> pd.DataFrame:
    """ML 예측 CSV 로드 및 ticker 정규화."""
    df = pd.read_csv(pred_path)
    # ticker 앞 0 복원 (int로 읽혔을 경우)
    df['tic'] = df['tic'].astype(str).str.zfill(6)
    df['tradedate'] = pd.to_datetime(df['tradedate'])
    df['datadate'] = pd.to_datetime(df['datadate'])
    logger.info(f"예측 데이터: {len(df)}행, {df['tic'].nunique()}종목, "
                f"기간: {df['tradedate'].min().date()} ~ {df['tradedate'].max().date()}")
    return df


def build_weight_signals(pred_df: pd.DataFrame,
                          top_n_per_bucket: int = 5,
                          equal_bucket_weight: bool = True) -> pd.DataFrame:
    """
    ML 예측 → 분기별 포트폴리오 비중 DataFrame.

    전략:
    - 각 버킷에서 predicted_return 상위 top_n_per_bucket 종목 선택
    - 버킷 내 equal weight → 버킷 간 equal weight
    """
    buckets = pred_df['bucket'].unique()
    all_dates = sorted(pred_df['tradedate'].unique())
    logger.info(f"버킷: {list(buckets)}, 분기: {len(all_dates)}개")

    rows = []
    for trade_dt in all_dates:
        qdf = pred_df[pred_df['tradedate'] == trade_dt]
        bucket_picks = {}

        for bucket in buckets:
            bdf = qdf[qdf['bucket'] == bucket].sort_values('predicted_return', ascending=False)
            top = bdf.head(top_n_per_bucket)
            if len(top) == 0:
                continue
            bucket_picks[bucket] = top['tic'].tolist()

        if not bucket_picks:
            continue

        # 버킷 간 equal weight
        n_buckets = len(bucket_picks)
        bucket_weight = 1.0 / n_buckets

        row = {}
        for bucket, tickers in bucket_picks.items():
            per_stock = bucket_weight / len(tickers)
            for tic in tickers:
                row[tic] = row.get(tic, 0.0) + per_stock

        rows.append({'date': trade_dt, **row})

    weights_df = pd.DataFrame(rows).set_index('date').fillna(0.0)
    weights_df.index = pd.DatetimeIndex(weights_df.index)
    logger.info(f"비중 테이블: {len(weights_df)}행 × {len(weights_df.columns)}종목")
    return weights_df


def load_price_wide(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """DataStore에서 wide format 가격 데이터 로드."""
    ds = get_data_store()
    long_df = ds.get_price_data(tickers, start_date, end_date)
    if long_df.empty:
        raise RuntimeError("가격 데이터가 없습니다. krx_fetcher.py --fetch-kospi200 를 먼저 실행하세요.")

    tic_col = next((c for c in ['tic', 'ticker'] if c in long_df.columns), None)
    date_col = next((c for c in ['datadate', 'date'] if c in long_df.columns), None)
    price_col = next((c for c in ['adj_close', 'close'] if c in long_df.columns), None)

    wide = (
        long_df[[date_col, tic_col, price_col]]
        .rename(columns={date_col: 'date', tic_col: 'ticker', price_col: 'close'})
        .pivot_table(index='date', columns='ticker', values='close', aggfunc='last')
    )
    wide.index = pd.to_datetime(wide.index)
    wide = wide.sort_index()
    logger.info(f"가격 데이터: {wide.shape[0]}일 × {wide.shape[1]}종목")
    return wide


def snap_weights_to_trading_days(weights_df: pd.DataFrame,
                                   trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """비중 날짜를 실제 거래일로 스냅 (당일 이후 첫 거래일)."""
    new_index = []
    for dt in weights_df.index:
        # 당일 이후 거래일 (tradedate는 보통 월초 → 해당 월 첫 거래일)
        candidates = trading_days[trading_days >= dt]
        if len(candidates) > 0:
            new_index.append(candidates[0])
        else:
            candidates_before = trading_days[trading_days <= dt]
            new_index.append(candidates_before[-1] if len(candidates_before) > 0 else dt)

    weights_df = weights_df.copy()
    weights_df.index = pd.DatetimeIndex(new_index)
    weights_df = weights_df.groupby(level=0).last()
    return weights_df


def main():
    parser = argparse.ArgumentParser(description='Korean Quarterly Factor Backtest')
    parser.add_argument('--predictions', required=True,
                        help='ML 예측 CSV 경로 (ml_bucket_predictions_*.csv)')
    parser.add_argument('--start', default='2023-01-01', help='백테스트 시작일')
    parser.add_argument('--end', default='2024-12-31', help='백테스트 종료일')
    parser.add_argument('--capital', type=float, default=100_000_000,
                        help='초기 자본 (원, 기본: 1억)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='버킷당 상위 종목 수 (기본: 5)')
    parser.add_argument('--transaction-cost', type=float, default=0.003)
    parser.add_argument('--output-dir', default='data/results')
    parser.add_argument('--pred-col', default='predicted_return',
                        help='예측값 컬럼 (기본: predicted_return, 또는 pred_LGBM, pred_ensemble_avg 등)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 1) 예측 로드
    pred_df = load_predictions(args.predictions)

    # 기간 필터
    pred_df = pred_df[
        (pred_df['tradedate'] >= args.start) &
        (pred_df['tradedate'] <= args.end)
    ]
    if pred_df.empty:
        logger.error(f"지정 기간({args.start}~{args.end})에 예측 데이터가 없습니다.")
        sys.exit(1)

    # 2) 비중 생성
    pred_col = args.pred_col
    if pred_col not in pred_df.columns:
        logger.error(f"--pred-col '{pred_col}'이 예측 CSV에 없습니다. 사용 가능: {[c for c in pred_df.columns if 'pred' in c]}")
        sys.exit(1)
    if pred_col != 'predicted_return':
        pred_df = pred_df.copy()
        pred_df['predicted_return'] = pred_df[pred_col]
        logger.info(f"예측 컬럼 교체: {pred_col} (std={pred_df['predicted_return'].std():.4f})")
    weight_signals = build_weight_signals(pred_df, top_n_per_bucket=args.top_n)

    # 3) 가격 데이터 로드
    all_tickers = weight_signals.columns.tolist()
    logger.info(f"가격 로드 중... {len(all_tickers)}종목")
    price_wide = load_price_wide(all_tickers, args.start, args.end)

    # 공통 종목만 사용
    common = [t for t in weight_signals.columns if t in price_wide.columns]
    missing = set(weight_signals.columns) - set(common)
    if missing:
        logger.warning(f"가격 없는 종목 {len(missing)}개 제외: {list(missing)[:5]}...")
    weight_signals = weight_signals[common]
    # 비중 재정규화
    row_sums = weight_signals.sum(axis=1)
    weight_signals = weight_signals.div(row_sums.where(row_sums > 0, 1.0), axis=0)

    # 4) 비중 날짜 → 실제 거래일 스냅
    weight_signals = snap_weights_to_trading_days(weight_signals, price_wide.index)

    # 5) 백테스트 실행
    config = BacktestConfig(
        start_date=args.start,
        end_date=args.end,
        rebalance_freq='Q',
        initial_capital=args.capital,
        transaction_cost=args.transaction_cost,
        exchange='XKRX',
        benchmark_tickers=['069500'],  # KODEX 200
    )
    bt_engine = BacktestEngine(config)
    logger.info("백테스트 실행 중...")
    result = bt_engine.run_backtest('KR-Quarterly', price_wide, weight_signals)

    # 6) 결과 출력
    print("\n" + "=" * 60)
    print("Korean Quarterly Factor Backtest Results")
    print("=" * 60)
    print(f"기간: {args.start} ~ {args.end}")
    print(f"초기 자본: {args.capital:,.0f}원")
    print(f"버킷당 종목수: {args.top_n}")
    print()

    metrics = result.metrics
    print(f"연환산 수익률 (CAGR): {result.annualized_return:.2%}")
    print(f"누적 수익률:          {metrics.get('total_return', 0):.2%}")
    print(f"샤프 지수:            {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"최대 낙폭 (MDD):      {metrics.get('max_drawdown', 0):.2%}")
    print(f"변동성 (연환산):      {metrics.get('annual_volatility', 0):.2%}")

    pv = result.portfolio_values if isinstance(result.portfolio_values, pd.Series) \
        else result.portfolio_values.iloc[:, 0]
    win_rate = (pv.pct_change().dropna() > 0).mean()
    print(f"일 기준 수익 비율:    {win_rate:.2%}")
    print()

    if result.benchmark_annualized:
        print("벤치마크 대비:")
        for bmk, bret in result.benchmark_annualized.items():
            print(f"  {bmk} (KODEX200): {bret:.2%} p.a.")

    # 7) 저장
    out_dir = Path(project_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = out_dir / f'quarterly_backtest_{suffix}.csv'
    result.portfolio_values.to_csv(csv_path)
    logger.info(f"포트폴리오 가치 저장: {csv_path}")

    weights_path = out_dir / f'quarterly_weights_{suffix}.csv'
    weight_signals.to_csv(weights_path)
    logger.info(f"비중 히스토리 저장: {weights_path}")

    print(f"\n결과 저장: {out_dir}")
    return csv_path


if __name__ == '__main__':
    main()
