"""
KRX Fundamental Data → ML Feature Table Preparation
====================================================

KOSPI200 데이터를 ml_bucket_selection.py가 사용하는 fundamental_data 스키마로 변환.

사용법:
    python src/data/prepare_krx_ml_data.py \
        --start 2015-01-01 --end 2024-12-31 \
        --out data/finrl_trading.db

결과:
    DB의 fundamental_data 테이블에 KOSPI200 레코드 추가 (gsector=WICS)
    y_return: 다음 분기 초 대비 1분기 후 종가 수익률
"""

import os
import sys
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)

# WICS sector → ml_bucket_selection SECTOR_TO_BUCKET mapping
WICS_TO_GSECTOR = {
    "IT": "45",                   # Technology (GICS 45)
    "커뮤니케이션서비스": "50",    # Communication Services (GICS 50)
    "경기관련소비재": "25",        # Consumer Discretionary (GICS 25)
    "금융": "40",                  # Financials (GICS 40)
    "산업재": "20",                # Industrials (GICS 20)
    "에너지": "10",                # Energy (GICS 10)
    "소재": "15",                  # Materials (GICS 15)
    "부동산": "60",                # Real Estate (GICS 60)
    "건강관리": "35",              # Health Care (GICS 35)
    "필수소비재": "30",            # Consumer Staples (GICS 30)
    "유틸리티": "55",              # Utilities (GICS 55)
}

# Quarter-end → trade date: first business day 2 months later
# (similar to US 45-day filing lag)
def qtr_end_to_trade_date(qtr_end: str) -> str:
    """Convert quarter-end date to trade date (2 months later)."""
    dt = pd.Timestamp(qtr_end)
    # Add ~60 days
    trade = dt + pd.DateOffset(days=60)
    # Round to first of month
    trade = trade.replace(day=1)
    return trade.strftime('%Y-%m-%d')


# KRX 업종 키워드 → WICS 버킷 매핑
INDUSTRY_TO_WICS = [
    # IT / 정보기술
    ('반도체', 'it'), ('전자부품', 'it'), ('전자기기', 'it'), ('컴퓨터', 'it'),
    ('소프트웨어', 'it'), ('정보', 'it'), ('데이터', 'it'), ('인터넷', 'it'),
    ('통신 및 방송 장비', 'it'), ('전기전자', 'it'),
    # 커뮤니케이션서비스
    ('통신업', '커뮤니케이션서비스'), ('방송업', '커뮤니케이션서비스'), ('영화', '커뮤니케이션서비스'),
    ('게임', '커뮤니케이션서비스'), ('포털', '커뮤니케이션서비스'),
    # 경기관련소비재
    ('자동차', '경기관련소비재'), ('의류', '경기관련소비재'), ('가구', '경기관련소비재'),
    ('호텔', '경기관련소비재'), ('여행', '경기관련소비재'), ('오락', '경기관련소비재'),
    ('백화점', '경기관련소비재'), ('소매업', '경기관련소비재'),
    # 금융
    ('금융업', '금융'), ('은행', '금융'), ('보험', '금융'), ('증권', '금융'),
    ('자산운용', '금융'), ('여신', '금융'),
    # 산업재
    ('기계', '산업재'), ('항공', '산업재'), ('조선', '산업재'), ('건설', '산업재'),
    ('철도', '산업재'), ('운송', '산업재'), ('물류', '산업재'), ('항만', '산업재'),
    ('포장', '산업재'), ('인쇄', '산업재'),
    # 에너지
    ('에너지', '에너지'), ('석유', '에너지'), ('가스', '에너지'), ('전기 생산', '에너지'),
    # 소재
    ('화학물질', '소재'), ('철강', '소재'), ('비철금속', '소재'), ('광업', '소재'),
    ('플라스틱', '소재'), ('고무', '소재'), ('종이', '소재'), ('목재', '소재'),
    ('전지', '소재'),  # 배터리/이차전지
    # 부동산
    ('부동산', '부동산'),
    # 건강관리
    ('의약', '건강관리'), ('의료', '건강관리'), ('병원', '건강관리'), ('제약', '건강관리'),
    ('바이오', '건강관리'),
    # 필수소비재
    ('식품', '필수소비재'), ('음료', '필수소비재'), ('담배', '필수소비재'),
    ('생활용품', '필수소비재'), ('농업', '필수소비재'), ('수산', '필수소비재'),
    # 유틸리티
    ('전기 공급', '유틸리티'), ('수도', '유틸리티'), ('폐수', '유틸리티'),
]


def industry_to_wics(industry_str: str) -> str:
    """KRX Industry 문자열 → WICS 섹터명 변환."""
    if not isinstance(industry_str, str):
        return '산업재'
    s = industry_str.strip()
    for keyword, wics in INDUSTRY_TO_WICS:
        if keyword in s:
            return wics
    return '산업재'  # default


def get_sector_map_fdr() -> Dict[str, str]:
    """FDR KRX-DESC에서 종목→WICS 섹터 딕셔너리 반환."""
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing('KRX-DESC')
        if df is None or df.empty:
            return {}
        if 'Code' in df.columns and 'Industry' in df.columns:
            df['Code'] = df['Code'].astype(str).str.zfill(6)
            df['wics'] = df['Industry'].apply(industry_to_wics)
            mapping = dict(zip(df['Code'], df['wics']))
            logger.info(f"FDR sector map: {len(mapping)} tickers")
            dist = df['wics'].value_counts()
            logger.info(f"Sector distribution:\n{dist.to_string()}")
            return mapping
    except Exception as e:
        logger.warning(f"FDR sector fetch failed: {e}")
    return {}


def build_krx_ml_table(db_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    krx_fundamental_data + price_data를 결합하여 ml_bucket_selection 호환 DataFrame 생성.

    Returns:
        DataFrame with columns matching fundamental_data schema
    """
    conn = sqlite3.connect(db_path)

    # 1) Load KRX fundamental data
    kf = pd.read_sql(
        f"SELECT * FROM krx_fundamental_data WHERE datadate >= '{start_date}' AND datadate <= '{end_date}'",
        conn
    )
    logger.info(f"KRX fundamentals: {len(kf)} records, {kf['ticker'].nunique()} tickers")

    if kf.empty:
        conn.close()
        raise RuntimeError("No KRX fundamental data found. Run krx_fetcher.py --fetch-kospi200 first.")

    # 2) Load price data
    tickers = kf['ticker'].unique().tolist()
    tickers_sql = ", ".join(f"'{t}'" for t in tickers)
    price = pd.read_sql(
        f"""SELECT ticker, date, close FROM price_data
            WHERE ticker IN ({tickers_sql})
            AND date >= '{start_date}' AND date <= '2025-12-31'
            ORDER BY ticker, date""",
        conn
    )
    conn.close()

    logger.info(f"Price data: {len(price)} records")
    price['date'] = pd.to_datetime(price['date'])

    # 3) Map KRX columns to ML schema
    kf['datadate'] = pd.to_datetime(kf['datadate'])

    # Sector: use FDR to get WICS sector names (stored as Korean string matching SECTOR_TO_BUCKET)
    logger.info("Fetching sector map from FDR...")
    sector_map = get_sector_map_fdr()  # ticker → sector string

    if sector_map:
        kf['gsector'] = kf['ticker'].map(sector_map).fillna('산업재')
        logger.info(f"Sector distribution:\n{kf['gsector'].value_counts().to_string()}")
    else:
        # Fallback: use krx_fundamental_data.gsector if present, else '산업재'
        kf['gsector'] = kf['gsector'].fillna('산업재')

    # Direct field mappings
    kf['pe']             = pd.to_numeric(kf['per'], errors='coerce')
    kf['pb']             = pd.to_numeric(kf['pbr'], errors='coerce')
    kf['EPS']            = pd.to_numeric(kf['eps'], errors='coerce')
    kf['BPS']            = pd.to_numeric(kf['bps'], errors='coerce')
    kf['dividend_yield'] = pd.to_numeric(kf['div_yield'], errors='coerce')

    # Income statement / balance sheet raw columns (may be None if DART didn't return them)
    for raw_col in ['revenue', 'gross_profit', 'operating_income',
                    'current_assets', 'current_liabilities',
                    'total_liabilities', 'net_income', 'total_equity']:
        if raw_col not in kf.columns:
            kf[raw_col] = np.nan
        else:
            kf[raw_col] = pd.to_numeric(kf[raw_col], errors='coerce')

    # Compute derived ratios
    # gross_margin = gross_profit / revenue  (or (revenue - COGS) / revenue)
    kf['gross_margin'] = np.where(
        kf['revenue'].notna() & (kf['revenue'] != 0) & kf['gross_profit'].notna(),
        kf['gross_profit'] / kf['revenue'],
        np.nan
    )
    # operating_margin = operating_income / revenue
    kf['operating_margin'] = np.where(
        kf['revenue'].notna() & (kf['revenue'] != 0) & kf['operating_income'].notna(),
        kf['operating_income'] / kf['revenue'],
        np.nan
    )
    # roe = net_income / total_equity
    kf['roe'] = np.where(
        kf['net_income'].notna() & kf['total_equity'].notna() & (kf['total_equity'] != 0),
        kf['net_income'] / kf['total_equity'],
        np.nan
    )
    # cur_ratio = current_assets / current_liabilities
    kf['cur_ratio'] = np.where(
        kf['current_assets'].notna() & kf['current_liabilities'].notna() & (kf['current_liabilities'] != 0),
        kf['current_assets'] / kf['current_liabilities'],
        np.nan
    )
    # debt_ratio = total_liabilities / total_equity (debt-to-equity)
    kf['debt_ratio'] = np.where(
        kf['total_liabilities'].notna() & kf['total_equity'].notna() & (kf['total_equity'] != 0),
        kf['total_liabilities'] / kf['total_equity'],
        np.nan
    )

    # debt_to_equity = same as debt_ratio (total_liabilities / total_equity)
    kf['debt_to_equity'] = kf['debt_ratio']

    # Sanity clipping: holding companies and unusual structures can produce
    # extreme ratios due to DART unit mismatches or near-zero denominators.
    # Values outside these ranges are set to NaN (treated as missing by ML).
    RATIO_BOUNDS = {
        'gross_margin':     (-2.0,  2.0),   # -200% ~ +200%
        'operating_margin': (-5.0,  5.0),   # -500% ~ +500%
        'roe':              (-5.0,  5.0),   # -500% ~ +500%
        'cur_ratio':        (0.0,  50.0),
        'debt_ratio':       (-20.0, 50.0),
        'debt_to_equity':   (-20.0, 50.0),
        'pe':               (0.0, 1000.0),  # PE > 1000 is economically meaningless
        'pb':               (0.0,   50.0),  # PB > 50 is extreme
    }
    for col, (lo, hi) in RATIO_BOUNDS.items():
        if col in kf.columns:
            bad = kf[col].notna() & ((kf[col] < lo) | (kf[col] > hi))
            if bad.sum() > 0:
                logger.debug(f"Clipping {bad.sum()} extreme {col} values to NaN")
                kf.loc[bad, col] = np.nan

    # Fields not available in KRX data → NaN (LightGBM handles NaN natively)
    for col in ['ps', 'peg', 'ev_multiple',
                'fcf_per_share', 'cash_per_share', 'capex_per_share', 'fcf_to_ocf', 'ocf_ratio',
                'debt_to_mktcap',
                'acc_rec_turnover', 'asset_turnover', 'payables_turnover',
                'interest_coverage', 'debt_service_coverage',
                'solvency_ratio']:
        kf[col] = np.nan

    # 4) Compute adj_close_q (closing price at quarter-end)
    price_idx = price.set_index(['ticker', 'date'])['close']

    def get_closest_price(ticker, dt, max_lag_days=5):
        """Get the closest trading day price (looking backward)."""
        try:
            subset = price_idx.loc[ticker]
            # Find dates <= dt within max_lag
            candidates = subset.index[subset.index <= dt]
            if len(candidates) == 0:
                return np.nan
            closest = candidates[-1]
            if (dt - closest).days > max_lag_days:
                return np.nan
            return subset.loc[closest]
        except KeyError:
            return np.nan

    logger.info("Computing adj_close_q (quarter-end prices)...")
    kf['adj_close_q'] = [
        get_closest_price(row['ticker'], row['datadate'])
        for _, row in kf.iterrows()
    ]

    # 5) Compute trade_price (price at trade date: 2 months after quarter-end)
    kf['trade_date'] = kf['datadate'].apply(lambda d: pd.Timestamp(qtr_end_to_trade_date(d.strftime('%Y-%m-%d'))))
    kf['filing_date'] = kf['datadate'].apply(lambda d: (d + pd.DateOffset(days=45)).strftime('%Y-%m-%d'))
    kf['accepted_date'] = kf['filing_date']

    logger.info("Computing trade_price...")
    kf['trade_price'] = [
        get_closest_price(row['ticker'], row['trade_date'], max_lag_days=10)
        for _, row in kf.iterrows()
    ]

    # 6) Compute y_return (1-quarter forward return from trade_date)
    logger.info("Computing y_return (1-quarter forward returns)...")

    def get_forward_price(ticker, trade_dt, forward_days=63):  # ~1 quarter
        try:
            subset = price_idx.loc[ticker]
            target = trade_dt + pd.Timedelta(days=forward_days)
            # Find dates >= target within 10 days window
            candidates = subset.index[(subset.index >= target) &
                                       (subset.index <= target + pd.Timedelta(days=10))]
            if len(candidates) == 0:
                return np.nan
            return subset.loc[candidates[0]]
        except KeyError:
            return np.nan

    kf['_fwd_price'] = [
        get_forward_price(row['ticker'], row['trade_date'])
        for _, row in kf.iterrows()
    ]

    # y_return = forward_price / trade_price - 1
    kf['y_return'] = np.where(
        (kf['trade_price'] > 0) & kf['_fwd_price'].notna(),
        kf['_fwd_price'] / kf['trade_price'] - 1,
        np.nan
    )

    # 7) Final cleanup
    result = kf.rename(columns={'ticker': 'tic'})[[
        'tic', 'datadate', 'gsector', 'adj_close_q', 'trade_price',
        'filing_date', 'accepted_date',
        'pe', 'ps', 'pb', 'peg', 'ev_multiple',
        'EPS', 'roe', 'gross_margin', 'operating_margin',
        'fcf_per_share', 'cash_per_share', 'capex_per_share', 'fcf_to_ocf', 'ocf_ratio',
        'debt_ratio', 'debt_to_equity', 'debt_to_mktcap',
        'cur_ratio',
        'acc_rec_turnover', 'asset_turnover', 'payables_turnover',
        'interest_coverage', 'debt_service_coverage',
        'dividend_yield',
        'solvency_ratio',
        'BPS',
        'y_return',
    ]].copy()

    result['datadate'] = result['datadate'].dt.strftime('%Y-%m-%d')

    # Filter out rows with no price data
    before = len(result)
    result = result[result['adj_close_q'].notna()].copy()
    logger.info(f"After price filter: {len(result)}/{before} records")

    return result


def insert_into_fundamental_data(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert KRX ML records into fundamental_data table."""
    cursor = conn.cursor()

    # Columns we populate (map from our df column → DB column)
    col_map = {
        'tic':           'ticker',
        'datadate':      'datadate',
        'gsector':       'gsector',
        'adj_close_q':   'adj_close_q',
        'trade_price':   'trade_price',
        'filing_date':   'filing_date',
        'accepted_date': 'accepted_date',
        'pe':            'pe',
        'ps':            'ps',
        'pb':            'pb',
        'peg':           'peg',
        'ev_multiple':   'ev_multiple',
        'EPS':           'EPS',
        'roe':           'roe',
        'gross_margin':  'gross_margin',
        'operating_margin': 'operating_margin',
        'fcf_per_share': 'fcf_per_share',
        'cash_per_share':'cash_per_share',
        'capex_per_share':'capex_per_share',
        'fcf_to_ocf':    'fcf_to_ocf',
        'ocf_ratio':     'ocf_ratio',
        'debt_ratio':    'debt_ratio',
        'debt_to_equity':'debt_to_equity',
        'debt_to_mktcap':'debt_to_mktcap',
        'cur_ratio':     'cur_ratio',
        'acc_rec_turnover':'acc_rec_turnover',
        'asset_turnover':'asset_turnover',
        'payables_turnover':'payables_turnover',
        'interest_coverage':'interest_coverage',
        'debt_service_coverage':'debt_service_coverage',
        'dividend_yield':'dividend_yield',
        'solvency_ratio':'solvency_ratio',
        'BPS':           'BPS',
        'y_return':      'y_return',
    }

    # Ensure all DB columns exist (add trade_price if missing)
    existing_cols = set(r[1] for r in cursor.execute("PRAGMA table_info(fundamental_data)").fetchall())
    for db_col in col_map.values():
        if db_col not in existing_cols and db_col not in ('id', 'ticker', 'datadate'):
            try:
                cursor.execute(f"ALTER TABLE fundamental_data ADD COLUMN {db_col} REAL")
                logger.info(f"Added column {db_col} to fundamental_data")
            except Exception as e:
                logger.debug(f"Column {db_col}: {e}")

    df_cols = list(col_map.keys())
    db_cols = [col_map[c] for c in df_cols]
    col_str = ', '.join(db_cols)
    placeholders = ', '.join(['?'] * len(db_cols))

    n = 0
    rows_data = []
    for _, row in df.iterrows():
        vals = []
        for c in df_cols:
            v = row.get(c, None)
            try:
                if pd.isna(v):
                    vals.append(None)
                else:
                    vals.append(v)
            except (TypeError, ValueError):
                vals.append(v)
        rows_data.append(tuple(vals))

    cursor.executemany(
        f"INSERT OR REPLACE INTO fundamental_data ({col_str}) VALUES ({placeholders})",
        rows_data
    )
    n = cursor.rowcount if cursor.rowcount >= 0 else len(rows_data)
    conn.commit()
    return n


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare KRX data for ML bucket selection')
    parser.add_argument('--db', default='data/finrl_trading.db')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no DB insert')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    db_path = os.path.join(project_root, args.db)
    logger.info(f"Building KRX ML feature table from {db_path}")

    df = build_krx_ml_table(db_path, args.start, args.end)

    print(f"\nKRX ML data prepared:")
    print(f"  Records: {len(df)}")
    print(f"  Tickers: {df['tic'].nunique()}")
    print(f"  Date range: {df['datadate'].min()} ~ {df['datadate'].max()}")
    print(f"  y_return coverage: {df['y_return'].notna().sum()}/{len(df)} ({df['y_return'].notna().mean():.1%})")
    print(f"\n  Feature NaN rates:")
    for col in ['pe', 'pb', 'EPS', 'BPS', 'dividend_yield',
                'roe', 'gross_margin', 'operating_margin',
                'cur_ratio', 'debt_ratio']:
        rate = df[col].isna().mean()
        print(f"    {col}: {rate:.1%} NaN")
    print(f"\n  y_return stats:")
    print(df['y_return'].describe().to_string())

    if not args.dry_run:
        conn = sqlite3.connect(db_path)
        n = insert_into_fundamental_data(conn, df)
        conn.close()
        logger.info(f"Inserted {n} records into fundamental_data table")
        print(f"\nInserted {n} records into fundamental_data (DB: {args.db})")
    else:
        print("\n[DRY RUN] No records inserted.")
        print(df[['tic', 'datadate', 'pe', 'pb', 'EPS', 'BPS', 'y_return']].head(10).to_string())


if __name__ == '__main__':
    main()
