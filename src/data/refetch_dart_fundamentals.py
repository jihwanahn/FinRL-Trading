"""
DART Fundamentals Refetch Script
=================================
기존 krx_fundamental_data의 revenue 등 신규 컬럼이 NULL인 레코드를
DART API로 재수집하여 UPDATE합니다.

사용법:
    python src/data/refetch_dart_fundamentals.py \
        --start 2015-01-01 --end 2024-12-31
"""

import os
import sys
import sqlite3
import logging
import time
import requests
from typing import Dict, Optional
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from tqdm import tqdm

logger = logging.getLogger(__name__)


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(str(s).replace(',', '').strip())
    except (ValueError, TypeError):
        return None


def get_dart_corp_codes(api_key: str) -> Dict[str, str]:
    """Download DART corp_code mapping (ticker → corp_code)."""
    import zipfile, io, xml.etree.ElementTree as ET
    url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        xml_data = z.read(z.namelist()[0])
    root = ET.fromstring(xml_data)
    mapping = {}
    for corp in root.findall('list'):
        stock_code = (corp.findtext('stock_code') or '').strip()
        corp_code  = (corp.findtext('corp_code') or '').strip()
        if stock_code and len(stock_code) == 6:
            mapping[stock_code] = corp_code
    logger.info(f"DART corp_code: {len(mapping)} entries")
    return mapping


TARGET_ACCOUNTS = {
    '기본주당이익':     'eps',
    '희석주당이익':     'eps',
    '주당순자산':      'bps',
    '1주당순자산':     'bps',
    '주당장부가치':     'bps',
    '주당배당금':      'dps',
    '주당현금배당금':   'dps',
    '당기순이익':      'net_income',
    '자본총계':       'total_equity',
    # Income Statement
    '매출액':          'revenue',
    '수익(매출액)':    'revenue',
    '영업수익':        'revenue',
    '매출총이익':      'gross_profit',
    '영업이익':        'operating_income',
    '영업이익(손실)':  'operating_income',
    # Balance Sheet
    '유동자산':        'current_assets',
    '유동부채':        'current_liabilities',
    '부채총계':        'total_liabilities',
}


def fetch_fs(corp_code: str, bsns_year: str, reprt_code: str,
             api_key: str, min_interval: float = 0.07) -> Dict:
    """DART fnlttSinglAcntAll.json에서 재무 항목 추출."""
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

    def _call(fs_div):
        time.sleep(min_interval)
        try:
            resp = requests.get(url, params={
                'crtfc_key': api_key,
                'corp_code': corp_code,
                'bsns_year': bsns_year,
                'reprt_code': reprt_code,
                'fs_div': fs_div,
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') != '000':
                return []
            return data.get('list', [])
        except Exception as e:
            logger.debug(f"DART {fs_div} error: {e}")
            return []

    items = _call('CFS') or _call('OFS')

    result = {}
    for item in items:
        acct_nm = (item.get('account_nm') or '').strip()
        for ko, key in TARGET_ACCOUNTS.items():
            if ko in acct_nm and key not in result:
                val = _safe_float((item.get('thstrm_amount') or '').replace(',', ''))
                if val is not None:
                    result[key] = val

    # EPS/BPS fallback to OFS
    if ('eps' not in result or 'bps' not in result) and items:
        ofs = _call('OFS')
        for item in ofs:
            acct_nm = (item.get('account_nm') or '').strip()
            for ko, key in TARGET_ACCOUNTS.items():
                if ko in acct_nm and key not in result:
                    val = _safe_float((item.get('thstrm_amount') or '').replace(',', ''))
                    if val is not None:
                        result[key] = val

    return result


REPRT_CODE_MAP = {
    '-03-31': '11013',
    '-06-30': '11012',
    '-09-30': '11014',
    '-12-31': '11011',
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='data/finrl_trading.db')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max tickers to process (0=all)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    dart_api_key = os.environ.get('DART_API_KEY', '')
    if not dart_api_key:
        print("ERROR: DART_API_KEY 환경변수 필요")
        sys.exit(1)

    db_path = os.path.join(project_root, args.db)

    # Load corp_codes
    print("DART corp_code 목록 로드 중...")
    corp_codes = get_dart_corp_codes(dart_api_key)

    # Find records needing refetch (revenue IS NULL)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        f"""SELECT ticker, datadate FROM krx_fundamental_data
            WHERE revenue IS NULL
              AND datadate >= ? AND datadate <= ?
            ORDER BY ticker, datadate""",
        (args.start, args.end)
    ).fetchall()
    conn.close()

    print(f"재수집 대상: {len(rows)}건")

    # Group by ticker for progress display
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for ticker, datadate in rows:
        by_ticker[ticker].append(datadate)

    tickers = list(by_ticker.keys())
    if args.limit > 0:
        tickers = tickers[:args.limit]

    updated = 0
    skipped = 0

    for ticker in tqdm(tickers, desc="DART Refetch"):
        corp_code = corp_codes.get(ticker, '')
        if not corp_code:
            skipped += 1
            continue

        for datadate in by_ticker[ticker]:
            # Determine year and reprt_code from datadate (e.g. 2023-03-31)
            year = datadate[:4]
            suffix = datadate[4:]  # e.g. -03-31
            reprt_code = REPRT_CODE_MAP.get(suffix)
            if not reprt_code:
                continue

            fs = fetch_fs(corp_code, year, reprt_code, dart_api_key)
            if not fs:
                continue

            # Update DB
            conn = sqlite3.connect(db_path)
            try:
                conn.execute("""
                    UPDATE krx_fundamental_data SET
                        revenue = ?,
                        gross_profit = ?,
                        operating_income = ?,
                        current_assets = ?,
                        current_liabilities = ?,
                        total_liabilities = ?,
                        net_income = ?,
                        total_equity = ?
                    WHERE ticker = ? AND datadate = ?
                """, (
                    fs.get('revenue'),
                    fs.get('gross_profit'),
                    fs.get('operating_income'),
                    fs.get('current_assets'),
                    fs.get('current_liabilities'),
                    fs.get('total_liabilities'),
                    fs.get('net_income'),
                    fs.get('total_equity'),
                    ticker,
                    datadate,
                ))
                conn.commit()
                updated += 1
            except Exception as e:
                logger.warning(f"DB update failed {ticker} {datadate}: {e}")
            finally:
                conn.close()

    print(f"\n완료: {updated}건 업데이트, {skipped}건 corp_code 없음 스킵")

    # Summary
    conn = sqlite3.connect(db_path)
    nn = conn.execute("SELECT COUNT(*) FROM krx_fundamental_data WHERE revenue IS NOT NULL").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM krx_fundamental_data").fetchone()[0]
    print(f"DB 상태: revenue non-null {nn}/{total} ({nn/total:.1%})")
    conn.close()


if __name__ == '__main__':
    main()
