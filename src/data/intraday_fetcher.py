"""
Intraday Data Fetcher Module
============================

15분봉 데이터 수집 및 저장 모듈.

제약사항:
- pykrx 분봉은 최근 수개월만 제공
- KIS API도 약 90일 한계
- 역사 백테스트는 일봉 OHLC 기반 시뮬레이션을 사용

사용법:
    fetcher = IntradayFetcher()
    df = fetcher.fetch_15min_data('005930', '2024-01-15')
    fetcher.collect_and_store(['005930', '000660'], '2024-01-01', '2024-12-31')
"""

import logging
import os
import sys
from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data.data_store import get_data_store

logger = logging.getLogger(__name__)


class IntradayFetcher:
    """15분봉 데이터 수집 및 DB 저장 클래스."""

    def __init__(self, interval_min: int = 15):
        self.interval_min = interval_min
        self.data_store = get_data_store()
        self._pykrx_available = self._check_pykrx()

    def _check_pykrx(self) -> bool:
        try:
            import pykrx  # noqa: F401
            return True
        except ImportError:
            logger.warning("pykrx not installed. pip install pykrx")
            return False

    def fetch_15min_data(self, ticker: str, date: str) -> pd.DataFrame:
        """
        특정 날짜의 15분봉 데이터를 반환한다.
        pykrx로 수집 실패 시 일봉 OHLC로 대체한다.

        Args:
            ticker: 종목 코드 (예: '005930')
            date: 날짜 'YYYY-MM-DD'

        Returns:
            DataFrame: columns=['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'interval_min']
        """
        # DB 캐시 확인
        start_dt = f"{date} 00:00:00"
        end_dt = f"{date} 23:59:59"
        cached = self.data_store.get_intraday_data([ticker], start_dt, end_dt, self.interval_min)
        if not cached.empty:
            return cached

        # pykrx 분봉 수집 시도
        if self._pykrx_available:
            df = self._fetch_from_pykrx(ticker, date)
            if not df.empty:
                self.data_store.save_intraday_data(df, self.interval_min)
                return df

        # fallback: 일봉 OHLC로 합성 (open-only, close-only 시뮬레이션용)
        logger.debug(f"Using daily OHLC fallback for {ticker} on {date}")
        return self._daily_ohlc_fallback(ticker, date)

    def _fetch_from_pykrx(self, ticker: str, date: str) -> pd.DataFrame:
        """pykrx로 분봉 데이터 수집."""
        try:
            from pykrx import stock as krx_stock
            date_fmt = date.replace('-', '')
            df_raw = krx_stock.get_market_ohlcv_by_minute(date_fmt, ticker)
            if df_raw is None or df_raw.empty:
                return pd.DataFrame()
            df_raw = df_raw.reset_index()
            date_col = df_raw.columns[0]
            col_map = {'시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume'}
            df_raw = df_raw.rename(columns=col_map)
            df_raw['ticker'] = ticker
            df_raw['datetime'] = pd.to_datetime(df_raw[date_col])
            df_raw['interval_min'] = self.interval_min

            # 지정된 interval로 리샘플링
            if self.interval_min != 1:
                df_raw = df_raw.set_index('datetime')
                rule = f'{self.interval_min}T'
                resampled = df_raw[['open', 'high', 'low', 'close', 'volume']].resample(rule).agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                }).dropna()
                resampled['ticker'] = ticker
                resampled['interval_min'] = self.interval_min
                resampled = resampled.reset_index()
                resampled = resampled.rename(columns={resampled.columns[0]: 'datetime'})
                return resampled[['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'interval_min']]

            return df_raw[['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'interval_min']]
        except Exception as e:
            logger.debug(f"pykrx intraday failed for {ticker} {date}: {e}")
            return pd.DataFrame()

    def _daily_ohlc_fallback(self, ticker: str, date: str) -> pd.DataFrame:
        """
        일봉 OHLC를 이용한 간이 intraday 시뮬레이션.
        장중 매수/매도 로직 테스트용.
        반환: 2개 행 (장시작=open, 장마감=close)
        """
        daily = self.data_store.get_price_data([ticker], date, date)
        if daily.empty:
            return pd.DataFrame()
        row = daily.iloc[0]
        open_dt = pd.to_datetime(f"{date} 09:00:00")
        close_dt = pd.to_datetime(f"{date} 15:15:00")
        records = [
            {'ticker': ticker, 'datetime': open_dt,
             'open': row['prcod'], 'high': row['prcod'], 'low': row['prcod'], 'close': row['prcod'],
             'volume': 0, 'interval_min': self.interval_min},
            {'ticker': ticker, 'datetime': close_dt,
             'open': row['prccd'], 'high': row['prchd'], 'low': row['prcld'], 'close': row['prccd'],
             'volume': row['cshtrd'], 'interval_min': self.interval_min},
        ]
        return pd.DataFrame(records)

    def collect_and_store(self, tickers: List[str],
                          start_date: str, end_date: str) -> int:
        """
        지정 기간의 분봉 데이터를 수집해서 DB에 저장한다.
        최근 데이터 우선, 오래된 날짜는 pykrx 한계로 수집 불가.

        Returns:
            int: 저장된 레코드 수
        """
        import pandas_market_calendars as mcal
        from src.data.trading_calendar import get_trading_days

        trading_days = get_trading_days(start_date, end_date, exchange='XKRX')
        total_saved = 0

        for ticker in tickers:
            logger.info(f"Collecting intraday data for {ticker} ({start_date} ~ {end_date})")
            for day in reversed(list(trading_days)):  # 최근 날짜 우선
                day_str = day.strftime('%Y-%m-%d')
                df = self.fetch_15min_data(ticker, day_str)
                if not df.empty:
                    saved = self.data_store.save_intraday_data(df, self.interval_min)
                    total_saved += saved

        logger.info(f"Intraday collection complete. Total records: {total_saved}")
        return total_saved


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Intraday Fetcher')
    parser.add_argument('--tickers', nargs='+', default=['005930'], help='종목 코드 목록')
    parser.add_argument('--start', default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--interval', type=int, default=15)
    args = parser.parse_args()

    fetcher = IntradayFetcher(interval_min=args.interval)
    saved = fetcher.collect_and_store(args.tickers, args.start, args.end)
    print(f"Saved {saved} intraday records")
