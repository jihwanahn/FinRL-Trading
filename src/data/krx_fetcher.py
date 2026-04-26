"""
KRX Data Fetcher Module
=======================

KIS API(한국투자증권) + FinanceDataReader(FDR) 기반 KRX 데이터 페처.
우선순위: KIS API > FDR > pykrx(optional)

주요 메서드:
    get_kospi200_components(date)  - KOSPI200 구성 종목
    get_price_data(tickers, start, end)  - 일봉 OHLCV
    get_fundamental_data(tickers, start, end)  - PER/PBR/EPS/BPS
    get_intraday_data(ticker, date, interval=15)  - 분봉

환경변수 (KIS API):
    KIS_APP_KEY, KIS_APP_SECRET
    KIS_BASE_URL (기본: https://openapi.koreainvestment.com:9443 실전)
    KIS_PAPER_TRADING=true 이면 모의 URL 자동 전환
"""

import logging
import os
import sys
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.data.data_fetcher import BaseDataFetcher, DataSource
from src.data.data_store import get_data_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KIS API 클라이언트 (데이터 조회 전용)
# ---------------------------------------------------------------------------

class KISDataClient:
    """
    KIS OpenAPI 데이터 조회 클라이언트.
    주문 실행은 KISManager를 사용하고, 여기서는 시세/재무 조회만 담당한다.

    환경변수:
        KIS_APP_KEY      - 앱 키
        KIS_APP_SECRET   - 앱 시크릿
        KIS_BASE_URL     - API URL (기본: 실전투자)
    """

    REAL_URL   = "https://openapi.koreainvestment.com:9443"
    PAPER_URL  = "https://openapivts.koreainvestment.com:9443"

    # API rate limit: 초당 ~20건 (안전하게 20건/초 = 0.05초 간격)
    _MIN_INTERVAL = 0.05

    def __init__(self):
        self.app_key    = os.environ.get("KIS_APP_KEY", "")
        self.app_secret = os.environ.get("KIS_APP_SECRET", "")
        paper = os.environ.get("KIS_PAPER_TRADING", "false").lower() == "true"
        self.base_url   = os.environ.get(
            "KIS_BASE_URL",
            self.PAPER_URL if paper else self.REAL_URL
        )
        self._token: str = ""
        self._token_expires_at: Optional[datetime] = None
        self._last_request_at: float = 0.0

    @property
    def available(self) -> bool:
        return bool(self.app_key and self.app_secret)

    def _ensure_token(self) -> None:
        """OAuth2 토큰 갱신 (24시간 만료)."""
        now = datetime.utcnow()
        if self._token and self._token_expires_at and now < self._token_expires_at - timedelta(minutes=5):
            return
        resp = requests.post(
            f"{self.base_url}/oauth2/tokenP",
            json={"grant_type": "client_credentials",
                  "appkey": self.app_key, "appsecret": self.app_secret},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data.get("access_token", "")
        expires_in = int(data.get("expires_in", 86400))
        self._token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        logger.info("KIS token refreshed")

    def _get(self, path: str, params: Dict, tr_id: str) -> Dict:
        """KIS API GET 요청 (rate limit 적용)."""
        self._ensure_token()
        # rate limit
        elapsed = time.time() - self._last_request_at
        if elapsed < self._MIN_INTERVAL:
            time.sleep(self._MIN_INTERVAL - elapsed)
        resp = requests.get(
            f"{self.base_url}{path}",
            headers={
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self._token}",
                "appkey": self.app_key,
                "appsecret": self.app_secret,
                "tr_id": tr_id,
                "custtype": "P",
            },
            params=params,
            timeout=20,
        )
        self._last_request_at = time.time()
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # 일별 시세 (기간별, 최대 100건/요청)
    # TR_ID: FHKST03010100
    # ------------------------------------------------------------------

    def fetch_daily_price(self, ticker: str, start_date: str, end_date: str) -> List[Dict]:
        """
        종목 일별 시세 조회 (10년치도 자동 페이지네이션).
        start_date, end_date: 'YYYY-MM-DD'

        Returns:
            List of dicts: {date, open, high, low, close, volume, adj_close}
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        all_rows: List[Dict] = []

        current_end = pd.to_datetime(end_date)
        start_dt    = pd.to_datetime(start_date)

        while current_end >= start_dt:
            params = {
                "FID_COND_MRKT_DIV_CODE": "J",       # 주식
                "FID_INPUT_ISCD": ticker,
                "FID_INPUT_DATE_1": start_dt.strftime("%Y%m%d"),
                "FID_INPUT_DATE_2": current_end.strftime("%Y%m%d"),
                "FID_PERIOD_DIV_CODE": "D",           # 일별
                "FID_ORG_ADJ_PRC": "1",               # 수정주가 적용
            }
            try:
                data = self._get(path, params, tr_id="FHKST03010100")
            except Exception as e:
                logger.warning(f"KIS daily price failed for {ticker}: {e}")
                break

            output2 = data.get("output2", [])
            if not output2:
                break

            for item in output2:
                date_str = item.get("stck_bsop_date", "")
                if not date_str:
                    continue
                close = _safe_float(item.get("stck_clpr"))
                if close is None:
                    continue
                all_rows.append({
                    "date":      f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
                    "open":      _safe_float(item.get("stck_oprc")) or close,
                    "high":      _safe_float(item.get("stck_hgpr")) or close,
                    "low":       _safe_float(item.get("stck_lwpr")) or close,
                    "close":     close,
                    "volume":    _safe_float(item.get("acml_vol")) or 0.0,
                    "adj_close": close,  # FID_ORG_ADJ_PRC=1 이므로 이미 수정주가
                })

            # 100건씩 반환 → 마지막 날짜 전날로 이동하여 계속
            oldest = min(output2, key=lambda x: x.get("stck_bsop_date", "99999999"))
            oldest_date = pd.to_datetime(oldest.get("stck_bsop_date", ""))
            if pd.isnull(oldest_date) or oldest_date <= start_dt:
                break
            current_end = oldest_date - timedelta(days=1)

        # 중복 제거 및 정렬
        if all_rows:
            df = pd.DataFrame(all_rows).drop_duplicates("date").sort_values("date")
            all_rows = df.to_dict("records")
        return all_rows

    # ------------------------------------------------------------------
    # KOSPI200 구성 종목 (시가총액 상위 200)
    # TR_ID: FHPUP02100000
    # ------------------------------------------------------------------

    def fetch_kospi200_components(self) -> List[Dict]:
        """
        KOSPI200 구성 종목 조회.
        KIS API: 국내주식 업종/지수 구성 종목 (FHPUP02100000)

        Returns:
            List of dicts: {ticker, name}
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-index-components"
        try:
            data = self._get(path, {"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": "0028"},
                             tr_id="FHPUP02100000")
            items = data.get("output2", [])
            result = []
            for item in items:
                code = item.get("stck_shrn_iscd", "") or item.get("iscd", "")
                if code:
                    result.append({"ticker": code.zfill(6), "name": item.get("hts_kor_isnm", "")})
            if result:
                logger.info(f"KIS API: {len(result)} KOSPI200 구성 종목 조회")
            return result
        except Exception as e:
            logger.debug(f"KIS KOSPI200 조회 실패: {e}")
            return []

    # ------------------------------------------------------------------
    # 현재가 + PER/PBR (단일 종목 스냅샷)
    # TR_ID: FHKST01010100
    # ------------------------------------------------------------------

    def fetch_current_fundamental(self, ticker: str) -> Dict:
        """
        현재 PER/PBR/EPS/BPS 조회 (스냅샷, 현재 시점만).

        Returns:
            dict: {per, pbr, eps, bps, div_yield}
        """
        path = "/uapi/domestic-stock/v1/quotations/inquire-price"
        try:
            data = self._get(path,
                             {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": ticker},
                             tr_id="FHKST01010100")
            out = data.get("output", {})
            return {
                "per":       _safe_float(out.get("per")),
                "pbr":       _safe_float(out.get("pbr")),
                "eps":       _safe_float(out.get("eps")),
                "bps":       _safe_float(out.get("bps")),
                "div_yield": _safe_float(out.get("divi_rate")),
            }
        except Exception as e:
            logger.debug(f"KIS fundamental failed for {ticker}: {e}")
            return {}


class KRXFetcher(BaseDataFetcher, DataSource):
    """KIS API + FDR 기반 KRX 데이터 페처."""

    def __init__(self, cache_dir: str = None):
        super().__init__(cache_dir)
        self._fdr_available = self._check_fdr()
        self._pykrx_available = self._check_pykrx()
        self._kis = KISDataClient()
        if self._kis.available:
            logger.info("KIS API 사용 가능 — 가격/구성 종목 조회에 KIS API 우선 사용")
        else:
            logger.info("KIS API 키 없음 — FDR fallback 사용")

    def _check_fdr(self) -> bool:
        try:
            import FinanceDataReader  # noqa: F401
            return True
        except ImportError:
            logger.warning("FinanceDataReader not installed. Run: pip install finance-datareader")
            return False

    def _check_pykrx(self) -> bool:
        try:
            from pykrx import stock as krx_stock  # noqa: F401
            # 실제 API 호출 없이 import만 확인
            return True
        except Exception:
            return False

    def is_available(self) -> bool:
        return self._fdr_available

    # ------------------------------------------------------------------
    # 인덱스 구성 종목
    # ------------------------------------------------------------------

    def get_kospi200_components(self, date: str = None, top_n: int = 200) -> pd.DataFrame:
        """
        KOSPI200 구성 종목 반환.

        우선순위:
          1. DB 캐시 (이전에 저장된 스냅샷)
          2. KRX 공공데이터 API (인덱스 구성 종목 조회)
          3. FDR StockListing('KOSPI') 시가총액 상위 top_n개 (근사값)

        date 파라미터: DB 캐시 키 (실제 과거 구성 종목 재현은 부정확할 수 있음)

        Returns:
            DataFrame: columns=['tickers', 'sectors', 'dateFirstAdded']
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # 1) DB 캐시
        tickers_str, sectors_str = self.data_store.get_kospi200_components(date)
        if tickers_str:
            tickers_list = tickers_str.split(',')
            sectors_list = sectors_str.split(',') if sectors_str else [''] * len(tickers_list)
            return pd.DataFrame({
                'tickers': tickers_list,
                'sectors': sectors_list,
                'dateFirstAdded': [''] * len(tickers_list),
            })

        if not self._fdr_available and not self._kis.available:
            logger.error("FDR과 KIS API 모두 사용 불가")
            return pd.DataFrame({'tickers': [], 'sectors': [], 'dateFirstAdded': []})

        # 2) KIS API (구성 종목 직접 조회)
        tickers, sectors = [], []
        if self._kis.available:
            items = self._kis.fetch_kospi200_components()
            if items:
                tickers = [i["ticker"] for i in items]
                sectors = [""] * len(tickers)

        # 3) KRX 공공데이터 포털 API
        if not tickers:
            tickers, sectors = self._fetch_kospi200_from_krx_api(date)

        # 3) FDR 시총 상위 top_n 근사
        if not tickers:
            tickers, sectors = self._fetch_kospi200_from_fdr_marcap(top_n)

        if not tickers:
            logger.error("KOSPI200 구성 종목 조회 실패 (모든 방법 실패)")
            return pd.DataFrame({'tickers': [], 'sectors': [], 'dateFirstAdded': []})

        result = pd.DataFrame({
            'tickers': tickers,
            'sectors': sectors,
            'dateFirstAdded': [''] * len(tickers),
        })

        # DB 저장
        self.data_store.save_kospi200_components(
            date,
            ','.join(tickers),
            ','.join(sectors),
        )
        logger.info(f"Fetched {len(tickers)} KOSPI200 components for {date}")
        return result

    def _fetch_kospi200_from_krx_api(self, date: str) -> tuple:
        """KRX 공공데이터 API에서 KOSPI200 구성 종목을 조회한다."""
        try:
            import requests
            # KRX 인덱스 구성 종목 조회 API
            date_fmt = date.replace('-', '')
            url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
            payload = {
                'bld': 'dbms/MDC/STAT/standard/MDCSTAT00601',
                'indIdx': '1',
                'indIdx2': '028',  # KOSPI200
                'trdDd': date_fmt,
                'money': '1',
                'csvxls_isNo': 'false',
            }
            headers = {
                'Referer': 'http://data.krx.co.kr/',
                'User-Agent': 'Mozilla/5.0',
            }
            resp = requests.post(url, data=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            items = data.get('OutBlock_1', [])
            if not items:
                return [], []
            tickers = [str(item.get('ISU_SRT_CD', '')).zfill(6) for item in items]
            sectors = [str(item.get('IDX_IND_NM', '')) for item in items]
            tickers = [t for t in tickers if t and t != '000000']
            logger.info(f"KRX API: {len(tickers)} KOSPI200 components for {date}")
            return tickers, sectors[:len(tickers)]
        except Exception as e:
            logger.debug(f"KRX API 조회 실패: {e}")
            return [], []

    def _fetch_kospi200_from_fdr_marcap(self, top_n: int = 200) -> tuple:
        """FDR StockListing('KOSPI') 시가총액 상위 top_n개로 KOSPI200을 근사한다."""
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing('KOSPI')
            if df is None or df.empty:
                return [], []

            code_col = next((c for c in ['Code', 'ISU_SRT_CD', 'Symbol'] if c in df.columns), None)
            if code_col is None:
                return [], []

            df[code_col] = df[code_col].astype(str).str.zfill(6)

            # 시가총액 기준 정렬
            if 'Marcap' in df.columns:
                df = df.sort_values('Marcap', ascending=False)
            df = df.head(top_n)

            tickers = df[code_col].tolist()
            sectors = [''] * len(tickers)
            logger.info(f"FDR 시총 상위 {len(tickers)}개로 KOSPI200 근사")
            return tickers, sectors
        except Exception as e:
            logger.debug(f"FDR marcap 조회 실패: {e}")
            return [], []

    def _get_sectors_fdr(self) -> Dict[str, str]:
        """FDR StockListing('KOSPI')에서 종목별 섹터 정보를 반환한다."""
        if not self._fdr_available:
            return {}
        try:
            import FinanceDataReader as fdr
            df = fdr.StockListing('KOSPI')
            if df is None or df.empty:
                return {}
            code_col = next((c for c in ['Code', 'Symbol', 'ISU_SRT_CD'] if c in df.columns), None)
            sect_col = next((c for c in ['Sector', 'Industry', '섹터', '업종'] if c in df.columns), None)
            if code_col and sect_col:
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                return dict(zip(df[code_col], df[sect_col].fillna('')))
        except Exception as e:
            logger.debug(f"Sector fetch failed: {e}")
        return {}

    # ------------------------------------------------------------------
    # 일봉 가격 데이터
    # ------------------------------------------------------------------

    def get_price_data(self, tickers: pd.DataFrame,
                       start_date: str, end_date: str) -> pd.DataFrame:
        """KOSPI/KOSDAQ 일봉 OHLCV 데이터 반환. FDR 사용."""
        if isinstance(tickers, pd.DataFrame):
            tickers_list = tickers['tickers'].astype(str).tolist()
        else:
            tickers_list = list(tickers)

        all_new_records: List[Dict[str, Any]] = []
        for ticker in tqdm(tickers_list, desc="Fetching KRX price data"):
            try:
                rows = self._fetch_price_one_ticker(ticker, start_date, end_date)
                all_new_records.extend(rows)
            except Exception as e:
                logger.warning(f"Failed price fetch for {ticker}: {e}")

        if all_new_records:
            df_new = pd.DataFrame(all_new_records)
            df_new = self._standardize_price_data(df_new)
            self.data_store.save_price_data(df_new)

        return self.data_store.get_price_data(tickers_list, start_date, end_date)

    def _fetch_price_one_ticker(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """단일 종목 일봉 수집 (KIS API 우선 → FDR → pykrx)."""
        rows = []

        # KIS API (1순위)
        if self._kis.available:
            try:
                kis_rows = self._kis.fetch_daily_price(ticker, start_date, end_date)
                for r in kis_rows:
                    rows.append({
                        'gvkey':     ticker,
                        'datadate':  r['date'],
                        'tic':       ticker,
                        'prccd':     r['close'],
                        'prcod':     r['open'],
                        'prchd':     r['high'],
                        'prcld':     r['low'],
                        'cshtrd':    r['volume'],
                        'adj_close': r['adj_close'],
                    })
                if rows:
                    return rows
            except Exception as e:
                logger.debug(f"KIS price failed for {ticker}: {e}")

        # FDR (2순위)
        if self._fdr_available:
            try:
                import FinanceDataReader as fdr
                df = fdr.DataReader(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    date_col = next((c for c in ['Date', 'date', '날짜'] if c in df.columns), df.columns[0])
                    for _, row in df.iterrows():
                        close = _safe_float(row.get('Close', row.get('close')))
                        if close is None:
                            continue
                        rows.append({
                            'gvkey': ticker,
                            'datadate': str(row[date_col])[:10],
                            'tic': ticker,
                            'prccd': close,
                            'prcod': _safe_float(row.get('Open', row.get('open'))) or close,
                            'prchd': _safe_float(row.get('High', row.get('high'))) or close,
                            'prcld': _safe_float(row.get('Low', row.get('low'))) or close,
                            'cshtrd': _safe_float(row.get('Volume', row.get('volume'))) or 0.0,
                            'adj_close': _safe_float(row.get('Adj Close', row.get('adj_close'))) or close,
                        })
                    if rows:
                        return rows
            except Exception as e:
                logger.debug(f"FDR failed for {ticker}: {e}")

        # pykrx fallback (선택적)
        if self._pykrx_available:
            try:
                from pykrx import stock as krx_stock
                start_fmt = start_date.replace('-', '')
                end_fmt = end_date.replace('-', '')
                df = krx_stock.get_market_ohlcv_by_date(start_fmt, end_fmt, ticker)
                if df is not None and not df.empty:
                    df = df.reset_index()
                    date_col = next((c for c in ['날짜', 'Date'] if c in df.columns), df.columns[0])
                    col_map = {'시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume'}
                    df = df.rename(columns=col_map)
                    for _, row in df.iterrows():
                        close = _safe_float(row.get('close')) or 0.0
                        rows.append({
                            'gvkey': ticker,
                            'datadate': str(row[date_col])[:10],
                            'tic': ticker,
                            'prccd': close,
                            'prcod': _safe_float(row.get('open')) or close,
                            'prchd': _safe_float(row.get('high')) or close,
                            'prcld': _safe_float(row.get('low')) or close,
                            'cshtrd': _safe_float(row.get('volume')) or 0.0,
                            'adj_close': close,
                        })
            except Exception as e:
                logger.debug(f"pykrx failed for {ticker}: {e}")

        return rows

    # ------------------------------------------------------------------
    # 재무 데이터 (PER/PBR/EPS/BPS/DPS)
    # ------------------------------------------------------------------

    def get_fundamental_data(self, tickers: pd.DataFrame,
                             start_date: str, end_date: str,
                             align_quarter_dates: bool = False) -> pd.DataFrame:
        """
        KRX 재무 데이터 (PER, PBR, EPS, BPS, DPS, DIV_YIELD) 반환.
        우선순위: KRX 공공데이터 포털 → pykrx(deprecated) → KIS 스냅샷
        """
        if isinstance(tickers, pd.DataFrame):
            tickers_list = tickers['tickers'].astype(str).tolist()
        else:
            tickers_list = list(tickers)

        all_records: List[Dict[str, Any]] = []
        for ticker in tqdm(tickers_list, desc="Fetching KRX fundamentals"):
            try:
                records = self._fetch_fundamental_one_ticker(ticker, start_date, end_date)
                if records:
                    # 종목별로 즉시 저장 (프로세스 중단 시 데이터 보존)
                    self.data_store.save_krx_fundamental_data(pd.DataFrame(records))
                    all_records.extend(records)
            except Exception as e:
                logger.debug(f"Failed fundamental fetch for {ticker}: {e}")

        if all_records:
            logger.info(f"Saved {len(all_records)} fundamental records total")

        return self.data_store.get_krx_fundamental_data(tickers_list, start_date, end_date)

    def _fetch_fundamental_one_ticker(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        단일 종목 일별 PER/PBR/EPS/BPS/DPS 수집.
        1) KRX 공공데이터 포털 (data.krx.co.kr) — 가장 안정적
        2) pykrx fallback (현재 broken, 에러 시 조용히 skip)
        3) KIS 현재가 스냅샷 (역사 데이터 없음, 최근 시점만)
        """
        # 1) KRX 공공데이터 포털
        records = self._fetch_fundamental_krx_portal(ticker, start_date, end_date)
        if records:
            return records

        # 2) pykrx (logging 오류 억제)
        records = self._fetch_fundamental_pykrx(ticker, start_date, end_date)
        if records:
            return records

        # 3) KIS 현재 스냅샷 (역사 없음 — 오늘 날짜 1건만)
        if self._kis.available:
            snap = self._kis.fetch_current_fundamental(ticker)
            if snap and snap.get('per') is not None:
                today = datetime.now().strftime('%Y-%m-%d')
                return [{'ticker': ticker, 'datadate': today, **snap, 'dps': None}]

        return []

    def _fetch_fundamental_krx_portal(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        DART OpenAPI HTTP 직접 호출로 분기 재무 데이터 조회.
        환경변수 DART_API_KEY 필요 (https://opendart.fss.or.kr/).
        데이터: 분기별 EPS, BPS, ROE → PER/PBR은 가격으로 계산
        """
        records = []
        dart_api_key = os.environ.get("DART_API_KEY", "")
        if not dart_api_key:
            return records

        # 종목코드 → DART corp_code
        corp_code = self._get_dart_corp_code_http(ticker, dart_api_key)
        if not corp_code:
            logger.debug(f"DART corp_code not found for {ticker}")
            return records

        # 해당 기간의 분기 말 날짜 목록
        start_yr = int(start_date[:4])
        end_yr   = int(end_date[:4])

        # 가격 데이터 (PER/PBR 계산용) — 분기 말 월 기준 마지막 종가
        price_df = self.data_store.get_price_data([ticker], start_date, end_date)
        price_map: Dict[str, float] = {}
        if not price_df.empty and 'datadate' in price_df.columns:
            tmp = price_df.copy()
            tmp['ym'] = tmp['datadate'].astype(str).str[:7]
            price_map = (tmp.sort_values('datadate')
                            .groupby('ym')['prccd'].last()
                            .astype(float).to_dict())

        # 이미 DB에 있는 날짜는 스킵 (재실행 시 중복 DART 호출 방지)
        # refetch=True이면 모든 날짜 재수집 (새 컬럼 backfill용)
        refetch = getattr(self, '_refetch_fundamentals', False)
        existing_dates: set = set()
        if not refetch:
            existing_df = self.data_store.get_krx_fundamental_data([ticker], start_date, end_date)
            if existing_df is not None and not existing_df.empty and 'datadate' in existing_df.columns:
                # 새 컬럼(revenue 등)이 NULL인 레코드는 재수집 대상
                has_new_cols = 'revenue' in existing_df.columns
                if has_new_cols:
                    filled = existing_df[existing_df['revenue'].notna()]['datadate'].astype(str).tolist()
                    existing_dates = set(filled)
                else:
                    existing_dates = set(existing_df['datadate'].astype(str).tolist())

        # reprt_code: 11013=1Q, 11012=반기, 11014=3Q, 11011=사업(연간)
        quarters = [
            ('11013', '-03-31'), ('11012', '-06-30'),
            ('11014', '-09-30'), ('11011', '-12-31'),
        ]
        for year in range(start_yr, end_yr + 1):
            for reprt_code, suffix in quarters:
                qtr_end = f"{year}{suffix}"
                if qtr_end < start_date or qtr_end > end_date:
                    continue
                if qtr_end in existing_dates:
                    continue  # 이미 저장됨
                fs_data = self._dart_fetch_fs(corp_code, str(year), reprt_code, dart_api_key)
                if not fs_data:
                    continue

                eps = fs_data.get('eps')
                bps = fs_data.get('bps')
                net_income   = fs_data.get('net_income')
                total_equity = fs_data.get('total_equity')

                # BPS fallback: total_equity / shares_outstanding
                if bps is None and total_equity and total_equity > 0:
                    shares = self._dart_fetch_shares(corp_code, str(year), reprt_code, dart_api_key)
                    if shares and shares > 0:
                        bps = total_equity / shares

                # ROE 계산
                roe = None
                if net_income is not None and total_equity and total_equity > 0:
                    roe = net_income / total_equity

                # PER/PBR 계산 (분기 말 주가 기준)
                ym = qtr_end[:7]
                price = price_map.get(ym)
                per = (price / eps) if (price and eps and eps > 0) else None
                pbr = (price / bps) if (price and bps and bps > 0) else None

                records.append({
                    'ticker':              ticker,
                    'datadate':            qtr_end,
                    'per':                 per,
                    'pbr':                 pbr,
                    'eps':                 eps,
                    'bps':                 bps,
                    'dps':                 fs_data.get('dps'),
                    'div_yield':           None,
                    'roe':                 roe,
                    'revenue':             fs_data.get('revenue'),
                    'gross_profit':        fs_data.get('gross_profit'),
                    'operating_income':    fs_data.get('operating_income'),
                    'current_assets':      fs_data.get('current_assets'),
                    'current_liabilities': fs_data.get('current_liabilities'),
                    'total_liabilities':   fs_data.get('total_liabilities'),
                    'net_income':          fs_data.get('net_income'),
                    'total_equity':        total_equity,
                })

        return records

    def _dart_fetch_fs(self, corp_code: str, bsns_year: str,
                       reprt_code: str, api_key: str) -> Dict:
        """DART API: 단일회사 전체 재무제표에서 핵심 지표 추출.
        CFS(연결) 우선, EPS/BPS 미발견 시 OFS(개별)로 재시도."""
        url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"

        def _fetch_items(fs_div: str):
            params = {
                'crtfc_key': api_key,
                'corp_code': corp_code,
                'bsns_year': bsns_year,
                'reprt_code': reprt_code,
                'fs_div': fs_div,
            }
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                if data.get('status') != '000':
                    return []
                return data.get('list', [])
            except Exception as e:
                logger.debug(f"DART API error ({fs_div}): {e}")
                return []

        items = _fetch_items('CFS')  # 연결재무제표 우선
        if not items:
            items = _fetch_items('OFS')  # 개별재무제표 fallback

        # 계정명 → 값 매핑 (연결재무제표 기준)
        # 실제 DART 응답 계정명:
        #   EPS: '기본주당이익(손실)'  (substring: '기본주당이익')
        #   BPS: '주당순자산' 또는 '1주당순자산' 또는 '주당순자산가치'
        #   DPS: '주당배당금' 또는 '주당현금배당금'
        #   Net income: '당기순이익'
        #   Total equity: '자본총계'
        target_accounts = {
            '기본주당이익':     'eps',   # matches '기본주당이익(손실)'
            '희석주당이익':     'eps',   # fallback: '희석주당이익(손실)'
            '주당순자산':      'bps',   # most common BPS label
            '1주당순자산':     'bps',   # alternative label
            '주당장부가치':     'bps',   # another variant
            '주당배당금':      'dps',
            '주당현금배당금':   'dps',
            '당기순이익':      'net_income',
            '자본총계':       'total_equity',
            # --- 손익계산서 (Income Statement) ---
            '매출액':          'revenue',        # 총매출
            '수익(매출액)':    'revenue',        # 금융업 표기
            '영업수익':        'revenue',        # 금융/지주회사 영업수익
            '매출총이익':      'gross_profit',   # 매출액 - 매출원가
            '영업이익':        'operating_income',
            '영업이익(손실)':  'operating_income',
            # --- 재무상태표 (Balance Sheet) ---
            '유동자산':        'current_assets',
            '유동부채':        'current_liabilities',
            '부채총계':        'total_liabilities',
        }

        def _parse_items(item_list, result_dict):
            for item in item_list:
                acct_nm = (item.get('account_nm') or '').strip()
                for ko, key in target_accounts.items():
                    if ko in acct_nm and key not in result_dict:
                        raw = (item.get('thstrm_amount') or '').replace(',', '')
                        val = _safe_float(raw)
                        if val is not None:
                            result_dict[key] = val

        result: Dict[str, Optional[float]] = {}
        _parse_items(items, result)

        # EPS/BPS가 없으면 OFS(개별재무제표)에서 추가 시도
        if ('eps' not in result or 'bps' not in result) and items:
            ofs_items = _fetch_items('OFS')
            if ofs_items:
                _parse_items(ofs_items, result)

        return result

    def _dart_fetch_shares(self, corp_code: str, bsns_year: str,
                           reprt_code: str, api_key: str) -> Optional[float]:
        """DART API: 발행주식총수 조회 (주당순자산 계산용)."""
        cache_key = f"shares_{corp_code}_{bsns_year}_{reprt_code}"
        if not hasattr(self, '_shares_cache'):
            self._shares_cache: Dict[str, Optional[float]] = {}
        if cache_key in self._shares_cache:
            return self._shares_cache[cache_key]

        url = "https://opendart.fss.or.kr/api/stockTotqySttus.json"
        params = {
            'crtfc_key': api_key,
            'corp_code': corp_code,
            'bsns_year': bsns_year,
            'reprt_code': reprt_code,
        }
        shares = None
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get('status') == '000':
                for item in data.get('list', []):
                    # '보통주' 또는 '합계' 행에서 발행주식총수 가져오기
                    stkrms = item.get('stkrms', '') or ''  # 주식의 종류
                    totqy = (item.get('istc_totqy') or '').replace(',', '')
                    val = _safe_float(totqy)
                    if val and val > 0:
                        if '보통주' in stkrms or '합계' in stkrms or not stkrms:
                            shares = val
                            if '보통주' in stkrms:
                                break  # 보통주 우선
        except Exception as e:
            logger.debug(f"DART shares fetch failed for {corp_code}: {e}")

        self._shares_cache[cache_key] = shares
        return shares

    def _get_dart_corp_code_http(self, ticker: str, api_key: str) -> str:
        """DART API: 종목코드 → corp_code 변환 (전체 회사 목록 ZIP 다운로드)."""
        if hasattr(self, '_dart_corp_cache'):
            return self._dart_corp_cache.get(ticker, "")

        self._dart_corp_cache: Dict[str, str] = {}
        try:
            import zipfile, io, xml.etree.ElementTree as ET
            url = f"https://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                xml_data = z.read(z.namelist()[0])
            root = ET.fromstring(xml_data)
            for corp in root.findall('list'):
                stock_code = (corp.findtext('stock_code') or '').strip()
                corp_code  = (corp.findtext('corp_code') or '').strip()
                if stock_code and len(stock_code) == 6:
                    self._dart_corp_cache[stock_code] = corp_code
            logger.info(f"DART corp_code 캐시 로드: {len(self._dart_corp_cache)}개 회사")
        except Exception as e:
            logger.warning(f"DART corp_code 목록 로드 실패: {e}")
        return self._dart_corp_cache.get(ticker, "")

    def _fetch_fundamental_pykrx(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """pykrx fallback — 1.2.7에서 broken. root logger까지 억제하고 조용히 시도."""
        if not self._pykrx_available:
            return []
        records = []
        try:
            import logging as _logging
            import warnings
            # pykrx는 root logging.info()를 직접 호출해 logging error를 유발함
            # → root logger 핸들러를 임시 제거해 완전히 억제
            root_logger = _logging.getLogger()
            saved_handlers = root_logger.handlers[:]
            root_logger.handlers = []
            saved_level = root_logger.level
            root_logger.setLevel(_logging.CRITICAL)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from pykrx import stock as krx_stock
                    start_fmt = start_date.replace('-', '')
                    end_fmt   = end_date.replace('-', '')
                    df = krx_stock.get_market_fundamental_by_date(start_fmt, end_fmt, ticker, freq='m')
            finally:
                root_logger.handlers = saved_handlers
                root_logger.setLevel(saved_level)

            if df is None or df.empty:
                return records
            df = df.reset_index()
            date_col = next((c for c in ['날짜', 'Date'] if c in df.columns), df.columns[0])
            for _, row in df.iterrows():
                records.append({
                    'ticker':    ticker,
                    'datadate':  str(row[date_col])[:10],
                    'per':       _safe_float(row.get('PER')),
                    'pbr':       _safe_float(row.get('PBR')),
                    'eps':       _safe_float(row.get('EPS')),
                    'bps':       _safe_float(row.get('BPS')),
                    'dps':       _safe_float(row.get('DPS')),
                    'div_yield': _safe_float(row.get('DIV')),
                })
        except Exception:
            pass
        return records

    def _get_isin(self, ticker: str) -> str:
        """종목 코드 → ISIN 코드 변환 (FDR StockListing 사용)."""
        if not self._fdr_available:
            return ""
        try:
            import FinanceDataReader as fdr
            # 캐시: 전체 KOSPI 리스트는 한 번만 조회
            if not hasattr(self, '_isin_cache'):
                df = fdr.StockListing('KOSPI')
                if df is None or df.empty:
                    self._isin_cache = {}
                    return ""
                code_col = next((c for c in ['Code', 'ISU_SRT_CD'] if c in df.columns), None)
                isin_col = next((c for c in ['ISU_CD', 'ISIN'] if c in df.columns), None)
                if code_col and isin_col:
                    df[code_col] = df[code_col].astype(str).str.zfill(6)
                    self._isin_cache = dict(zip(df[code_col], df[isin_col]))
                else:
                    self._isin_cache = {}
            return self._isin_cache.get(ticker, "")
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # 분봉 데이터 (pykrx optional)
    # ------------------------------------------------------------------

    def get_intraday_data(self, ticker: str, date: str, interval: int = 15) -> pd.DataFrame:
        """분봉 데이터 반환. pykrx 의존, 최근 데이터만 지원."""
        if not self._pykrx_available:
            logger.warning("분봉 데이터는 pykrx가 필요합니다.")
            return pd.DataFrame()
        try:
            from pykrx import stock as krx_stock
            date_fmt = date.replace('-', '')
            df = krx_stock.get_market_ohlcv_by_minute(date_fmt, ticker)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.reset_index()
            date_col = df.columns[0]
            col_map = {'시가': 'open', '고가': 'high', '저가': 'low', '종가': 'close', '거래량': 'volume'}
            df = df.rename(columns=col_map)
            df['ticker'] = ticker
            df['datetime'] = pd.to_datetime(df[date_col])
            df['interval_min'] = interval
            return df[['ticker', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'interval_min']]
        except Exception as e:
            logger.warning(f"Intraday fetch failed for {ticker} on {date}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Protocol stubs
    # ------------------------------------------------------------------

    def get_sp500_components(self, date: str = None) -> pd.DataFrame:
        return self.get_kospi200_components(date)

    def get_news(self, ticker: str, from_date: str, to_date: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("News not supported for KRXFetcher")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if not (f != f) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description='KRX Data Fetcher (FDR 기반)')
    parser.add_argument('--fetch-kospi200', action='store_true',
                        help='KOSPI200 구성 종목 + 가격 데이터 수집')
    parser.add_argument('--fetch-price', metavar='TICKER',
                        help='단일 종목 가격 데이터 수집 (예: 005930)')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--refetch-fundamentals', action='store_true',
                        help='기존 fundamental 데이터도 재수집 (새 컬럼 backfill용)')
    args = parser.parse_args()

    fetcher = KRXFetcher()
    if getattr(args, 'refetch_fundamentals', False):
        fetcher._refetch_fundamentals = True
        print("[refetch] 기존 fundamental 데이터 포함하여 재수집합니다.")
    print(f"FDR available: {fetcher._fdr_available}")
    print(f"pykrx available: {fetcher._pykrx_available}")

    if args.fetch_price:
        rows = fetcher._fetch_price_one_ticker(args.fetch_price, args.start, args.end)
        print(f"Fetched {len(rows)} rows for {args.fetch_price}")
        if rows:
            print(pd.DataFrame(rows[:5]))

    elif args.fetch_kospi200:
        # 분기 스냅샷 수집
        quarters = pd.date_range(args.start, args.end, freq='QS')
        all_tickers: set = set()
        for q in quarters:
            q_str = q.strftime('%Y-%m-%d')
            df = fetcher.get_kospi200_components(q_str)
            all_tickers.update(df['tickers'].tolist())
            print(f"  {q_str}: {len(df)} components")

        print(f"\nTotal unique tickers: {len(all_tickers)}")

        if all_tickers:
            tickers_df = pd.DataFrame({'tickers': list(all_tickers)})
            print(f"\nFetching price data for {len(all_tickers)} tickers...")
            price_df = fetcher.get_price_data(tickers_df, args.start, args.end)
            print(f"Fetched {len(price_df)} price records")

            print(f"\nFetching fundamental data (DART API)...")
            fund_df = fetcher.get_fundamental_data(tickers_df, args.start, args.end)
            print(f"Fetched {len(fund_df)} fundamental records")
    else:
        parser.print_help()
