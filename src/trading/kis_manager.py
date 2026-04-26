"""
KIS (Korea Investment Securities) Trading Manager Module
=========================================================

한국투자증권 KIS API를 이용한 주문 실행 모듈.
AlpacaManager와 동일한 인터페이스를 제공한다.

참고: https://apiportal.koreainvestment.com/
"""

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import requests
import pandas as pd

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)

# KST timezone offset
_KST_OFFSET = timedelta(hours=9)


@dataclass
class KISAccount:
    """KIS 계좌 설정."""
    name: str
    app_key: str
    app_secret: str
    account_number: str = ""          # 계좌번호 (예: '12345678-01')
    base_url: str = "https://openapivts.koreainvestment.com:9443"  # 모의투자
    access_token: str = field(default="", repr=False)
    token_expires_at: Optional[datetime] = field(default=None, repr=False)

    @property
    def is_paper(self) -> bool:
        return "vts" in self.base_url


@dataclass
class OrderRequest:
    """주문 요청 구조체."""
    symbol: str          # 종목 코드 (6자리, 예: '005930')
    quantity: int        # 수량 (한국은 정수)
    side: str            # 'buy' | 'sell'
    order_type: str = 'market'   # 'market' | 'limit'
    limit_price: Optional[float] = None
    time_in_force: str = 'day'


@dataclass
class OrderResponse:
    """주문 응답 구조체."""
    order_id: str
    status: str
    symbol: str
    quantity: int
    filled_quantity: int
    side: str
    order_type: str
    submitted_at: datetime
    filled_at: Optional[datetime] = None
    average_fill_price: Optional[float] = None


class KISManager:
    """KIS API 주문 실행 매니저."""

    # KIS API 주문 유형 코드
    _ORDER_TYPE_MAP = {
        'market': '01',   # 시장가
        'limit': '00',    # 지정가
    }

    def __init__(self, accounts: List[KISAccount]):
        self.accounts = {acc.name: acc for acc in accounts}
        self.current_account: Optional[KISAccount] = None
        if len(self.accounts) == 1:
            self.current_account = list(self.accounts.values())[0]

    def set_account(self, account_name: str):
        if account_name not in self.accounts:
            raise ValueError(f"Account '{account_name}' not found")
        self.current_account = self.accounts[account_name]

    # ------------------------------------------------------------------
    # OAuth2 토큰 관리
    # ------------------------------------------------------------------

    def _ensure_token(self, account: Optional[KISAccount] = None) -> None:
        """토큰 만료 체크 및 자동 갱신 (24시간 만료)."""
        acct = account or self._get_account()
        now = datetime.utcnow()
        if acct.access_token and acct.token_expires_at and now < acct.token_expires_at - timedelta(minutes=5):
            return  # 유효한 토큰 존재

        url = f"{acct.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": acct.app_key,
            "appsecret": acct.app_secret,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            acct.access_token = data.get("access_token", "")
            expires_in = int(data.get("expires_in", 86400))
            acct.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
            logger.info(f"KIS token refreshed for account '{acct.name}'")
        except Exception as e:
            logger.error(f"KIS token refresh failed: {e}")
            raise

    # ------------------------------------------------------------------
    # 계좌 조회
    # ------------------------------------------------------------------

    def get_account_info(self, account_name: Optional[str] = None) -> Dict[str, Any]:
        """계좌 잔고 조회."""
        acct = self._get_account(account_name)
        self._ensure_token(acct)
        acct_num, acct_prod = self._split_account_number(acct.account_number)
        params = {
            "CANO": acct_num,
            "ACNT_PRDT_CD": acct_prod,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        tr_id = "VTTC8434R" if acct.is_paper else "TTTC8434R"
        return self._api_request("GET", "/uapi/domestic-stock/v1/trading/inquire-balance",
                                  acct, params=params, tr_id=tr_id)

    def get_positions(self, account_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """보유 포지션 조회."""
        info = self.get_account_info(account_name)
        output2 = info.get("output2", [{}])
        output1 = info.get("output1", [])
        positions = []
        for item in output1:
            qty = int(item.get("hldg_qty", "0") or "0")
            if qty <= 0:
                continue
            avg_price = float(item.get("pchs_avg_pric", "0") or "0")
            cur_price = float(item.get("prpr", "0") or "0")
            market_value = qty * cur_price
            positions.append({
                "symbol": item.get("pdno", ""),
                "qty": qty,
                "avg_entry_price": avg_price,
                "current_price": cur_price,
                "market_value": market_value,
            })
        return positions

    def get_portfolio_value(self, account_name: Optional[str] = None) -> float:
        """총 포트폴리오 평가 금액."""
        info = self.get_account_info(account_name)
        output2 = info.get("output2", [{}])
        if output2:
            return float(output2[0].get("tot_evlu_amt", "0") or "0")
        return 0.0

    # ------------------------------------------------------------------
    # 주문 실행
    # ------------------------------------------------------------------

    def place_order(self, order: OrderRequest,
                    account_name: Optional[str] = None) -> OrderResponse:
        """단일 주문 실행."""
        acct = self._get_account(account_name)
        self._ensure_token(acct)
        acct_num, acct_prod = self._split_account_number(acct.account_number)

        order_type_code = self._ORDER_TYPE_MAP.get(order.order_type, '01')
        is_buy = order.side.lower() == 'buy'

        if acct.is_paper:
            tr_id = "VTTC0802U" if is_buy else "VTTC0801U"
        else:
            tr_id = "TTTC0802U" if is_buy else "TTTC0801U"

        payload = {
            "CANO": acct_num,
            "ACNT_PRDT_CD": acct_prod,
            "PDNO": order.symbol,
            "ORD_DVSN": order_type_code,
            "ORD_QTY": str(int(order.quantity)),
            "ORD_UNPR": str(int(order.limit_price)) if order.limit_price else "0",
        }

        resp = self._api_request("POST",
                                  "/uapi/domestic-stock/v1/trading/order-cash",
                                  acct, json_body=payload, tr_id=tr_id)
        output = resp.get("output", {})
        order_id = output.get("ODNO", "")
        return OrderResponse(
            order_id=order_id,
            status="submitted",
            symbol=order.symbol,
            quantity=int(order.quantity),
            filled_quantity=0,
            side=order.side,
            order_type=order.order_type,
            submitted_at=datetime.now(),
        )

    def place_orders_batch(self, orders: List[OrderRequest],
                           account_name: Optional[str] = None) -> List[OrderResponse]:
        """복수 주문 실행."""
        responses = []
        for order in orders:
            try:
                resp = self.place_order(order, account_name)
                responses.append(resp)
                time.sleep(0.2)  # API rate limit
            except Exception as e:
                logger.error(f"Order failed for {order.symbol}: {e}")
                responses.append(OrderResponse(
                    order_id="", status="failed", symbol=order.symbol,
                    quantity=int(order.quantity), filled_quantity=0,
                    side=order.side, order_type=order.order_type,
                    submitted_at=datetime.now(),
                ))
        return responses

    def execute_portfolio_rebalance(self, target_weights: Dict[str, float],
                                    account_name: Optional[str] = None,
                                    dry_run: bool = False,
                                    market_closed_action: str = 'skip') -> Dict[str, Any]:
        """
        목표 비중으로 포트폴리오 리밸런싱 실행.
        AlpacaManager.execute_portfolio_rebalance()와 동일한 시그니처.
        """
        acct = self._get_account(account_name)
        is_open = self._is_market_open()

        if not is_open and market_closed_action == 'skip' and not dry_run:
            logger.info("KIS market closed; skipping rebalance")
            return {'orders_placed': 0, 'orders': [], 'market_open': False,
                    'target_weights': target_weights}

        positions = self.get_positions(account_name)
        portfolio_value = self.get_portfolio_value(account_name)

        current_weights: Dict[str, float] = {}
        for pos in positions:
            sym = pos['symbol']
            mv = pos['market_value']
            current_weights[sym] = (mv / portfolio_value) if portfolio_value > 0 else 0.0

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        full_target = {s: float(target_weights.get(s, 0.0)) for s in all_symbols}

        # Phase 1: SELL
        sell_orders = []
        for sym, tgt in full_target.items():
            cur = current_weights.get(sym, 0.0)
            if tgt - cur < -0.005:
                value_diff = (tgt - cur) * portfolio_value  # negative
                price = self._get_latest_price(sym, acct)
                if price and price > 0:
                    qty = abs(int(value_diff / price))
                    if qty > 0:
                        sell_orders.append(OrderRequest(symbol=sym, quantity=qty, side='sell'))

        # Phase 2: BUY
        buy_orders = []
        for sym, tgt in full_target.items():
            cur = current_weights.get(sym, 0.0)
            if tgt - cur > 0.005:
                value_diff = (tgt - cur) * portfolio_value  # positive
                price = self._get_latest_price(sym, acct)
                if price and price > 0:
                    qty = int(value_diff / price)
                    if qty > 0:
                        buy_orders.append(OrderRequest(symbol=sym, quantity=qty, side='buy'))

        if dry_run:
            return {
                'orders_placed': 0,
                'orders_plan': {
                    'sell': [o.__dict__ for o in sell_orders],
                    'buy': [o.__dict__ for o in buy_orders],
                },
                'market_open': is_open,
                'target_weights': full_target,
            }

        results_sell = self.place_orders_batch(sell_orders, account_name)
        results_buy = self.place_orders_batch(buy_orders, account_name)
        all_results = results_sell + results_buy

        return {
            'orders_placed': len(all_results),
            'orders': [r.__dict__ for r in all_results],
            'market_open': is_open,
            'target_weights': full_target,
        }

    # ------------------------------------------------------------------
    # 시세 조회
    # ------------------------------------------------------------------

    def _get_latest_price(self, symbol: str,
                           account: Optional[KISAccount] = None) -> Optional[float]:
        """현재가 조회."""
        acct = account or self._get_account()
        self._ensure_token(acct)
        try:
            tr_id = "VTTC8434R" if acct.is_paper else "FHKST01010100"
            params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol}
            resp = self._api_request("GET",
                                      "/uapi/domestic-stock/v1/quotations/inquire-price",
                                      acct, params=params, tr_id="FHKST01010100")
            output = resp.get("output", {})
            price_str = output.get("stck_prpr") or output.get("stck_clpr")
            if price_str:
                return float(price_str)
        except Exception as e:
            logger.debug(f"Price query failed for {symbol}: {e}")
        return None

    def _is_market_open(self) -> bool:
        """KST 기준 장 운영 시간 (09:00~15:30) 체크."""
        now_kst = datetime.utcnow() + _KST_OFFSET
        weekday = now_kst.weekday()  # 0=Mon, 6=Sun
        if weekday >= 5:
            return False
        h, m = now_kst.hour, now_kst.minute
        market_open = (h > 9) or (h == 9 and m >= 0)
        market_close = (h < 15) or (h == 15 and m <= 30)
        return market_open and market_close

    # ------------------------------------------------------------------
    # 내부 유틸
    # ------------------------------------------------------------------

    def _get_account(self, account_name: Optional[str] = None) -> KISAccount:
        if account_name:
            if account_name not in self.accounts:
                raise ValueError(f"Account '{account_name}' not found")
            return self.accounts[account_name]
        if self.current_account:
            return self.current_account
        raise ValueError("No account specified and no current account set")

    @staticmethod
    def _split_account_number(account_number: str):
        """'12345678-01' → ('12345678', '01')"""
        parts = account_number.replace('-', '')
        if len(parts) >= 10:
            return parts[:8], parts[8:10]
        return parts, '01'

    def _api_request(self, method: str, path: str,
                     account: KISAccount,
                     json_body: Optional[Dict] = None,
                     params: Optional[Dict] = None,
                     tr_id: str = "",
                     timeout: int = 30) -> Any:
        """KIS API 공통 요청."""
        url = f"{account.base_url}{path}"
        headers = {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {account.access_token}",
            "appkey": account.app_key,
            "appsecret": account.app_secret,
            "tr_id": tr_id,
            "custtype": "P",
        }
        try:
            resp = requests.request(
                method, url, headers=headers,
                json=json_body, params=params, timeout=timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"KIS API request failed [{method} {path}]: {e}")


# ------------------------------------------------------------------
# 팩토리 함수
# ------------------------------------------------------------------

def create_kis_manager_from_env(name: str = "default") -> KISManager:
    """환경변수에서 KIS 설정을 읽어 KISManager를 생성한다."""
    from src.config.settings import get_config
    config = get_config()
    kis_cfg = config.kis

    if not kis_cfg.app_key or not kis_cfg.app_secret:
        raise ValueError("KIS_APP_KEY and KIS_APP_SECRET must be set in .env")

    account = KISAccount(
        name=name,
        app_key=kis_cfg.app_key,
        app_secret=kis_cfg.app_secret,
        base_url=kis_cfg.base_url,
    )
    return KISManager([account])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    try:
        mgr = create_kis_manager_from_env()
        print("KIS Manager initialized successfully")
        print(f"Market open: {mgr._is_market_open()}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("KIS_APP_KEY, KIS_APP_SECRET 환경변수를 설정하세요")
