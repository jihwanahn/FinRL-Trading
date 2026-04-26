"""
Swing ML Engine Module (Korean Stocks)
======================================

ML 기반 단기 수익률 예측 스윙 트레이딩 전략.
롤링 윈도우 학습으로 5거래일 후 수익률을 예측한다.

피처:
    - 5일/20일 수익률
    - RSI, MACD 히스토그램
    - 볼린저 밴드 %B
    - 거래량 변화율

타겟:
    - y_return_5d: 5거래일 후 로그 수익률
"""

import logging
import os
import sys
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

logger = logging.getLogger(__name__)

# 학습 모델 기본값
_DEFAULT_MODEL = 'lightgbm'


class SwingMLEngine:
    """
    ML 기반 단기 수익률 예측 엔진.
    롤링 윈도우 (훈련 2년, 예측 1주) 방식으로 학습한다.
    """

    def __init__(
        self,
        model_type: str = _DEFAULT_MODEL,
        train_window_days: int = 504,   # 약 2년
        predict_horizon_days: int = 5,  # 5거래일
        n_estimators: int = 100,
        max_depth: int = 5,
        min_signal_threshold: float = 0.0,
        max_positions: int = 10,
    ):
        self.model_type = model_type
        self.train_window_days = train_window_days
        self.predict_horizon_days = predict_horizon_days
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_signal_threshold = min_signal_threshold
        self.max_positions = max_positions
        self._models: Dict = {}  # ticker -> trained model

    # ------------------------------------------------------------------
    # 피처 계산
    # ------------------------------------------------------------------

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """단일 종목 피처 계산."""
        df = df.copy()
        close = df['close'].astype(float)
        volume = df['volume'].astype(float)

        # 수익률
        df['ret_5d'] = close.pct_change(5)
        df['ret_20d'] = close.pct_change(20)

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD 히스토그램
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal

        # 볼린저 밴드 %B
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_pct_b'] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)

        # 거래량 변화율
        df['vol_ratio'] = volume / volume.rolling(20).mean()

        # 타겟: 5거래일 후 로그 수익률
        df['y_return_5d'] = np.log(close.shift(-self.predict_horizon_days) / close)

        return df

    # ------------------------------------------------------------------
    # 학습 및 예측
    # ------------------------------------------------------------------

    FEATURE_COLS = ['ret_5d', 'ret_20d', 'rsi', 'macd_hist', 'bb_pct_b', 'vol_ratio']

    def _build_model(self):
        """모델 인스턴스 생성."""
        if self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=0.05,
                    verbose=-1,
                )
            except ImportError:
                pass
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
            )
        except ImportError:
            from sklearn.linear_model import Ridge
            return Ridge()

    def train_and_predict(self, price_data: Dict[str, pd.DataFrame],
                           predict_date: pd.Timestamp) -> Dict[str, float]:
        """
        롤링 윈도우 학습 + 예측.

        Args:
            price_data: {ticker: DataFrame(date, open, high, low, close, volume)}
            predict_date: 예측 기준일

        Returns:
            {ticker: predicted_5d_return}
        """
        predictions: Dict[str, float] = {}
        train_start = predict_date - pd.Timedelta(days=self.train_window_days + 30)

        for ticker, df in price_data.items():
            try:
                df_feat = self._compute_features(df)
                # 학습 구간
                train_mask = (df_feat['date'] >= train_start) & (df_feat['date'] < predict_date)
                train_df = df_feat[train_mask].dropna(subset=self.FEATURE_COLS + ['y_return_5d'])
                if len(train_df) < 60:
                    continue

                X_train = train_df[self.FEATURE_COLS].values
                y_train = train_df['y_return_5d'].values

                model = self._build_model()
                model.fit(X_train, y_train)

                # 예측 (가장 최근 행)
                pred_row = df_feat[df_feat['date'] == predict_date]
                if pred_row.empty:
                    pred_row = df_feat[df_feat['date'] <= predict_date].tail(1)
                if pred_row.empty:
                    continue
                X_pred = pred_row[self.FEATURE_COLS].values
                if np.any(np.isnan(X_pred)):
                    continue
                pred = float(model.predict(X_pred)[0])
                predictions[ticker] = pred
                self._models[ticker] = model

            except Exception as e:
                logger.debug(f"ML prediction failed for {ticker}: {e}")

        return predictions

    def generate_weights(self, price_data: Dict[str, pd.DataFrame],
                          predict_date: pd.Timestamp) -> Dict[str, float]:
        """
        예측값을 포트폴리오 비중으로 변환한다.

        Returns:
            {ticker: weight}
        """
        preds = self.train_and_predict(price_data, predict_date)
        # 양수 수익률 예측 종목만 선택
        positive = {t: v for t, v in preds.items() if v > self.min_signal_threshold}
        if not positive:
            return {}
        # 상위 max_positions 종목만
        top = sorted(positive.items(), key=lambda x: x[1], reverse=True)[:self.max_positions]
        total = sum(v for _, v in top)
        if total <= 0:
            return {t: 1.0 / len(top) for t, _ in top}
        return {t: v / total for t, v in top}
