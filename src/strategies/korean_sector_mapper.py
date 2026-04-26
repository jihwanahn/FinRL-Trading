"""
Korean Sector Mapper Module
===========================

WICS(한국형 GICS) 섹터를 기존 버킷 전략의 bucket 분류로 매핑한다.

WICS (WISEfn Industry Classification Standard) 분류 체계를 사용하는
pykrx / FinanceDataReader 데이터와 호환된다.
"""

from typing import Dict, Optional, List

# WICS 섹터 → 버킷 매핑
# 기존 S&P 500 GICS 기반 bucket과 동일한 분류 체계 유지
WICS_TO_BUCKET: Dict[str, str] = {
    # Growth / Tech
    "IT": "growth_tech",
    "정보기술": "growth_tech",
    "커뮤니케이션서비스": "growth_tech",
    "Communication Services": "growth_tech",
    "통신서비스": "growth_tech",

    # Cyclical
    "경기관련소비재": "cyclical",
    "Consumer Discretionary": "cyclical",
    "금융": "cyclical",
    "Financials": "cyclical",
    "산업재": "cyclical",
    "Industrials": "cyclical",

    # Real Assets
    "에너지": "real_assets",
    "Energy": "real_assets",
    "소재": "real_assets",
    "Materials": "real_assets",
    "부동산": "real_assets",
    "Real Estate": "real_assets",

    # Defensive
    "건강관리": "defensive",
    "Health Care": "defensive",
    "필수소비재": "defensive",
    "Consumer Staples": "defensive",
    "유틸리티": "defensive",
    "Utilities": "defensive",
}

# 기존 S&P 500 GICS → bucket 매핑 (역방향 참조용)
GICS_TO_BUCKET: Dict[str, str] = {
    "Information Technology": "growth_tech",
    "Communication Services": "growth_tech",
    "Consumer Discretionary": "cyclical",
    "Financials": "cyclical",
    "Industrials": "cyclical",
    "Energy": "real_assets",
    "Materials": "real_assets",
    "Real Estate": "real_assets",
    "Health Care": "defensive",
    "Consumer Staples": "defensive",
    "Utilities": "defensive",
}

# 유효한 버킷 목록
VALID_BUCKETS = ["growth_tech", "cyclical", "real_assets", "defensive"]


def map_wics_to_bucket(sector: str) -> Optional[str]:
    """
    WICS 섹터명을 버킷으로 매핑한다.

    Args:
        sector: WICS 섹터명 (한국어 또는 영어)

    Returns:
        버킷명 또는 None (매핑 없음)
    """
    if not sector:
        return None
    # 정확 매핑
    if sector in WICS_TO_BUCKET:
        return WICS_TO_BUCKET[sector]
    # 부분 매칭 시도
    sector_lower = sector.lower()
    for key, bucket in WICS_TO_BUCKET.items():
        if key.lower() in sector_lower or sector_lower in key.lower():
            return bucket
    return None


def map_gics_to_bucket(sector: str) -> Optional[str]:
    """GICS 섹터명을 버킷으로 매핑한다."""
    if not sector:
        return None
    return GICS_TO_BUCKET.get(sector)


def map_sector_to_bucket(sector: str, source: str = 'auto') -> Optional[str]:
    """
    섹터명을 버킷으로 매핑한다. 데이터 소스에 관계없이 동작한다.

    Args:
        sector: 섹터명
        source: 'wics' | 'gics' | 'auto'

    Returns:
        버킷명 또는 None
    """
    if source == 'wics':
        return map_wics_to_bucket(sector)
    if source == 'gics':
        return map_gics_to_bucket(sector)
    # auto: WICS 우선, 없으면 GICS 시도
    result = map_wics_to_bucket(sector)
    if result is None:
        result = map_gics_to_bucket(sector)
    return result


def enrich_tickers_with_bucket(tickers_df, sector_col: str = 'sectors',
                                bucket_col: str = 'bucket') -> 'pd.DataFrame':
    """
    종목 DataFrame에 버킷 컬럼을 추가한다.

    Args:
        tickers_df: DataFrame with sector column
        sector_col: 섹터 컬럼명
        bucket_col: 추가할 버킷 컬럼명

    Returns:
        DataFrame with bucket column added
    """
    import pandas as pd
    df = tickers_df.copy()
    df[bucket_col] = df[sector_col].apply(lambda s: map_sector_to_bucket(str(s) if s else ''))
    unmapped = df[df[bucket_col].isna()][sector_col].unique()
    if len(unmapped) > 0:
        import logging
        logging.getLogger(__name__).info(
            f"Unmapped sectors (defaulting to 'cyclical'): {list(unmapped)}"
        )
        df[bucket_col] = df[bucket_col].fillna('cyclical')
    return df


if __name__ == '__main__':
    # 매핑 테스트
    test_sectors = [
        "IT", "정보기술", "커뮤니케이션서비스", "경기관련소비재",
        "금융", "산업재", "에너지", "소재", "부동산",
        "건강관리", "필수소비재", "유틸리티",
        "Information Technology", "Financials", "Health Care",
        "Unknown Sector",
    ]
    print("WICS/GICS → Bucket 매핑 테스트")
    print("-" * 40)
    for s in test_sectors:
        b = map_sector_to_bucket(s)
        print(f"  {s:30s} → {b}")
