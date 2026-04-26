#!/bin/bash
# Full ML Pipeline: DART Refetch → prepare_krx_ml_data → ml_bucket_selection
# Usage: bash scripts/run_full_ml_pipeline.sh

set -e
source ~/.bashrc 2>/dev/null || true

DART_API_KEY=$(grep "^DART_API_KEY=" .env | cut -d= -f2)
export DART_API_KEY

echo "========================================"
echo "  FinRL Korean ML Pipeline"
echo "  $(date)"
echo "========================================"

# Step 1: DART refetch (if not already done)
echo ""
echo "[Step 1] DART 재무 데이터 재수집..."
conda run -n FinRL bash -c "
DART_API_KEY=$DART_API_KEY python src/data/refetch_dart_fundamentals.py \
    --start 2015-01-01 --end 2024-12-31
"

# Step 2: Prepare ML feature table
echo ""
echo "[Step 2] ML 피처 테이블 생성..."
conda run -n FinRL python src/data/prepare_krx_ml_data.py \
    --start 2015-01-01 --end 2024-12-31

# Step 3: ML training (GPU)
echo ""
echo "[Step 3] ML 모델 학습 (GPU)..."
conda run -n FinRL python src/strategies/ml_bucket_selection.py \
    --universe kospi200 \
    --val-cutoff 2022-12-31 \
    --val-quarters 4

echo ""
echo "========================================"
echo "  Pipeline complete! $(date)"
echo "========================================"
echo "다음 단계: 분기 팩터 백테스트 실행"
echo "  conda run -n FinRL python src/backtest/run_quarterly_backtest.py \\"
echo "    --predictions data/kospi200_ml_bucket_predictions_*.csv \\"
echo "    --start 2023-01-01 --end 2024-12-31"
