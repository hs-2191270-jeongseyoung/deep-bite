"""
DeepBite (딥바이트) - FastAPI 양방향 상권 적합도 추천 API 라우터
Repository: https://github.com/hs-2191270-jeongseyoung/deep-bite

사용 방법:
기존 `main.py` 코드를 수정하지 않고 아래 두 줄만 추가하면 즉시 연동됩니다:
    from ai_router import router as ai_suitability_router
    app.include_router(ai_suitability_router)

제공 엔드포인트:
1. GET /api/ai/recommend-industry?region=안암동&top_k=5
   - 사용자가 선택한 성북구 권역(또는 lat, lng 위경도 좌표)에 가장 적합한 음식점 업종 추천
2. GET /api/ai/recommend-location?category=카페/디저트&top_k=5
   - 사용자가 선택한 희망 음식점 업종으로 개업하기 가장 좋은 성북구 행정동 권역 추천
3. GET /api/ai/grid-summary
   - 성북구 12대 권역 x 9대 업종(108개 상권 그리드) 전체 적합도 및 공공데이터 지표 반환
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Query

from deepbite_bidirectional_ai import DeepBiteRecommender, train_bidirectional_model

router = APIRouter(prefix="/api/ai", tags=["DeepBite AI Suitability"])

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURES_CSV = os.path.join(_BASE_DIR, "data", "seongbuk_training_features.csv")
if not os.path.exists(_FEATURES_CSV):
    _FEATURES_CSV = os.path.join(_BASE_DIR, "seongbuk_training_features.csv")

_MODEL_DIR = os.path.join(_BASE_DIR, "models")
_RECOMMENDER: Optional[DeepBiteRecommender] = None


def get_recommender() -> DeepBiteRecommender:
    global _RECOMMENDER
    if _RECOMMENDER is None:
        artifacts = train_bidirectional_model(
            features_csv_path=_FEATURES_CSV,
            model_save_dir=_MODEL_DIR,
            epochs=450,
            lr=0.008,
        )
        _RECOMMENDER = DeepBiteRecommender(
            features_csv_path=_FEATURES_CSV,
            artifacts=artifacts,
        )
    return _RECOMMENDER


@router.get("/recommend-industry")
def recommend_industry(
    region: Optional[str] = Query(default=None, description="성북구 권역명 (예: 안암동, 동선동, 성북동)"),
    lat: Optional[float] = Query(default=None, description="클릭한 위치의 위도 (예: 37.5862)"),
    lng: Optional[float] = Query(default=None, description="클릭한 위치의 경도 (예: 127.0290)"),
    top_k: int = Query(default=5, ge=1, le=9, description="반환할 상위 추천 업종 수"),
) -> Dict[str, object]:
    """
    [방향 1: 지역 선택 -> 최적 업종 추천]
    """
    rec = get_recommender()
    items = rec.recommend_categories_for_region(region=region, lat=lat, lng=lng, top_k=top_k)
    resolved_region = items[0]["권역"] if items else (region or "동선동")
    return {
        "direction": "region_to_industry",
        "selected_region": resolved_region,
        "recommendations": items,
    }


@router.get("/recommend-location")
def recommend_location(
    category: str = Query(..., description="희망 음식점 업종 (예: 카페/디저트, 한식, 일식/돈까스/초밥)"),
    top_k: int = Query(default=5, ge=1, le=12, description="반환할 상위 추천 성북구 권역 수"),
) -> Dict[str, object]:
    """
    [방향 2: 업종 선택 -> 최적 지역 추천]
    """
    rec = get_recommender()
    items = rec.recommend_regions_for_category(category=category, top_k=top_k)
    resolved_cat = items[0]["희망_업종"] if items else category
    return {
        "direction": "industry_to_region",
        "selected_category": resolved_cat,
        "recommendations": items,
    }


@router.get("/grid-summary")
def get_grid_summary() -> Dict[str, object]:
    """
    성북구 12대 권역 x 9대 업종 전체 상권 적합도 그리드 데이터를 반환합니다.
    """
    rec = get_recommender()
    return {
        "total_grids": len(rec.df),
        "regions": rec.artifacts.regions,
        "categories": rec.artifacts.categories,
        "items": rec.df.to_dict(orient="records"),
    }