"""
DeepBite (딥바이트) - 성북구 구글 리뷰 데이터 전처리 및 공공데이터 3종(길단위인구·점포·추정매출) 통합 파이프라인
Repository: https://github.com/hs-2191270-jeongseyoung/deep-bite

주요 기능:
1. 성북구 외 타 자치구 경계 데이터 및 비음식점(아파트, 역, 도서관, 주민센터, 주유소 등) 노이즈 제거
2. 도로명주소 및 위경도 좌표를 활용한 성북구 12대 행정/법정 권역 자동 매핑
3. 미분류 '음식점' 카테고리의 상호명·핵심 리뷰 텍스트 기반 9대 표준 업종 복원
4. 핵심 리뷰 텍스트 감성 점수(긍정/부정 키워드 비율) 및 리뷰 신뢰도 가중 평점 산출
5. 서울시 상권분석서비스 공공데이터 3종 실데이터 병합:
   - 서울시 상권분석서비스(길단위인구-행정동).csv: 총 유동인구, 2030 청년층 비중, 야간/저녁 유동 비중
   - 서울시 상권분석서비스(점포-행정동).csv: 공식 외식업 점포수, 프랜차이즈 비율, 개업률, 폐업률
   - 서울시 상권분석서비스(추정매출-행정동).csv: 업종별 당월 매출 금액/건수, 점포당 평균 매출, 주말/2030 매출 비중
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# 성북구 12대 주요 상권/법정동 중심 좌표 (위도, 경도) 및 기본 유동인구 베이스라인
SEONGBUK_REGIONS: Dict[str, Tuple[float, float, float]] = {
    "성북동": (37.5925, 126.9985, 68.0),
    "삼선동": (37.5865, 127.0110, 74.0),
    "동선동": (37.5928, 127.0175, 92.0),   # 성신여대입구역 핵심 상권
    "돈암동": (37.5975, 127.0165, 82.0),
    "안암동": (37.5862, 127.0290, 90.0),   # 고려대/안암역 핵심 상권
    "보문동": (37.5815, 127.0195, 70.0),
    "정릉동": (37.6065, 127.0075, 76.0),   # 국민대/정릉 상권
    "길음동": (37.6045, 127.0245, 88.0),   # 길음역/길음뉴타운 상권
    "종암동": (37.5990, 127.0335, 79.0),
    "월곡동": (37.6048, 127.0425, 81.0),   # 동덕여대/월곡역 상권 (월곡1·2동 통합)
    "장위동": (37.6145, 127.0495, 75.0),
    "석관동": (37.6110, 127.0605, 77.0),   # 한예종/석계역 인접 상권
}

# 성북구 20개 행정동 -> 12대 권역 매핑
DONG_TO_REGION: Dict[str, str] = {
    "성북동": "성북동",
    "삼선동": "삼선동",
    "동선동": "동선동",
    "돈암1동": "돈암동",
    "돈암2동": "돈암동",
    "안암동": "안암동",
    "보문동": "보문동",
    "정릉1동": "정릉동",
    "정릉2동": "정릉동",
    "정릉3동": "정릉동",
    "정릉4동": "정릉동",
    "길음1동": "길음동",
    "길음2동": "길음동",
    "종암동": "종암동",
    "월곡1동": "월곡동",
    "월곡2동": "월곡동",
    "장위1동": "장위동",
    "장위2동": "장위동",
    "장위3동": "장위동",
    "석관동": "석관동",
}

# 서울시 상권분석서비스 외식업종명 -> DeepBite 9대 표준 업종 매핑 (다대다 가중 매핑 지원)
PUBLIC_CAT_TO_STD: Dict[str, List[str]] = {
    "한식음식점": ["한식", "아시안/해산물/기타"],
    "커피-음료": ["카페/디저트"],
    "제과점": ["카페/디저트"],
    "일식음식점": ["일식/돈까스/초밥", "아시안/해산물/기타"],
    "중식음식점": ["중식"],
    "양식음식점": ["양식/브런치", "아시안/해산물/기타"],
    "패스트푸드점": ["양식/브런치"],
    "호프-간이주점": ["주점/바", "치킨/호프"],
    "분식전문점": ["분식/면류"],
    "치킨전문점": ["치킨/호프"],
}

# 도로명 -> 성북구 법정동/권역 매핑 사전
ROAD_TO_REGION: Dict[str, str] = {
    "성북로": "성북동",
    "대사관로": "성북동",
    "선잠로": "성북동",
    "창경궁로": "성북동",
    "혜화로": "성북동",
    "삼선교로": "삼선동",
    "보문로": "보문동",
    "고려대로": "안암동",
    "인촌로": "안암동",
    "개운사길": "안암동",
    "안암로": "안암동",
    "동소문로": "동선동",
    "아리랑로": "돈암동",
    "보국문로": "정릉동",
    "솔샘로": "정릉동",
    "정릉로": "정릉동",
    "길음로": "길음동",
    "삼양로": "길음동",
    "종암로": "종암동",
    "오패산로": "월곡동",
    "월곡로": "월곡동",
    "화랑로": "월곡동",
    "장월로": "월곡동",
    "돌곶이로": "장위동",
    "장위로": "장위동",
    "한천로": "석관동",
}

# 9대 표준 음식점 업종 분류 체계
STANDARD_CATEGORIES: List[str] = [
    "한식",
    "카페/디저트",
    "일식/돈까스/초밥",
    "중식",
    "양식/브런치",
    "주점/바",
    "분식/면류",
    "치킨/호프",
    "아시안/해산물/기타",
]

# 비음식점(노이즈 POI) 필터링 키워드
NON_FOOD_KEYWORDS: List[str] = [
    "아파트", "래미안", "편한세상", "푸르지오", "아이파크", "주민센터",
    "도서관", "주유소", "세차", "정비", "부동산", "약국", "병원", "의원",
    "치과", "한의원", "초등학교", "중학교", "고등학교", "대학교", "어린이집",
    "유치원", "교회", "성당", "사찰", "우체국", "파출소", "지구대", "은행",
]

# 감성 분석용 한국어 핵심 리뷰 키워드
POSITIVE_KEYWORDS: List[str] = [
    "맛집", "맛있", "친절", "가성비", "최고", "추천", "신선", "깔끔",
    "푸짐", "재방문", "단골", "분위기", "훌륭", "정성", "바삭", "부드럽",
    "국물", "좋아", "만족", "인생",
]

NEGATIVE_KEYWORDS: List[str] = [
    "불친절", "별로", "실망", "비위생", "최악", "비추", "비싸", "느림",
    "짜다", "싱겁", "냄새", "기름지", "아쉽", "불쾌", "다시는", "줄었",
]


def _read_csv_auto(path: str) -> pd.DataFrame:
    for enc in ("cp949", "utf-8-sig", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {path}")


def is_valid_restaurant(row: pd.Series) -> bool:
    """성북구 소재 실제 음식점/카페인지 판별합니다."""
    addr = str(row.get("주소", ""))
    name = str(row.get("음식점 이름", ""))
    cat = str(row.get("판매 품목(카테고리)", ""))

    if "성북구" not in addr:
        return False

    if name.endswith("역") and len(name) <= 5:
        return False

    for kw in NON_FOOD_KEYWORDS:
        if kw in name or kw in cat:
            return False

    return True


def map_seongbuk_region(address: str, lat: float, lng: float) -> str:
    """주소 텍스트(법정동/도로명)와 위경도 최근접 중심점을 결합해 성북구 12대 권역으로 매핑합니다."""
    addr = str(address)

    if "성북동" in addr:
        return "성북동"
    if "삼선동" in addr:
        return "삼선동"
    if "동선동" in addr or "동소문동" in addr:
        return "동선동"
    if "돈암동" in addr:
        return "돈암동"
    if "안암동" in addr:
        return "안암동"
    if "보문동" in addr:
        return "보문동"
    if "정릉" in addr:
        return "정릉동"
    if "길음" in addr:
        return "길음동"
    if "종암" in addr:
        return "종암동"
    if "월곡" in addr:
        return "월곡동"
    if "장위" in addr:
        return "장위동"
    if "석관" in addr:
        return "석관동"

    for road_kw, region_name in ROAD_TO_REGION.items():
        if road_kw in addr:
            if road_kw == "동소문로" and not pd.isna(lat) and lat >= 37.601:
                return "길음동"
            return region_name

    if not pd.isna(lat) and not pd.isna(lng):
        best_region = "동선동"
        min_dist = float("inf")
        for reg_name, (r_lat, r_lng, _) in SEONGBUK_REGIONS.items():
            dist = (lat - r_lat) ** 2 + (lng - r_lng) ** 2
            if dist < min_dist:
                min_dist = dist
                best_region = reg_name
        return best_region

    return "동선동"


def _match_category_from_text(text: str) -> Optional[str]:
    """단일 텍스트(카테고리명, 상호명, 리뷰 순)에서 표준 업종 키워드를 우선순위로 매칭합니다."""
    if any(k in text for k in [
        "카페", "커피", "디저트", "베이커리", "제과", "빵집", "케이크", "마카롱",
        "빙수", "티하우스", "와플", "도넛", "아이스크림", "로스터", "에스프레소", "찻집",
    ]):
        return "카페/디저트"

    if any(k in text for k in ["치킨", "통닭", "닭강정", "호프", "비어"]):
        return "치킨/호프"

    if any(k in text for k in [
        "술집", "주점", "포차", "이자카야", "와인바", "칵테일", "펍",
        "생맥주", "막걸리", "전집", "오뎅바",
    ]) or text.strip() in ["바"]:
        return "주점/바"

    if any(k in text for k in [
        "중식", "중국", "짜장", "짬뽕", "마라", "양꼬치", "탕수육", "딤섬", "훠궈", "중화",
    ]):
        return "중식"

    if any(k in text for k in [
        "일식", "초밥", "스시", "라멘", "돈까스", "돈카츠", "우동", "소바",
        "사케동", "텐동", "샤브", "횟집", "오마카세", "일식당",
    ]):
        return "일식/돈까스/초밥"

    if any(k in text for k in [
        "양식", "이탈리아", "이탈리안", "파스타", "피자", "스테이크", "브런치", "버거", "햄버거",
        "샌드위치", "샐러드", "비스트로", "타코", "멕시코", "멕시칸", "프랑스",
    ]):
        return "양식/브런치"

    if any(k in text for k in [
        "분식", "떡볶이", "김밥", "순대", "칼국수", "국수", "수제비",
        "냉면", "만두", "라면", "쫄면",
    ]):
        return "분식/면류"

    if any(k in text for k in [
        "쌀국수", "베트남", "태국", "인도", "커리", "카레", "아시안", "해물", "해산물", "조개", "장어",
    ]):
        return "아시안/해산물/기타"

    if any(k in text for k in [
        "한식", "고기", "삼겹살", "갈비", "곱창", "막창", "족발", "보쌈", "국밥",
        "찌개", "순두부", "설렁탕", "곰탕", "감자탕", "해장국", "백반", "한정식",
        "비빔밥", "불고기", "닭갈비", "삼계탕", "찜닭", "아구찜", "생선구이", "육회",
        "반찬", "김치", "된장", "불백", "정식", "식당", "밥집",
    ]):
        return "한식"

    return None


def standardize_category(raw_cat: str, name: str, review: str) -> str:
    c = str(raw_cat).strip() if not pd.isna(raw_cat) else ""
    n = str(name).strip() if not pd.isna(name) else ""
    r = str(review).strip() if not pd.isna(review) else ""

    if c and c not in ["음식점", "식당", "레스토랑", "음식"]:
        matched_c = _match_category_from_text(c)
        if matched_c:
            return matched_c

    matched_n = _match_category_from_text(n)
    if matched_n:
        return matched_n

    matched_r = _match_category_from_text(r)
    if matched_r:
        return matched_r

    return "한식"


def compute_review_sentiment(review_text: str) -> float:
    """핵심 리뷰 텍스트에서 긍정/부정 감성 점수(0.0 ~ 1.0)를 계산합니다."""
    if pd.isna(review_text) or not str(review_text).strip():
        return 0.55

    text = str(review_text)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

    raw_score = 0.58 + 0.09 * pos_hits - 0.14 * neg_hits
    return float(np.clip(raw_score, 0.10, 0.98))


def load_seoul_public_datasets(
    floating_pop_csv: Optional[str] = None,
    store_csv: Optional[str] = None,
    estimated_sales_csv: Optional[str] = None,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float]],
    Dict[Tuple[str, str], Dict[str, float]],
]:
    """
    서울시 상권분석서비스 3종(길단위인구, 점포, 추정매출)에서 성북구(행정동코드 11290*) 최근 4개 분기
    평균 지표를 추출하여 12대 권역 × 9대 표준 업종 그리드로 매핑합니다.
    """
    pop_metrics: Dict[str, Dict[str, float]] = {
        reg: {
            "총_유동인구": base_pop * 15000.0,
            "유동인구_지수": base_pop,
            "청년층_2030_유동비중": 0.34,
            "야간_17_24_유동비중": 0.32,
        }
        for reg, (_, _, base_pop) in SEONGBUK_REGIONS.items()
    }
    store_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
    sales_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}

    # 1. 길단위인구-행정동 처리
    if floating_pop_csv and os.path.exists(floating_pop_csv):
        pop_df = _read_csv_auto(floating_pop_csv)
        sb_pop = pop_df[pop_df["행정동_코드"].astype(str).str.startswith("11290")].copy()
        if not sb_pop.empty:
            recent_q = sorted(sb_pop["기준_년분기_코드"].unique())[-4:]
            sb_pop = sb_pop[sb_pop["기준_년분기_코드"].isin(recent_q)].copy()
            sb_pop["권역"] = sb_pop["행정동_코드_명"].map(DONG_TO_REGION)
            sb_pop = sb_pop.dropna(subset=["권역"])

            # 동일 분기 내 복수 행정동(예: 길음1·2동) 합산 후 분기 평균
            q_reg = sb_pop.groupby(["기준_년분기_코드", "권역"], as_index=False)[
                [
                    "총_유동인구_수",
                    "연령대_20_유동인구_수",
                    "연령대_30_유동인구_수",
                    "시간대_17_21_유동인구_수",
                    "시간대_21_24_유동인구_수",
                ]
            ].sum()
            reg_avg = q_reg.groupby("권역", as_index=False).mean()

            max_pop = float(reg_avg["총_유동인구_수"].max()) or 1.0
            min_pop = float(reg_avg["총_유동인구_수"].min()) or 0.0
            for _, row in reg_avg.iterrows():
                reg = str(row["권역"])
                tot = max(float(row["총_유동인구_수"]), 1.0)
                youth_ratio = float(row["연령대_20_유동인구_수"] + row["연령대_30_유동인구_수"]) / tot
                night_ratio = float(row["시간대_17_21_유동인구_수"] + row["시간대_21_24_유동인구_수"]) / tot
                pop_idx = 52.0 + 44.0 * ((tot - min_pop) / max(max_pop - min_pop, 1.0))
                pop_metrics[reg] = {
                    "총_유동인구": round(tot, 1),
                    "유동인구_지수": round(float(np.clip(pop_idx, 45.0, 98.0)), 2),
                    "청년층_2030_유동비중": round(youth_ratio, 4),
                    "야간_17_24_유동비중": round(night_ratio, 4),
                }

    # 2. 점포-행정동 처리
    if store_csv and os.path.exists(store_csv):
        st_df = _read_csv_auto(store_csv)
        sb_st = st_df[
            st_df["행정동_코드"].astype(str).str.startswith("11290")
            & st_df["서비스_업종_코드_명"].isin(PUBLIC_CAT_TO_STD.keys())
        ].copy()
        if not sb_st.empty:
            recent_q = sorted(sb_st["기준_년분기_코드"].unique())[-4:]
            sb_st = sb_st[sb_st["기준_년분기_코드"].isin(recent_q)].copy()
            sb_st["권역"] = sb_st["행정동_코드_명"].map(DONG_TO_REGION)
            sb_st = sb_st.dropna(subset=["권역"])

            expanded_rows = []
            for _, r in sb_st.iterrows():
                mapped_cats = PUBLIC_CAT_TO_STD.get(str(r["서비스_업종_코드_명"]), [])
                for std_cat in mapped_cats:
                    weight = 0.45 if std_cat == "아시안/해산물/기타" else 1.0
                    expanded_rows.append({
                        "기준_년분기_코드": r["기준_년분기_코드"],
                        "권역": r["권역"],
                        "표준업종": std_cat,
                        "전체_점포_수": float(r["전체_점포_수"]) * weight,
                        "프랜차이즈_점포_수": float(r["프랜차이즈_점포_수"]) * weight,
                        "개업_점포_수": float(r["개업_점포_수"]) * weight,
                        "폐업_점포_수": float(r["폐업_점포_수"]) * weight,
                    })
            exp_df = pd.DataFrame(expanded_rows)
            q_agg = exp_df.groupby(["기준_년분기_코드", "권역", "표준업종"], as_index=False).sum()
            avg_st = q_agg.groupby(["권역", "표준업종"], as_index=False).mean()

            for _, row in avg_st.iterrows():
                reg = str(row["권역"])
                cat = str(row["표준업종"])
                tot_s = max(float(row["전체_점포_수"]), 1.0)
                fran_ratio = float(row["프랜차이즈_점포_수"]) / tot_s
                open_rate = (float(row["개업_점포_수"]) / tot_s) * 100.0
                close_rate = (float(row["폐업_점포_수"]) / tot_s) * 100.0
                store_metrics[(reg, cat)] = {
                    "공식_점포수": round(tot_s, 1),
                    "프랜차이즈_비중": round(float(np.clip(fran_ratio, 0.0, 1.0)), 4),
                    "공식_개업률": round(open_rate, 2),
                    "공식_폐업률": round(close_rate, 2),
                }

    # 3. 추정매출-행정동 처리
    if estimated_sales_csv and os.path.exists(estimated_sales_csv):
        sl_df = _read_csv_auto(estimated_sales_csv)
        sb_sl = sl_df[
            sl_df["행정동_코드"].astype(str).str.startswith("11290")
            & sl_df["서비스_업종_코드_명"].isin(PUBLIC_CAT_TO_STD.keys())
        ].copy()
        if not sb_sl.empty:
            recent_q = sorted(sb_sl["기준_년분기_코드"].unique())[-4:]
            sb_sl = sb_sl[sb_sl["기준_년분기_코드"].isin(recent_q)].copy()
            sb_sl["권역"] = sb_sl["행정동_코드_명"].map(DONG_TO_REGION)
            sb_sl = sb_sl.dropna(subset=["권역"])

            expanded_sales = []
            for _, r in sb_sl.iterrows():
                mapped_cats = PUBLIC_CAT_TO_STD.get(str(r["서비스_업종_코드_명"]), [])
                for std_cat in mapped_cats:
                    weight = 0.45 if std_cat == "아시안/해산물/기타" else 1.0
                    expanded_sales.append({
                        "기준_년분기_코드": r["기준_년분기_코드"],
                        "권역": r["권역"],
                        "표준업종": std_cat,
                        "당월_매출_금액": float(r["당월_매출_금액"]) * weight,
                        "당월_매출_건수": float(r["당월_매출_건수"]) * weight,
                        "주말_매출_금액": float(r["주말_매출_금액"]) * weight,
                        "연령대_2030_매출": float(r["연령대_20_매출_금액"] + r["연령대_30_매출_금액"]) * weight,
                    })
            exp_sl_df = pd.DataFrame(expanded_sales)
            q_sl = exp_sl_df.groupby(["기준_년분기_코드", "권역", "표준업종"], as_index=False).sum()
            avg_sl = q_sl.groupby(["권역", "표준업종"], as_index=False).mean()

            # 점포당 월매출 산출
            per_store_list = []
            for _, row in avg_sl.iterrows():
                reg = str(row["권역"])
                cat = str(row["표준업종"])
                st_cnt = store_metrics.get((reg, cat), {}).get("공식_점포수", 15.0)
                per_store_sales = float(row["당월_매출_금액"]) / max(st_cnt, 1.0)
                per_store_list.append(per_store_sales)

            avg_sl["점포당_매출"] = per_store_list
            log_sales = np.log1p(avg_sl["점포당_매출"].values)
            min_ls, max_ls = float(log_sales.min()), float(log_sales.max())

            for idx_i, row in avg_sl.iterrows():
                reg = str(row["권역"])
                cat = str(row["표준업종"])
                tot_amt = max(float(row["당월_매출_금액"]), 1.0)
                ls_val = float(np.log1p(row["점포당_매출"]))
                sales_idx = 48.0 + 48.0 * ((ls_val - min_ls) / max(max_ls - min_ls, 1e-6))
                sales_metrics[(reg, cat)] = {
                    "월평균_매출금액_백만원": round(tot_amt / 1e6, 2),
                    "점포당_월매출_만원": round(float(row["점포당_매출"]) / 1e4, 1),
                    "추정매출_지수": round(float(np.clip(sales_idx, 42.0, 98.0)), 2),
                    "연령대_2030_매출비중": round(float(row["연령대_2030_매출"]) / tot_amt, 4),
                    "주말_매출비중": round(float(row["주말_매출_금액"]) / tot_amt, 4),
                }

    return pop_metrics, store_metrics, sales_metrics


def run_preprocessing_pipeline(
    input_excel_path: str,
    output_cleaned_csv: str,
    output_features_csv: str,
    floating_pop_csv: Optional[str] = None,
    store_csv: Optional[str] = None,
    estimated_sales_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    성북구 음식점 엑셀 데이터와 서울시 상권분석서비스 3종 CSV를 정제·병합하여
    양방향 PyTorch 모델 학습용 피처 테이블을 생성합니다.
    """
    raw_df = pd.read_excel(input_excel_path)

    valid_mask = raw_df.apply(is_valid_restaurant, axis=1)
    df = raw_df[valid_mask].copy()
    df = df.drop_duplicates(subset=["음식점 이름", "위도", "경도"]).reset_index(drop=True)

    df["권역(행정동)"] = df.apply(
        lambda r: map_seongbuk_region(r["주소"], r["위도"], r["경도"]), axis=1
    )
    df["표준업종"] = df.apply(
        lambda r: standardize_category(
            r["판매 품목(카테고리)"], r["음식점 이름"], r["핵심 리뷰"]
        ),
        axis=1,
    )

    df["리뷰 수"] = pd.to_numeric(df["리뷰 수"], errors="coerce").fillna(0).astype(int)
    df["평점"] = pd.to_numeric(df["평점"], errors="coerce")
    group_median = df.groupby(["권역(행정동)", "표준업종"])["평점"].transform("median")
    df["평점_보정"] = df["평점"].fillna(group_median).fillna(4.30).round(2)

    df["리뷰_감성점수"] = df["핵심 리뷰"].apply(compute_review_sentiment).round(3)
    df["로그_리뷰수"] = np.log1p(df["리뷰 수"]).round(4)

    global_mean_rating = float(df["평점_보정"].mean())
    m_prior = 15.0
    df["신뢰도가중_평점"] = (
        (df["리뷰 수"] / (df["리뷰 수"] + m_prior)) * df["평점_보정"]
        + (m_prior / (df["리뷰 수"] + m_prior)) * global_mean_rating
    ).round(3)

    pop_map, store_map, sales_map = load_seoul_public_datasets(
        floating_pop_csv=floating_pop_csv,
        store_csv=store_csv,
        estimated_sales_csv=estimated_sales_csv,
    )

    region_total_stores = df.groupby("권역(행정동)").size().to_dict()
    region_total_reviews = df.groupby("권역(행정동)")["리뷰 수"].sum().to_dict()

    feature_rows: List[Dict[str, object]] = []
    for reg_name, (r_lat, r_lng, _) in SEONGBUK_REGIONS.items():
        reg_stores = max(region_total_stores.get(reg_name, 1), 1)
        reg_reviews = max(region_total_reviews.get(reg_name, 1), 1)

        p_info = pop_map.get(reg_name, {})
        pop_index = float(p_info.get("유동인구_지수", 75.0))
        tot_floating_pop = float(p_info.get("총_유동인구", 1000000.0))
        youth_pop_ratio = float(p_info.get("청년층_2030_유동비중", 0.34))
        night_pop_ratio = float(p_info.get("야간_17_24_유동비중", 0.32))

        for cat_name in STANDARD_CATEGORIES:
            sub = df[(df["권역(행정동)"] == reg_name) & (df["표준업종"] == cat_name)]
            review_store_cnt = len(sub)
            competition_ratio = review_store_cnt / reg_stores

            if review_store_cnt > 0:
                avg_rating = float(sub["신뢰도가중_평점"].mean())
                avg_sentiment = float(sub["리뷰_감성점수"].mean())
                total_cat_reviews = int(sub["리뷰 수"].sum())
                mean_log_reviews = float(sub["로그_리뷰수"].mean())
                center_lat = float(sub["위도"].mean())
                center_lng = float(sub["경도"].mean())
            else:
                avg_rating = global_mean_rating
                avg_sentiment = 0.55
                total_cat_reviews = 0
                mean_log_reviews = 0.0
                center_lat = r_lat
                center_lng = r_lng

            st_info = store_map.get((reg_name, cat_name), {})
            official_store_cnt = float(st_info.get("공식_점포수", max(review_store_cnt, 5)))
            franchise_ratio = float(st_info.get("프랜차이즈_비중", 0.18))
            open_rate = float(st_info.get("공식_개업률", 3.2))
            close_rate = float(st_info.get("공식_폐업률", 2.8))

            sl_info = sales_map.get((reg_name, cat_name), {})
            est_sales_idx = float(
                sl_info.get(
                    "추정매출_지수",
                    np.clip(pop_index * 0.65 + mean_log_reviews * 8.5, 42.0, 96.0),
                )
            )
            monthly_sales_m = float(sl_info.get("월평균_매출금액_백만원", 180.0))
            per_store_sales_10k = float(sl_info.get("점포당_월매출_만원", 1500.0))
            youth_sales_ratio = float(sl_info.get("연령대_2030_매출비중", youth_pop_ratio))

            demand_share = total_cat_reviews / reg_reviews

            # 상권 적합도 종합 산식 (공공데이터 유동인구 + 공식 추정매출 + 리뷰 감성/평점 + 개폐업 안정성 - 과포화 패널티)
            demand_vitality = (
                0.26 * (pop_index / 100.0)
                + 0.28 * (est_sales_idx / 100.0)
                + 0.16 * min(demand_share * 3.8, 1.0)
                + 0.14 * ((avg_rating - 3.5) / 1.5)
                + 0.10 * avg_sentiment
                + 0.06 * min(youth_sales_ratio * 2.0, 1.0)
            )
            survival_stability = np.clip((open_rate - close_rate) * 0.008, -0.05, 0.05)
            saturation_penalty = max(0.0, (competition_ratio - 0.17) * 0.55)
            blue_ocean_bonus = 0.05 if (0 < review_store_cnt <= 10 and avg_rating >= 4.30) else 0.0

            suitability_score = float(
                np.clip(
                    (demand_vitality + survival_stability - saturation_penalty + blue_ocean_bonus) * 100.0,
                    18.0,
                    97.5,
                )
            )

            feature_rows.append({
                "권역": reg_name,
                "표준업종": cat_name,
                "중심_위도": round(center_lat, 6),
                "중심_경도": round(center_lng, 6),
                "점포수": review_store_cnt,
                "공식_외식점포수": round(official_store_cnt, 1),
                "프랜차이즈_비중": round(franchise_ratio, 4),
                "공식_개업률": round(open_rate, 2),
                "공식_폐업률": round(close_rate, 2),
                "권역내_업종점유율": round(competition_ratio, 4),
                "평균_신뢰도평점": round(avg_rating, 3),
                "평균_리뷰감성점수": round(avg_sentiment, 3),
                "총_리뷰수": total_cat_reviews,
                "평균_로그리뷰수": round(mean_log_reviews, 4),
                "권역내_리뷰수요비중": round(demand_share, 4),
                "총_유동인구": round(tot_floating_pop, 0),
                "유동인구_지수": round(pop_index, 2),
                "청년층_2030_유동비중": round(youth_pop_ratio, 4),
                "야간_17_24_유동비중": round(night_pop_ratio, 4),
                "월평균_매출금액_백만원": round(monthly_sales_m, 2),
                "점포당_월매출_만원": round(per_store_sales_10k, 1),
                "연령대_2030_매출비중": round(youth_sales_ratio, 4),
                "추정매출_지수": round(est_sales_idx, 2),
                "상권적합도_점수": round(suitability_score, 2),
            })

    features_df = pd.DataFrame(feature_rows)

    Path(output_cleaned_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(output_features_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_cleaned_csv, index=False, encoding="utf-8-sig")
    features_df.to_csv(output_features_csv, index=False, encoding="utf-8-sig")

    return df, features_df


if __name__ == "__main__":
    upload_dir = "/working_dir/c_d0b86430e3c81b72/user_uploaded_files/ttl=63d/user_uploaded"
    default_excel = os.path.join(upload_dir, "성북구_음식점_최대수집_과제용.xlsx")
    pop_csv = os.path.join(upload_dir, "서울시 상권분석서비스(길단위인구-행정동).csv")
    store_csv = os.path.join(upload_dir, "서울시 상권분석서비스(점포-행정동).csv")
    sales_csv = os.path.join(upload_dir, "서울시 상권분석서비스(추정매출-행정동).csv")

    out_dir = "/working_dir/c_d0b86430e3c81b72/artifacts/file_generation/ttl=63d/output"
    cleaned_path = os.path.join(out_dir, "seongbuk_reviews_cleaned.csv")
    features_path = os.path.join(out_dir, "seongbuk_training_features.csv")

    cleaned_df, feat_df = run_preprocessing_pipeline(
        input_excel_path=default_excel,
        output_cleaned_csv=cleaned_path,
        output_features_csv=features_path,
        floating_pop_csv=pop_csv,
        store_csv=store_csv,
        estimated_sales_csv=sales_csv,
    )
    print(f"[전처리 완료] 정제된 성북구 음식점 수: {len(cleaned_df):,}개")
    print(f"[공공데이터 3종 병합 완료] (권역 x 업종) 학습 샘플 수: {len(feat_df):,}개, 피처 컬럼 수: {len(feat_df.columns)}개")
    print("\n[성북구 권역별 평균 유동인구 지수 및 점포당 월매출 요약]")
    summary = feat_df.groupby("권역")[["총_유동인구", "유동인구_지수", "점포당_월매출_만원", "상권적합도_점수"]].mean().round(1)
    print(summary.sort_values(by="상권적합도_점수", ascending=False))