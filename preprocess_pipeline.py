"""
DeepBite (딥바이트) - 서울시 전체 시계열 학습 데이터(18,000+샘플) 및 성북구 구글 리뷰 통합 전처리 파이프라인
Repository: https://github.com/hs-2191270-jeongseyoung/deep-bite

핵심 구조 (데이터 누수 제거 · 역할 분리):
1. [AI 학습용 진짜 정답 데이터] seoul_timeseries_training.csv
   - 서울시 전체 425개 행정동 x 10개 외식업종 x 6개 분기(2025Q1~2026Q2) 시계열 결합
   - 입력(X): t 분기의 유동인구(총/2030/야간/주말), 점포(점포수/프랜차이즈/개폐업률), 매출(총매출/점포당매출/객단가/2030·주말·야간 비중)
   - 정답(y): t+1 분기의 '실제 점포당 월매출(log)' 및 '실제 생존율(100 - 폐업률)' (가짜 공식 배제)
   - 검증 분할: 2025Q1~2026Q1 학습(train) -> 2026Q2 실제 성북구·서울시 데이터로 미래 예측 검증(val)
2. [성북구 양방향 추론 + 구글 리뷰 보정용] seongbuk_training_features.csv
   - 성북구 유효 음식점 2,261건(구글 리뷰 평점·감성·리뷰수요) + 성북구 최신(2026Q2) 공공데이터 결합
   - 성북구 12대 권역 x 9대 표준 업종(108개 상권 그리드) 추론 입력 테이블 생성
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SEONGBUK_REGIONS: Dict[str, Tuple[float, float]] = {
    "성북동": (37.5925, 126.9985),
    "삼선동": (37.5865, 127.0110),
    "동선동": (37.5928, 127.0175),
    "돈암동": (37.5975, 127.0165),
    "안암동": (37.5862, 127.0290),
    "보문동": (37.5815, 127.0195),
    "정릉동": (37.6065, 127.0075),
    "길음동": (37.6045, 127.0245),
    "종암동": (37.5990, 127.0335),
    "월곡동": (37.6048, 127.0425),
    "장위동": (37.6145, 127.0495),
    "석관동": (37.6110, 127.0605),
}

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

OFFICIAL_FOOD_CATS: List[str] = [
    "한식음식점",
    "중식음식점",
    "일식음식점",
    "양식음식점",
    "제과점",
    "패스트푸드점",
    "치킨전문점",
    "분식전문점",
    "호프-간이주점",
    "커피-음료",
]

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

STD_TO_PRIMARY_PUBLIC_CAT: Dict[str, str] = {
    "한식": "한식음식점",
    "카페/디저트": "커피-음료",
    "일식/돈까스/초밥": "일식음식점",
    "중식": "중식음식점",
    "양식/브런치": "양식음식점",
    "주점/바": "호프-간이주점",
    "분식/면류": "분식전문점",
    "치킨/호프": "치킨전문점",
    "아시안/해산물/기타": "한식음식점",
}

ROAD_TO_REGION: Dict[str, str] = {
    "성북로": "성북동", "대사관로": "성북동", "선잠로": "성북동", "창경궁로": "성북동", "혜화로": "성북동",
    "삼선교로": "삼선동", "보문로": "보문동", "고려대로": "안암동", "인촌로": "안암동",
    "개운사길": "안암동", "안암로": "안암동", "동소문로": "동선동", "아리랑로": "돈암동",
    "보국문로": "정릉동", "솔샘로": "정릉동", "정릉로": "정릉동", "길음로": "길음동",
    "삼양로": "길음동", "종암로": "종암동", "오패산로": "월곡동", "월곡로": "월곡동",
    "화랑로": "월곡동", "장월로": "월곡동", "돌곶이로": "장위동", "장위로": "장위동", "한천로": "석관동",
}

STANDARD_CATEGORIES: List[str] = [
    "한식", "카페/디저트", "일식/돈까스/초밥", "중식", "양식/브런치",
    "주점/바", "분식/면류", "치킨/호프", "아시안/해산물/기타",
]

NON_FOOD_KEYWORDS: List[str] = [
    "아파트", "래미안", "편한세상", "푸르지오", "아이파크", "주민센터",
    "도서관", "주유소", "세차", "정비", "부동산", "약국", "병원", "의원",
    "치과", "한의원", "초등학교", "중학교", "고등학교", "대학교", "어린이집",
    "유치원", "교회", "성당", "사찰", "우체국", "파출소", "지구대", "은행",
]

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
    addr = str(address)
    for kw, reg in [
        ("성북동", "성북동"), ("삼선동", "삼선동"), ("동선동", "동선동"), ("동소문동", "동선동"),
        ("돈암동", "돈암동"), ("안암동", "안암동"), ("보문동", "보문동"), ("정릉", "정릉동"),
        ("길음", "길음동"), ("종암", "종암동"), ("월곡", "월곡동"), ("장위", "장위동"), ("석관", "석관동"),
    ]:
        if kw in addr:
            return reg
    for road_kw, region_name in ROAD_TO_REGION.items():
        if road_kw in addr:
            if road_kw == "동소문로" and not pd.isna(lat) and lat >= 37.601:
                return "길음동"
            return region_name
    if not pd.isna(lat) and not pd.isna(lng):
        return min(
            SEONGBUK_REGIONS.keys(),
            key=lambda r: (lat - SEONGBUK_REGIONS[r][0]) ** 2 + (lng - SEONGBUK_REGIONS[r][1]) ** 2,
        )
    return "동선동"


def _match_category_from_text(text: str) -> Optional[str]:
    if any(k in text for k in ["카페", "커피", "디저트", "베이커리", "제과", "빵집", "케이크", "마카롱", "빙수", "티하우스", "와플", "도넛", "아이스크림", "로스터", "에스프레소", "찻집"]):
        return "카페/디저트"
    if any(k in text for k in ["치킨", "통닭", "닭강정", "호프", "비어"]):
        return "치킨/호프"
    if any(k in text for k in ["술집", "주점", "포차", "이자카야", "와인바", "칵테일", "펍", "생맥주", "막걸리", "전집", "오뎅바"]) or text.strip() in ["바"]:
        return "주점/바"
    if any(k in text for k in ["중식", "중국", "짜장", "짬뽕", "마라", "양꼬치", "탕수육", "딤섬", "훠궈", "중화"]):
        return "중식"
    if any(k in text for k in ["일식", "초밥", "스시", "라멘", "돈까스", "돈카츠", "우동", "소바", "사케동", "텐동", "샤브", "횟집", "오마카세", "일식당"]):
        return "일식/돈까스/초밥"
    if any(k in text for k in ["양식", "이탈리아", "이탈리안", "파스타", "피자", "스테이크", "브런치", "버거", "햄버거", "샌드위치", "샐러드", "비스트로", "타코", "멕시코", "멕시칸", "프랑스"]):
        return "양식/브런치"
    if any(k in text for k in ["분식", "떡볶이", "김밥", "순대", "칼국수", "국수", "수제비", "냉면", "만두", "라면", "쫄면"]):
        return "분식/면류"
    if any(k in text for k in ["쌀국수", "베트남", "태국", "인도", "커리", "카레", "아시안", "해물", "해산물", "조개", "장어"]):
        return "아시안/해산물/기타"
    if any(k in text for k in ["한식", "고기", "삼겹살", "갈비", "곱창", "막창", "족발", "보쌈", "국밥", "찌개", "순두부", "설렁탕", "곰탕", "감자탕", "해장국", "백반", "한정식", "비빔밥", "불고기", "닭갈비", "삼계탕", "찜닭", "아구찜", "생선구이", "육회", "반찬", "김치", "된장", "불백", "정식", "식당", "밥집"]):
        return "한식"
    return None


def standardize_category(raw_cat: str, name: str, review: str) -> str:
    c = str(raw_cat).strip() if not pd.isna(raw_cat) else ""
    n = str(name).strip() if not pd.isna(name) else ""
    r = str(review).strip() if not pd.isna(review) else ""
    if c and c not in ["음식점", "식당", "레스토랑", "음식"]:
        m = _match_category_from_text(c)
        if m:
            return m
    for txt in (n, r):
        m = _match_category_from_text(txt)
        if m:
            return m
    return "한식"


def compute_review_sentiment(review_text: str) -> float:
    if pd.isna(review_text) or not str(review_text).strip():
        return 0.55
    text = str(review_text)
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    return float(np.clip(0.58 + 0.09 * pos_hits - 0.14 * neg_hits, 0.10, 0.98))


def build_seoul_timeseries_dataset(
    floating_pop_csv: str,
    store_csv: str,
    estimated_sales_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    서울시 전체 425개 행정동 x 10개 외식업종 x 6개 분기(2025Q1~2026Q2) 공공데이터를 결합하여
    t 분기 피처 -> t+1 분기 실제 점포당 월매출 및 실제 생존율을 예측하는 시계열 학습 데이터셋을 구축합니다.
    """
    pop = _read_csv_auto(floating_pop_csv)
    st = _read_csv_auto(store_csv)
    sl = _read_csv_auto(estimated_sales_csv)

    # 외식업 10종 필터
    st = st[st["서비스_업종_코드_명"].isin(OFFICIAL_FOOD_CATS)].copy()
    sl = sl[sl["서비스_업종_코드_명"].isin(OFFICIAL_FOOD_CATS)].copy()

    # 유동인구 파생 지표
    pop["총_유동인구_수"] = pop["총_유동인구_수"].clip(lower=1)
    pop["청년층_2030_유동비중"] = (pop["연령대_20_유동인구_수"] + pop["연령대_30_유동인구_수"]) / pop["총_유동인구_수"]
    pop["야간_17_24_유동비중"] = (pop["시간대_17_21_유동인구_수"] + pop["시간대_21_24_유동인구_수"]) / pop["총_유동인구_수"]
    pop["주말_유동비중"] = (pop["토요일_유동인구_수"] + pop["일요일_유동인구_수"]) / pop["총_유동인구_수"]
    pop["총_유동인구_로그"] = np.log1p(pop["총_유동인구_수"])

    pop_slim = pop[[
        "기준_년분기_코드", "행정동_코드", "행정동_코드_명",
        "총_유동인구_수", "총_유동인구_로그", "청년층_2030_유동비중", "야간_17_24_유동비중", "주말_유동비중",
    ]].drop_duplicates(subset=["기준_년분기_코드", "행정동_코드"])

    # 점포 파생 지표
    st["전체_점포_수_safe"] = st["전체_점포_수"].clip(lower=1)
    st["전체_점포_수_로그"] = np.log1p(st["전체_점포_수"])
    st["프랜차이즈_비중"] = (st["프랜차이즈_점포_수"] / st["전체_점포_수_safe"]).clip(0.0, 1.0)
    st["생존율"] = (100.0 - st["폐업_률"].clip(0.0, 50.0))

    st_slim = st[[
        "기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명",
        "전체_점포_수", "전체_점포_수_safe", "전체_점포_수_로그",
        "프랜차이즈_비중", "개업_율", "폐업_률", "생존율",
    ]].drop_duplicates(subset=["기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명"])

    # 매출 파생 지표
    sl["당월_매출_금액_safe"] = sl["당월_매출_금액"].clip(lower=1)
    sl["당월_매출_건수_safe"] = sl["당월_매출_건수"].clip(lower=1)
    sl["당월_매출_로그"] = np.log1p(sl["당월_매출_금액_safe"])
    sl["건당_객단가_로그"] = np.log1p(sl["당월_매출_금액_safe"] / sl["당월_매출_건수_safe"])
    sl["주말_매출비중"] = (sl["주말_매출_금액"] / sl["당월_매출_금액_safe"]).clip(0.0, 1.0)
    sl["연령대_2030_매출비중"] = ((sl["연령대_20_매출_금액"] + sl["연령대_30_매출_금액"]) / sl["당월_매출_금액_safe"]).clip(0.0, 1.0)
    sl["야간_매출비중"] = ((sl["시간대_17~21_매출_금액"] + sl["시간대_21~24_매출_금액"]) / sl["당월_매출_금액_safe"]).clip(0.0, 1.0)

    sl_slim = sl[[
        "기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명",
        "당월_매출_금액", "당월_매출_로그", "건당_객단가_로그",
        "주말_매출비중", "연령대_2030_매출비중", "야간_매출비중",
    ]].drop_duplicates(subset=["기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명"])

    # 3종 공공데이터 병합 (기준_년분기_코드 x 행정동_코드 x 서비스_업종_코드_명)
    panel = sl_slim.merge(
        st_slim, on=["기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명"], how="inner"
    ).merge(
        pop_slim, on=["기준_년분기_코드", "행정동_코드"], how="inner"
    )

    # 분기당 점포당 월평균 매출 (추정매출은 분기 합계이므로 3개월 환산 or 분기 기준 일관화 -> 만원 단위)
    panel["점포당_매출_만원"] = (panel["당월_매출_금액"] / panel["전체_점포_수_safe"]) / 1e4
    panel["점포당_매출_로그"] = np.log1p(panel["점포당_매출_만원"].clip(lower=1.0))

    # 행정동 내 업종 점유율 (경쟁도)
    dong_tot_stores = panel.groupby(["기준_년분기_코드", "행정동_코드"])["전체_점포_수"].transform("sum").clip(lower=1)
    panel["동내_업종점유율"] = (panel["전체_점포_수"] / dong_tot_stores).clip(0.0, 1.0)

    # 분기 계절성 (1~4분기)
    q_num = panel["기준_년분기_코드"] % 10
    panel["분기_sin"] = np.sin(2 * np.pi * q_num / 4.0)
    panel["분기_cos"] = np.cos(2 * np.pi * q_num / 4.0)
    panel["is_seongbuk"] = panel["행정동_코드"].astype(str).str.startswith("11290").astype(int)

    # t -> t+1 시계열 페어 생성
    quarters = sorted(panel["기준_년분기_코드"].unique().tolist())
    q_next_map = {quarters[i]: quarters[i + 1] for i in range(len(quarters) - 1)}

    curr = panel.copy()
    curr["target_quarter"] = curr["기준_년분기_코드"].map(q_next_map)
    curr = curr.dropna(subset=["target_quarter"]).copy()
    curr["target_quarter"] = curr["target_quarter"].astype(int)

    nxt = panel[[
        "기준_년분기_코드", "행정동_코드", "서비스_업종_코드_명",
        "점포당_매출_만원", "점포당_매출_로그", "생존율", "폐업_률",
    ]].rename(columns={
        "기준_년분기_코드": "target_quarter",
        "점포당_매출_만원": "target_점포당_매출_만원",
        "점포당_매출_로그": "target_점포당_매출_로그",
        "생존율": "target_생존율",
        "폐업_률": "target_폐업률",
    })

    ts_df = curr.merge(nxt, on=["target_quarter", "행정동_코드", "서비스_업종_코드_명"], how="inner")
    latest_q = max(quarters)
    ts_df["split"] = np.where(ts_df["target_quarter"] == latest_q, "val", "train")

    return ts_df, panel


def run_preprocessing_pipeline(
    input_excel_path: str,
    floating_pop_csv: str,
    store_csv: str,
    estimated_sales_csv: str,
    output_cleaned_csv: str,
    output_timeseries_csv: str,
    output_features_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. 성북구 구글 리뷰 엑셀 정제 (seongbuk_reviews_cleaned.csv)
    2. 서울시 전체 425개 행정동 시계열 학습 데이터 생성 (seoul_timeseries_training.csv)
    3. 성북구 12대 권역 x 9대 업종 최신(2026Q2) 추론 + 구글 리뷰 피처 테이블 생성 (seongbuk_training_features.csv)
    """
    # 1) 구글 리뷰 정제
    raw_df = pd.read_excel(input_excel_path)
    df = raw_df[raw_df.apply(is_valid_restaurant, axis=1)].copy()
    df = df.drop_duplicates(subset=["음식점 이름", "위도", "경도"]).reset_index(drop=True)
    df["권역(행정동)"] = df.apply(lambda r: map_seongbuk_region(r["주소"], r["위도"], r["경도"]), axis=1)
    df["표준업종"] = df.apply(
        lambda r: standardize_category(r["판매 품목(카테고리)"], r["음식점 이름"], r["핵심 리뷰"]), axis=1
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

    # 2) 서울시 전체 시계열 학습 데이터 구축
    ts_df, panel = build_seoul_timeseries_dataset(
        floating_pop_csv=floating_pop_csv,
        store_csv=store_csv,
        estimated_sales_csv=estimated_sales_csv,
    )

    # 3) 성북구 최신 분기(2026Q2) 공공데이터 패널 + 구글 리뷰 결합 (12권역 x 9업종 = 108개 그리드)
    latest_q = int(panel["기준_년분기_코드"].max())
    sb_latest = panel[(panel["is_seongbuk"] == 1) & (panel["기준_년분기_코드"] == latest_q)].copy()
    sb_latest["권역"] = sb_latest["행정동_코드_명"].map(DONG_TO_REGION)
    sb_latest = sb_latest.dropna(subset=["권역"])

    # 표준업종별 확장 매핑
    exp_rows = []
    for _, r in sb_latest.iterrows():
        for std_cat in PUBLIC_CAT_TO_STD.get(str(r["서비스_업종_코드_명"]), []):
            exp_rows.append({
                "권역": r["권역"],
                "표준업종": std_cat,
                "대표_공식업종": STD_TO_PRIMARY_PUBLIC_CAT[std_cat],
                "총_유동인구_수": r["총_유동인구_수"],
                "총_유동인구_로그": r["총_유동인구_로그"],
                "청년층_2030_유동비중": r["청년층_2030_유동비중"],
                "야간_17_24_유동비중": r["야간_17_24_유동비중"],
                "주말_유동비중": r["주말_유동비중"],
                "전체_점포_수": r["전체_점포_수"],
                "전체_점포_수_로그": r["전체_점포_수_로그"],
                "프랜차이즈_비중": r["프랜차이즈_비중"],
                "개업_율": r["개업_율"],
                "폐업_률": r["폐업_률"],
                "당월_매출_로그": r["당월_매출_로그"],
                "점포당_매출_만원": r["점포당_매출_만원"],
                "점포당_매출_로그": r["점포당_매출_로그"],
                "건당_객단가_로그": r["건당_객단가_로그"],
                "주말_매출비중": r["주말_매출비중"],
                "연령대_2030_매출비중": r["연령대_2030_매출비중"],
                "야간_매출비중": r["야간_매출비중"],
                "동내_업종점유율": r["동내_업종점유율"],
                "분기_sin": r["분기_sin"],
                "분기_cos": r["분기_cos"],
            })
    sb_agg = pd.DataFrame(exp_rows).groupby(["권역", "표준업종", "대표_공식업종"], as_index=False).mean()
    sb_lookup = {(r["권역"], r["표준업종"]): r for _, r in sb_agg.iterrows()}
    sb_reg_fallback = sb_agg.groupby("권역", as_index=False).mean(numeric_only=True).set_index("권역")

    region_total_stores = df.groupby("권역(행정동)").size().to_dict()
    region_total_reviews = df.groupby("권역(행정동)")["리뷰 수"].sum().to_dict()

    feature_rows: List[Dict[str, object]] = []
    for reg_name, (r_lat, r_lng) in SEONGBUK_REGIONS.items():
        reg_stores = max(region_total_stores.get(reg_name, 1), 1)
        reg_reviews = max(region_total_reviews.get(reg_name, 1), 1)
        fb = sb_reg_fallback.loc[reg_name]

        for cat_name in STANDARD_CATEGORIES:
            sub = df[(df["권역(행정동)"] == reg_name) & (df["표준업종"] == cat_name)]
            rev_cnt = len(sub)
            comp_ratio = rev_cnt / reg_stores
            if rev_cnt > 0:
                avg_rating = float(sub["신뢰도가중_평점"].mean())
                avg_sent = float(sub["리뷰_감성점수"].mean())
                tot_rev = int(sub["리뷰 수"].sum())
                mean_log_rev = float(sub["로그_리뷰수"].mean())
                c_lat = float(sub["위도"].mean())
                c_lng = float(sub["경도"].mean())
            else:
                avg_rating = global_mean_rating
                avg_sent = 0.55
                tot_rev = 0
                mean_log_rev = 0.0
                c_lat, c_lng = r_lat, r_lng

            demand_share = tot_rev / reg_reviews
            # 성북구 구글 리뷰 기반 로컬 보정 점수 (0~100점, 추론 단계에서 AI 예측과 30% 비중 결합)
            review_local_score = float(np.clip(
                (0.45 * ((avg_rating - 3.6) / 1.4) + 0.35 * avg_sent + 0.20 * min(demand_share * 4.0, 1.0)) * 100.0,
                20.0, 96.0
            ))

            pub = sb_lookup.get((reg_name, cat_name), fb)
            feature_rows.append({
                "권역": reg_name,
                "표준업종": cat_name,
                "서비스_업종_코드_명": STD_TO_PRIMARY_PUBLIC_CAT[cat_name],
                "중심_위도": round(c_lat, 6),
                "중심_경도": round(c_lng, 6),
                "점포수": rev_cnt,
                "권역내_업종점유율": round(comp_ratio, 4),
                "평균_신뢰도평점": round(avg_rating, 3),
                "평균_리뷰감성점수": round(avg_sent, 3),
                "총_리뷰수": tot_rev,
                "평균_로그리뷰수": round(mean_log_rev, 4),
                "권역내_리뷰수요비중": round(demand_share, 4),
                "구글리뷰_로컬점수": round(review_local_score, 2),
                # 서울시 AI 모델 입력과 동일한 16개 공공데이터 피처
                "총_유동인구": round(float(pub["총_유동인구_수"]), 0),
                "총_유동인구_로그": round(float(pub["총_유동인구_로그"]), 4),
                "청년층_2030_유동비중": round(float(pub["청년층_2030_유동비중"]), 4),
                "야간_17_24_유동비중": round(float(pub["야간_17_24_유동비중"]), 4),
                "주말_유동비중": round(float(pub["주말_유동비중"]), 4),
                "공식_외식점포수": round(float(pub["전체_점포_수"]), 1),
                "전체_점포_수_로그": round(float(pub["전체_점포_수_로그"]), 4),
                "프랜차이즈_비중": round(float(pub["프랜차이즈_비중"]), 4),
                "공식_개업률": round(float(pub["개업_율"]), 2),
                "공식_폐업률": round(float(pub["폐업_률"]), 2),
                "당월_매출_로그": round(float(pub["당월_매출_로그"]), 4),
                "점포당_월매출_만원": round(float(pub["점포당_매출_만원"]), 1),
                "점포당_매출_로그": round(float(pub["점포당_매출_로그"]), 4),
                "건당_객단가_로그": round(float(pub["건당_객단가_로그"]), 4),
                "주말_매출비중": round(float(pub["주말_매출비중"]), 4),
                "연령대_2030_매출비중": round(float(pub["연령대_2030_매출비중"]), 4),
                "야간_매출비중": round(float(pub["야간_매출비중"]), 4),
                "동내_업종점유율": round(float(pub["동내_업종점유율"]), 4),
                "분기_sin": round(float(pub["분기_sin"]), 4),
                "분기_cos": round(float(pub["분기_cos"]), 4),
            })

    features_df = pd.DataFrame(feature_rows)

    for p in (output_cleaned_csv, output_timeseries_csv, output_features_csv):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_cleaned_csv, index=False, encoding="utf-8-sig")
    ts_df.to_csv(output_timeseries_csv, index=False, encoding="utf-8-sig")
    features_df.to_csv(output_features_csv, index=False, encoding="utf-8-sig")

    return df, ts_df, features_df


if __name__ == "__main__":
    upload_dir = "/working_dir/c_d0b86430e3c81b72/user_uploaded_files/ttl=63d/user_uploaded"
    out_dir = "/working_dir/c_d0b86430e3c81b72/artifacts/file_generation/ttl=63d/output"

    cleaned_df, ts_df, feat_df = run_preprocessing_pipeline(
        input_excel_path=os.path.join(upload_dir, "성북구_음식점_최대수집_과제용.xlsx"),
        floating_pop_csv=os.path.join(upload_dir, "서울시 상권분석서비스(길단위인구-행정동).csv"),
        store_csv=os.path.join(upload_dir, "서울시 상권분석서비스(점포-행정동).csv"),
        estimated_sales_csv=os.path.join(upload_dir, "서울시 상권분석서비스(추정매출-행정동).csv"),
        output_cleaned_csv=os.path.join(out_dir, "seongbuk_reviews_cleaned.csv"),
        output_timeseries_csv=os.path.join(out_dir, "seoul_timeseries_training.csv"),
        output_features_csv=os.path.join(out_dir, "seongbuk_training_features.csv"),
    )
    train_n = int((ts_df["split"] == "train").sum())
    val_n = int((ts_df["split"] == "val").sum())
    sb_val_n = int(((ts_df["split"] == "val") & (ts_df["is_seongbuk"] == 1)).sum())
    print(f"[전처리 완료] 성북구 정제 음식점: {len(cleaned_df):,}건")
    print(f"[시계열 학습셋] 전체: {len(ts_df):,}개 | Train(25Q1~26Q1): {train_n:,}개 | Val(26Q2 전체): {val_n:,}개 (성북구 26Q2: {sb_val_n:,}개)")
    print(f"[성북구 추론 그리드] {len(feat_df):,}개 (12권역 x 9업종)")