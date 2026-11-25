"""
DeepBite (딥바이트) - 서울시 18,000+ 시계열 실데이터 기반 양방향 PyTorch AI 상권 적합도 모델
Repository: https://github.com/hs-2191270-jeongseyoung/deep-bite

핵심 메커니즘 (진짜 정답 데이터 학습 + 성북구 구글 리뷰 결합):
1. [학습 단계 - 가짜 공식 제거]
   - 서울시 전체 425개 행정동 x 10개 외식업종 시계열 데이터(Train: 2025Q1~2026Q1 약 14,600건)로
     Multi-Task PyTorch 신경망(`DeepBiteMarketForecastNet`)을 학습합니다.
   - 예측 타겟(실제 정답):
     (1) 다음 분기 실제 점포당 월매출 (log scale)
     (2) 다음 분기 실제 점포 생존율 (100 - 실제 폐업률)
2. [객관적 미래 검증]
   - 학습에 전혀 쓰지 않은 미래 시점(2026년 2분기) 서울시 전체(~3,600건) 및 성북구(161건) 실제 데이터로
     Validation R2, RMSE, MAE를 산출해 모델의 일반화 성능을 증명합니다.
3. [양방향 추론 + 성북구 구글 리뷰 결합 + 기존 main.py 100% 호환]
   - 기존 main.py 드롭다운 업종명('까페', '중국식', '경양식', '호프/통닭', '통닭(치킨)', '분식', '일식', '한식')
     및 법정동명('하월곡동' -> '월곡동') 자동 매핑 지원.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


TS_NUMERIC_FEATURE_COLS: List[str] = [
    "총_유동인구_로그",
    "청년층_2030_유동비중",
    "야간_17_24_유동비중",
    "주말_유동비중",
    "전체_점포_수_로그",
    "프랜차이즈_비중",
    "개업_율",
    "폐업_률",
    "당월_매출_로그",
    "점포당_매출_로그",
    "건당_객단가_로그",
    "주말_매출비중",
    "연령대_2030_매출비중",
    "야간_매출비중",
    "동내_업종점유율",
    "분기_sin",
    "분기_cos",
]

SEONGBUK_COL_ALIAS: Dict[str, str] = {
    "개업_율": "공식_개업률",
    "폐업_률": "공식_폐업률",
}

OFFICIAL_FOOD_CATS: List[str] = [
    "한식음식점", "중식음식점", "일식음식점", "양식음식점", "제과점",
    "패스트푸드점", "치킨전문점", "분식전문점", "호프-간이주점", "커피-음료",
]

# 기존 main.py 업태명 / 드롭다운 명칭 -> 표준 업종 1:1 매핑 사전
INDUSTRY_ALIAS_MAP: Dict[str, str] = {
    "한식": "한식",
    "분식": "분식/면류",
    "분식/면류": "분식/면류",
    "호프/통닭": "주점/바",
    "주점/바": "주점/바",
    "주점": "주점/바",
    "까페": "카페/디저트",
    "카페": "카페/디저트",
    "카페/디저트": "카페/디저트",
    "커피숍": "카페/디저트",
    "일식": "일식/돈까스/초밥",
    "일식/돈까스/초밥": "일식/돈까스/초밥",
    "중국식": "중식",
    "중식": "중식",
    "경양식": "양식/브런치",
    "양식": "양식/브런치",
    "양식/브런치": "양식/브런치",
    "통닭(치킨)": "치킨/호프",
    "치킨": "치킨/호프",
    "치킨/호프": "치킨/호프",
    "패스트푸드": "양식/브런치",
    "아시안/해산물/기타": "아시안/해산물/기타",
}

# 기존 main.py 법정동명 -> 권역명 매핑
REGION_ALIAS_MAP: Dict[str, str] = {
    "하월곡동": "월곡동",
    "상월곡동": "월곡동",
    "동소문동": "동선동",
}


def resolve_category_name(category: str, available_categories: List[str]) -> str:
    """기존 main.py의 '까페', '중국식', '경양식', '통닭(치킨)' 등을 표준 업종명으로 정확히 변환합니다."""
    key = str(category).strip()
    if key in INDUSTRY_ALIAS_MAP and INDUSTRY_ALIAS_MAP[key] in available_categories:
        return INDUSTRY_ALIAS_MAP[key]
    for c in available_categories:
        if key in c or c in key:
            return c
    return "한식"


def resolve_region_name(region: str, available_regions: List[str]) -> str:
    """기존 main.py의 '하월곡동' 등을 모델 권역명('월곡동')으로 정확히 변환합니다."""
    key = str(region).strip()
    if key in REGION_ALIAS_MAP:
        key = REGION_ALIAS_MAP[key]
    if key in available_regions:
        return key
    for r in available_regions:
        if key[:2] in r:
            return r
    return "동선동"


if TORCH_AVAILABLE:
    class DeepBiteMarketForecastNet(nn.Module):
        """
        업종 임베딩(10개 외식업종 -> 8차원) + 17개 시계열 상권 수치 피처를 결합해
        다음 분기 [실제 점포당 월매출(log), 실제 점포 생존율]을 동시 예측하는 Multi-Task PyTorch 신경망
        """

        def __init__(
            self,
            num_categories: int = len(OFFICIAL_FOOD_CATS),
            num_numeric_features: int = len(TS_NUMERIC_FEATURE_COLS),
            cat_emb_dim: int = 8,
        ) -> None:
            super().__init__()
            self.cat_emb = nn.Embedding(num_categories, cat_emb_dim)
            in_dim = cat_emb_dim + num_numeric_features
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            self.head = nn.Linear(16, 2)

        def forward(self, cat_idx: "torch.Tensor", numeric_x: "torch.Tensor") -> "torch.Tensor":
            c_e = self.cat_emb(cat_idx)
            h = self.encoder(torch.cat([c_e, numeric_x], dim=-1))
            return self.head(h)


@dataclass
class TrainingArtifacts:
    """학습된 모델 가중치, 정규화 통계 및 2026Q2 미래 검증 성능 컨테이너"""
    regions: List[str]
    categories: List[str]
    official_categories: List[str]
    feature_cols: List[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    train_samples: int
    val_seoul_samples: int
    val_seongbuk_samples: int
    seoul_val_sales_r2: float
    seoul_val_sales_rmse: float
    seongbuk_val_sales_r2: float
    seongbuk_val_sales_rmse: float
    seongbuk_val_survival_mae: float
    weights: Dict[str, np.ndarray]
    torch_model: Optional[object] = None


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-8))
    return round(r2, 4), round(rmse, 4), round(mae, 4)


def _resolve_timeseries_path(features_csv_path: str, timeseries_csv_path: Optional[str]) -> str:
    if timeseries_csv_path and os.path.exists(timeseries_csv_path):
        return timeseries_csv_path
    base_dir = os.path.dirname(os.path.abspath(features_csv_path))
    candidates = [
        os.path.join(base_dir, "seoul_timeseries_training.csv"),
        os.path.join(os.path.dirname(base_dir), "seoul_timeseries_training.csv"),
        "seoul_timeseries_training.csv",
        os.path.join("data", "seoul_timeseries_training.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("seoul_timeseries_training.csv 파일을 찾을 수 없습니다.")


def _predict_with_artifacts(
    cat_idx: np.ndarray,
    norm_x: np.ndarray,
    artifacts: TrainingArtifacts,
) -> Tuple[np.ndarray, np.ndarray]:
    if TORCH_AVAILABLE and artifacts.torch_model is not None:
        model = artifacts.torch_model
        model.eval()
        with torch.no_grad():
            preds_norm = model(
                torch.tensor(cat_idx, dtype=torch.long),
                torch.tensor(norm_x, dtype=torch.float32),
            ).numpy()
    else:
        c_oh = np.eye(len(artifacts.official_categories), dtype=np.float32)[cat_idx]
        full_x = np.concatenate([c_oh, norm_x], axis=1)
        w = artifacts.weights
        a1 = _relu(full_x @ w["w1"] + w["b1"])
        a2 = _relu(a1 @ w["w2"] + w["b2"])
        preds_norm = a2 @ w["w3"] + w["b3"]

    preds = preds_norm * artifacts.target_std + artifacts.target_mean
    return preds[:, 0], preds[:, 1]


def train_bidirectional_model(
    features_csv_path: str,
    model_save_dir: str = "models",
    timeseries_csv_path: Optional[str] = None,
    epochs: int = 80,
    lr: float = 0.005,
    batch_size: int = 256,
    seed: int = 42,
) -> TrainingArtifacts:
    np.random.seed(seed)
    ts_path = _resolve_timeseries_path(features_csv_path, timeseries_csv_path)
    ts_df = pd.read_csv(ts_path)
    sb_grid = pd.read_csv(features_csv_path)

    cat_to_idx = {c: i for i, c in enumerate(OFFICIAL_FOOD_CATS)}
    ts_df = ts_df[ts_df["서비스_업종_코드_명"].isin(cat_to_idx)].copy()

    train_df = ts_df[ts_df["split"] == "train"].reset_index(drop=True)
    val_df = ts_df[ts_df["split"] == "val"].reset_index(drop=True)
    sb_val_df = val_df[val_df["is_seongbuk"] == 1].reset_index(drop=True)

    x_train_raw = train_df[TS_NUMERIC_FEATURE_COLS].values.astype(np.float32)
    x_val_raw = val_df[TS_NUMERIC_FEATURE_COLS].values.astype(np.float32)
    x_sb_val_raw = sb_val_df[TS_NUMERIC_FEATURE_COLS].values.astype(np.float32)

    feat_mean = x_train_raw.mean(axis=0)
    feat_std = np.where(x_train_raw.std(axis=0) < 1e-6, 1.0, x_train_raw.std(axis=0))

    x_train = (x_train_raw - feat_mean) / feat_std
    x_val = (x_val_raw - feat_mean) / feat_std
    x_sb_val = (x_sb_val_raw - feat_mean) / feat_std

    c_train = train_df["서비스_업종_코드_명"].map(cat_to_idx).values.astype(np.int64)
    c_val = val_df["서비스_업종_코드_명"].map(cat_to_idx).values.astype(np.int64)
    c_sb_val = sb_val_df["서비스_업종_코드_명"].map(cat_to_idx).values.astype(np.int64)

    y_train_raw = train_df[["target_점포당_매출_로그", "target_생존율"]].values.astype(np.float32)
    y_val_raw = val_df[["target_점포당_매출_로그", "target_생존율"]].values.astype(np.float32)
    y_sb_val_raw = sb_val_df[["target_점포당_매출_로그", "target_생존율"]].values.astype(np.float32)

    target_mean = y_train_raw.mean(axis=0)
    target_std = np.where(y_train_raw.std(axis=0) < 1e-6, 1.0, y_train_raw.std(axis=0))
    y_train_norm = (y_train_raw - target_mean) / target_std

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    torch_model = None
    weights_dict: Dict[str, np.ndarray] = {}

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        model = DeepBiteMarketForecastNet(
            num_categories=len(OFFICIAL_FOOD_CATS),
            num_numeric_features=len(TS_NUMERIC_FEATURE_COLS),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.tensor(c_train, dtype=torch.long),
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train_norm, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model.train()
        for _ in range(epochs):
            for b_c, b_x, b_y in loader:
                optimizer.zero_grad()
                out = model(b_c, b_x)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

        torch_model = model
        pt_path = os.path.join(model_save_dir, "deepbite_suitability_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "official_categories": OFFICIAL_FOOD_CATS,
                "feature_cols": TS_NUMERIC_FEATURE_COLS,
                "feat_mean": feat_mean.tolist(),
                "feat_std": feat_std.tolist(),
                "target_mean": target_mean.tolist(),
                "target_std": target_std.tolist(),
            },
            pt_path,
        )
    else:
        c_oh = np.eye(len(OFFICIAL_FOOD_CATS), dtype=np.float32)[c_train]
        full_x = np.concatenate([c_oh, x_train], axis=1)

        in_dim = full_x.shape[1]
        h1_dim, h2_dim = 48, 24
        w1 = np.random.randn(in_dim, h1_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        b1 = np.zeros((1, h1_dim), dtype=np.float32)
        w2 = np.random.randn(h1_dim, h2_dim).astype(np.float32) * np.sqrt(2.0 / h1_dim)
        b2 = np.zeros((1, h2_dim), dtype=np.float32)
        w3 = np.random.randn(h2_dim, 2).astype(np.float32) * np.sqrt(1.0 / h2_dim)
        b3 = np.zeros((1, 2), dtype=np.float32)

        params = [w1, b1, w2, b2, w3, b3]
        m_state = [np.zeros_like(p) for p in params]
        v_state = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        n_train = len(full_x)

        for _ in range(epochs):
            perm = np.random.permutation(n_train)
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                bx = full_x[idx]
                by = y_train_norm[idx]
                bs = float(len(idx))
                step += 1

                z1 = bx @ w1 + b1
                a1 = _relu(z1)
                z2 = a1 @ w2 + b2
                a2 = _relu(z2)
                out = a2 @ w3 + b3

                dout = (2.0 / (bs * 2.0)) * (out - by)
                dw3 = a2.T @ dout + 1e-4 * w3
                db3 = dout.sum(axis=0, keepdims=True)

                da2 = dout @ w3.T
                dz2 = da2 * (z2 > 0)
                dw2 = a1.T @ dz2 + 1e-4 * w2
                db2 = dz2.sum(axis=0, keepdims=True)

                da1 = dz2 @ w2.T
                dz1 = da1 * (z1 > 0)
                dw1 = bx.T @ dz1 + 1e-4 * w1
                db1 = dz1.sum(axis=0, keepdims=True)

                grads = [dw1, db1, dw2, db2, dw3, db3]
                for i, (p, g) in enumerate(zip(params, grads)):
                    m_state[i] = beta1 * m_state[i] + (1.0 - beta1) * g
                    v_state[i] = beta2 * v_state[i] + (1.0 - beta2) * (g * g)
                    m_hat = m_state[i] / (1.0 - beta1 ** step)
                    v_hat = v_state[i] / (1.0 - beta2 ** step)
                    p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        weights_dict = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}

    regions = sorted(sb_grid["권역"].unique().tolist())
    categories = sorted(sb_grid["표준업종"].unique().tolist())

    temp_art = TrainingArtifacts(
        regions=regions,
        categories=categories,
        official_categories=OFFICIAL_FOOD_CATS,
        feature_cols=TS_NUMERIC_FEATURE_COLS,
        feat_mean=feat_mean,
        feat_std=feat_std,
        target_mean=target_mean,
        target_std=target_std,
        train_samples=len(train_df),
        val_seoul_samples=len(val_df),
        val_seongbuk_samples=len(sb_val_df),
        seoul_val_sales_r2=0.0,
        seoul_val_sales_rmse=0.0,
        seongbuk_val_sales_r2=0.0,
        seongbuk_val_sales_rmse=0.0,
        seongbuk_val_survival_mae=0.0,
        weights=weights_dict,
        torch_model=torch_model,
    )

    seoul_pred_sales, _ = _predict_with_artifacts(c_val, x_val, temp_art)
    sb_pred_sales, sb_pred_surv = _predict_with_artifacts(c_sb_val, x_sb_val, temp_art)

    s_r2, s_rmse, _ = _eval_regression(y_val_raw[:, 0], seoul_pred_sales)
    sb_r2, sb_rmse, _ = _eval_regression(y_sb_val_raw[:, 0], sb_pred_sales)
    _, _, sb_surv_mae = _eval_regression(y_sb_val_raw[:, 1], sb_pred_surv)

    temp_art.seoul_val_sales_r2 = s_r2
    temp_art.seoul_val_sales_rmse = s_rmse
    temp_art.seongbuk_val_sales_r2 = sb_r2
    temp_art.seongbuk_val_sales_rmse = sb_rmse
    temp_art.seongbuk_val_survival_mae = sb_surv_mae

    npz_path = os.path.join(model_save_dir, "deepbite_model_weights.npz")
    np.savez(
        npz_path,
        regions=np.array(regions),
        categories=np.array(categories),
        official_categories=np.array(OFFICIAL_FOOD_CATS),
        feature_cols=np.array(TS_NUMERIC_FEATURE_COLS),
        feat_mean=feat_mean,
        feat_std=feat_std,
        target_mean=target_mean,
        target_std=target_std,
        **weights_dict,
    )

    return temp_art


class DeepBiteRecommender:
    """
    양방향 상권 적합도 추론 엔진 (기존 main.py 업종명 '까페', '중국식', '경양식', '통닭(치킨)' 등 100% 호환)
    """

    def __init__(self, features_csv_path: str, artifacts: TrainingArtifacts) -> None:
        self.df = pd.read_csv(features_csv_path).copy()
        self.artifacts = artifacts

        cat_to_idx = {c: i for i, c in enumerate(artifacts.official_categories)}
        c_idx = self.df["서비스_업종_코드_명"].map(cat_to_idx).fillna(0).values.astype(np.int64)

        cols_mapped = [SEONGBUK_COL_ALIAS.get(c, c) for c in artifacts.feature_cols]
        raw_x = self.df[cols_mapped].values.astype(np.float32)
        norm_x = (raw_x - artifacts.feat_mean) / artifacts.feat_std

        pred_sales_log, pred_survival = _predict_with_artifacts(c_idx, norm_x, artifacts)
        pred_sales_10k = np.expm1(np.clip(pred_sales_log, 0.0, 12.0))
        pred_survival = np.clip(pred_survival, 50.0, 100.0)

        s_min, s_max = float(pred_sales_log.min()), float(pred_sales_log.max())
        sales_score = 45.0 + 50.0 * (pred_sales_log - s_min) / max(s_max - s_min, 1e-6)
        surv_score = np.clip((pred_survival - 88.0) / 12.0 * 100.0, 35.0, 98.0)
        ai_forecast_score = 0.72 * sales_score + 0.28 * surv_score

        review_score = self.df["구글리뷰_로컬점수"].values.astype(np.float32)
        final_score = np.clip(0.70 * ai_forecast_score + 0.30 * review_score, 20.0, 98.0)

        self.df["AI_예측_차기분기_점포당매출_만원"] = np.round(pred_sales_10k, 1)
        self.df["AI_예측_생존율_퍼센트"] = np.round(pred_survival, 2)
        self.df["AI_예측_상권점수"] = np.round(ai_forecast_score, 2)
        self.df["AI_예측_적합도"] = np.round(final_score, 2)

    def find_nearest_region(self, lat: float, lng: float) -> str:
        reg_coords = self.df.groupby("권역")[["중심_위도", "중심_경도"]].mean()
        dists = (reg_coords["중심_위도"] - lat) ** 2 + (reg_coords["중심_경도"] - lng) ** 2
        return str(dists.idxmin())

    def get_pair_score(self, region: str, category: str) -> Optional[Dict[str, object]]:
        """특정 (권역, 업종) 쌍의 AI 예측 점수·매출·생존율·리뷰 평점을 조회합니다."""
        resolved_reg = resolve_region_name(region, self.artifacts.regions)
        resolved_cat = resolve_category_name(category, self.artifacts.categories)
        sub = self.df[(self.df["권역"] == resolved_reg) & (self.df["표준업종"] == resolved_cat)]
        if sub.empty:
            return None
        row = sub.iloc[0]
        return {
            "권역": resolved_reg,
            "표준업종": resolved_cat,
            "AI_상권적합도_점수": round(float(row["AI_예측_적합도"]), 1),
            "AI_예측_차기분기_점포당매출_만원": round(float(row["AI_예측_차기분기_점포당매출_만원"]), 1),
            "AI_예측_생존율_퍼센트": round(float(row["AI_예측_생존율_퍼센트"]), 1),
            "평균_신뢰도평점": round(float(row["평균_신뢰도평점"]), 2),
        }

    def recommend_categories_for_region(
        self,
        region: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """[방향 1: 지역 선택 -> 최적 업종 추천]"""
        if region:
            target_region = resolve_region_name(region, self.artifacts.regions)
        elif lat is not None and lng is not None:
            target_region = self.find_nearest_region(lat, lng)
        else:
            target_region = "동선동"

        sub = self.df[self.df["권역"] == target_region].sort_values(
            by="AI_예측_적합도", ascending=False
        )
        results: List[Dict[str, object]] = []
        for rank, (_, row) in enumerate(sub.head(top_k).iterrows(), start=1):
            score = round(float(row["AI_예측_적합도"]), 2)
            sales = round(float(row["AI_예측_차기분기_점포당매출_만원"]), 1)
            surv = round(float(row["AI_예측_생존율_퍼센트"]), 2)
            rating = round(float(row["평균_신뢰도평점"]), 2)
            results.append({
                "순위": rank,
                "권역": row["권역"],
                "추천_업종": row["표준업종"],
                "AI_상권적합도_점수": score,
                "AI_예측_차기분기_점포당매출_만원": sales,
                "AI_예측_생존율_퍼센트": surv,
                "구글리뷰_로컬점수": round(float(row["구글리뷰_로컬점수"]), 2),
                "공식_외식점포수": round(float(row["공식_외식점포수"]), 1),
                "평균_신뢰도평점": rating,
                "추천_분석코멘트": (
                    f"{row['권역']} '{row['표준업종']}' 업종은 AI 예측 다음 분기 점포당 매출 "
                    f"{sales:,.0f}만 원·생존율 {surv:.1f}%이며, "
                    f"성북구 구글 리뷰 평점({rating:.2f})을 반영한 종합 적합도가 우수합니다."
                ),
            })
        return results

    def recommend_regions_for_category(
        self,
        category: str,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """[방향 2: 업종 선택 -> 최적 지역 추천]"""
        matched_cat = resolve_category_name(category, self.artifacts.categories)
        sub = self.df[self.df["표준업종"] == matched_cat].sort_values(
            by="AI_예측_적합도", ascending=False
        )
        results: List[Dict[str, object]] = []
        for rank, (_, row) in enumerate(sub.head(top_k).iterrows(), start=1):
            score = round(float(row["AI_예측_적합도"]), 2)
            sales = round(float(row["AI_예측_차기분기_점포당매출_만원"]), 1)
            surv = round(float(row["AI_예측_생존율_퍼센트"]), 2)
            rating = round(float(row["평균_신뢰도평점"]), 2)
            results.append({
                "순위": rank,
                "희망_업종": matched_cat,
                "추천_권역": row["권역"],
                "AI_상권적합도_점수": score,
                "중심_위도": round(float(row["중심_위도"]), 6),
                "중심_경도": round(float(row["중심_경도"]), 6),
                "AI_예측_차기분기_점포당매출_만원": sales,
                "AI_예측_생존율_퍼센트": surv,
                "총_유동인구": int(row["총_유동인구"]),
                "구글리뷰_로컬점수": round(float(row["구글리뷰_로컬점수"]), 2),
                "평균_신뢰도평점": rating,
                "추천_분석코멘트": (
                    f"'{matched_cat}' 창업 시 {row['권역']}(AI 예측 점포당 매출 "
                    f"{sales:,.0f}만 원, 예측 생존율 {surv:.1f}%, "
                    f"리뷰 평점 {rating:.2f})이 기대 수익성과 안정성 면에서 가장 유리합니다."
                ),
            })
        return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    features_csv = os.path.join(base_dir, "seongbuk_training_features.csv")
    ts_csv = os.path.join(base_dir, "seoul_timeseries_training.csv")

    artifacts = train_bidirectional_model(
        features_csv_path=features_csv,
        timeseries_csv_path=ts_csv,
        model_save_dir=base_dir,
        epochs=80,
        lr=0.005,
    )
    print(f"[학습 완료] Train 샘플: {artifacts.train_samples:,}개 (서울시 425개 행정동 2025Q1~2026Q1)")
    print(f"[미래 검증 - 서울시 2026Q2 ({artifacts.val_seoul_samples:,}개)] 점포당 매출(log) R2: {artifacts.seoul_val_sales_r2}, RMSE: {artifacts.seoul_val_sales_rmse}")
    print(f"[미래 검증 - 성북구 2026Q2 ({artifacts.val_seongbuk_samples:,}개)] 점포당 매출(log) R2: {artifacts.seongbuk_val_sales_r2}, RMSE: {artifacts.seongbuk_val_sales_rmse}, 생존율 MAE: {artifacts.seongbuk_val_survival_mae}%p")

    recommender = DeepBiteRecommender(features_csv_path=features_csv, artifacts=artifacts)

    print("\n=== [기존 main.py 드롭다운 8개 업종 호환 테스트] ===")
    for old_cat in ["한식", "분식", "호프/통닭", "까페", "일식", "중국식", "경양식", "통닭(치킨)"]:
        top1 = recommender.recommend_regions_for_category(category=old_cat, top_k=1)[0]
        print(f"  입력 '{old_cat}' -> 매핑 '{top1['희망_업종']}' | 1위 {top1['추천_권역']} ({top1['AI_상권적합도_점수']}점)")