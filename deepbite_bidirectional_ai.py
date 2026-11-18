"""
DeepBite (딥바이트) - 양방향 PyTorch AI 상권 적합도 평가 및 추천 모델 (공공데이터 3종 + 구글 리뷰 통합 버전)
Repository: https://github.com/hs-2191270-jeongseyoung/deep-bite

지원 기능:
1. 양방향 추론 아키텍처 (Bidirectional Suitability Neural Network)
   - 방향 1 (지역 선택 -> 최적 업종 추천): 특정 성북구 행정동 또는 (위도, 경도) 좌표 입력 시
     9대 음식점 업종별 창업 적합도 점수(0~100점), 점포당 월평균 추정매출, 폐업률, 추천 사유를 반환
   - 방향 2 (업종 선택 -> 최적 지역 추천): 희망 음식점 업종 입력 시
     성북구 12대 행정동/권역별 창업 적합도 점수(0~100점), 유동인구, 점포당 매출, 추천 사유를 반환
2. 서울시 상권분석서비스 3종(길단위인구·점포·추정매출) 및 구글 리뷰 감성·평점 14개 핵심 수치 피처 학습
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


NUMERIC_FEATURE_COLS: List[str] = [
    "점포수",
    "공식_외식점포수",
    "프랜차이즈_비중",
    "공식_개업률",
    "공식_폐업률",
    "권역내_업종점유율",
    "평균_신뢰도평점",
    "평균_리뷰감성점수",
    "평균_로그리뷰수",
    "권역내_리뷰수요비중",
    "유동인구_지수",
    "청년층_2030_유동비중",
    "야간_17_24_유동비중",
    "추정매출_지수",
]


if TORCH_AVAILABLE:
    class DeepBiteSuitabilityNet(nn.Module):
        """
        성북구 권역 임베딩 + 업종 임베딩 + 14개 상권/공공데이터/리뷰 수치 피처를 결합한
        양방향 상권 적합도 예측 PyTorch 신경망 모델
        """

        def __init__(
            self,
            num_regions: int,
            num_categories: int,
            num_numeric_features: int = len(NUMERIC_FEATURE_COLS),
            region_emb_dim: int = 8,
            category_emb_dim: int = 8,
        ) -> None:
            super().__init__()
            self.region_emb = nn.Embedding(num_regions, region_emb_dim)
            self.category_emb = nn.Embedding(num_categories, category_emb_dim)

            in_dim = region_emb_dim + category_emb_dim + num_numeric_features
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            region_idx: "torch.Tensor",
            category_idx: "torch.Tensor",
            numeric_x: "torch.Tensor",
        ) -> "torch.Tensor":
            r_e = self.region_emb(region_idx)
            c_e = self.category_emb(category_idx)
            x = torch.cat([r_e, c_e, numeric_x], dim=-1)
            return self.mlp(x).squeeze(-1) * 100.0


@dataclass
class TrainingArtifacts:
    """학습된 모델 가중치 및 정규화 메타데이터 컨테이너"""
    regions: List[str]
    categories: List[str]
    feature_cols: List[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    train_rmse: float
    train_r2: float
    weights: Dict[str, np.ndarray]


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def train_bidirectional_model(
    features_csv_path: str,
    model_save_dir: str,
    epochs: int = 450,
    lr: float = 0.008,
    seed: int = 42,
) -> TrainingArtifacts:
    """
    전처리된 성북구 (권역 x 업종) 피처 데이터를 학습하여
    PyTorch 체크포인트(`.pt`) 및 경량 추론 가중치(`.npz`)를 저장합니다.
    """
    np.random.seed(seed)
    df = pd.read_csv(features_csv_path)

    regions = sorted(df["권역"].unique().tolist())
    categories = sorted(df["표준업종"].unique().tolist())
    reg_to_idx = {r: i for i, r in enumerate(regions)}
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    r_idx = df["권역"].map(reg_to_idx).values.astype(np.int64)
    c_idx = df["표준업종"].map(cat_to_idx).values.astype(np.int64)

    raw_x = df[NUMERIC_FEATURE_COLS].values.astype(np.float32)
    feat_mean = raw_x.mean(axis=0)
    feat_std = np.where(raw_x.std(axis=0) < 1e-6, 1.0, raw_x.std(axis=0))
    norm_x = (raw_x - feat_mean) / feat_std

    y_true = df["상권적합도_점수"].values.astype(np.float32)

    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        model = DeepBiteSuitabilityNet(
            num_regions=len(regions),
            num_categories=len(categories),
            num_numeric_features=len(NUMERIC_FEATURE_COLS),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        dataset = TensorDataset(
            torch.tensor(r_idx, dtype=torch.long),
            torch.tensor(c_idx, dtype=torch.long),
            torch.tensor(norm_x, dtype=torch.float32),
            torch.tensor(y_true, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=27, shuffle=True)

        model.train()
        for _ in range(epochs):
            for b_r, b_c, b_x, b_y in loader:
                optimizer.zero_grad()
                preds = model(b_r, b_c, b_x)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds_np = model(
                torch.tensor(r_idx, dtype=torch.long),
                torch.tensor(c_idx, dtype=torch.long),
                torch.tensor(norm_x, dtype=torch.float32),
            ).numpy()

        pt_path = os.path.join(model_save_dir, "deepbite_suitability_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "regions": regions,
                "categories": categories,
                "feature_cols": NUMERIC_FEATURE_COLS,
                "feat_mean": feat_mean.tolist(),
                "feat_std": feat_std.tolist(),
            },
            pt_path,
        )
        weights_dict = {"preds_cache": preds_np}
    else:
        r_oh = np.eye(len(regions), dtype=np.float32)[r_idx]
        c_oh = np.eye(len(categories), dtype=np.float32)[c_idx]
        full_x = np.concatenate([r_oh, c_oh, norm_x], axis=1)

        in_dim = full_x.shape[1]
        h1_dim, h2_dim = 48, 24
        w1 = np.random.randn(in_dim, h1_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        b1 = np.zeros((1, h1_dim), dtype=np.float32)
        w2 = np.random.randn(h1_dim, h2_dim).astype(np.float32) * np.sqrt(2.0 / h1_dim)
        b2 = np.zeros((1, h2_dim), dtype=np.float32)
        w3 = np.random.randn(h2_dim, 1).astype(np.float32) * np.sqrt(1.0 / h2_dim)
        b3 = np.zeros((1, 1), dtype=np.float32)

        y_scaled = (y_true / 100.0).reshape(-1, 1)
        n_samples = float(len(full_x))

        params = [w1, b1, w2, b2, w3, b3]
        m_state = [np.zeros_like(p) for p in params]
        v_state = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for step in range(1, epochs + 1):
            z1 = full_x @ w1 + b1
            a1 = _relu(z1)
            z2 = a1 @ w2 + b2
            a2 = _relu(z2)
            z3 = a2 @ w3 + b3
            out = _sigmoid(z3)

            diff = out - y_scaled
            dz3 = (2.0 / n_samples) * diff * out * (1.0 - out)
            dw3 = a2.T @ dz3
            db3 = dz3.sum(axis=0, keepdims=True)

            da2 = dz3 @ w3.T
            dz2 = da2 * (z2 > 0)
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0, keepdims=True)

            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0)
            dw1 = full_x.T @ dz1
            db1 = dz1.sum(axis=0, keepdims=True)

            grads = [dw1, db1, dw2, db2, dw3, db3]
            for i, (p, g) in enumerate(zip(params, grads)):
                m_state[i] = beta1 * m_state[i] + (1.0 - beta1) * g
                v_state[i] = beta2 * v_state[i] + (1.0 - beta2) * (g * g)
                m_hat = m_state[i] / (1.0 - beta1 ** step)
                v_hat = v_state[i] / (1.0 - beta2 ** step)
                p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        preds_np = (_sigmoid(_relu(_relu(full_x @ w1 + b1) @ w2 + b2) @ w3 + b3).squeeze(-1)) * 100.0
        weights_dict = {
            "w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3,
            "preds_cache": preds_np,
        }

    rmse = float(np.sqrt(np.mean((preds_np - y_true) ** 2)))
    ss_res = float(np.sum((y_true - preds_np) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / max(ss_tot, 1e-6))

    npz_path = os.path.join(model_save_dir, "deepbite_model_weights.npz")
    np.savez(
        npz_path,
        regions=np.array(regions),
        categories=np.array(categories),
        feature_cols=np.array(NUMERIC_FEATURE_COLS),
        feat_mean=feat_mean,
        feat_std=feat_std,
        **weights_dict,
    )

    return TrainingArtifacts(
        regions=regions,
        categories=categories,
        feature_cols=NUMERIC_FEATURE_COLS,
        feat_mean=feat_mean,
        feat_std=feat_std,
        train_rmse=round(rmse, 3),
        train_r2=round(r2, 4),
        weights=weights_dict,
    )


class DeepBiteRecommender:
    """
    양방향 상권 적합도 추론 엔진
    - Direction 1: `recommend_categories_for_region(region_or_coords)` (지역 -> 최적 업종 추천)
    - Direction 2: `recommend_regions_for_category(category)` (업종 -> 최적 지역 추천)
    """

    def __init__(self, features_csv_path: str, artifacts: TrainingArtifacts) -> None:
        self.df = pd.read_csv(features_csv_path).copy()
        self.artifacts = artifacts
        self.df["AI_예측_적합도"] = np.round(artifacts.weights["preds_cache"], 2)

    def find_nearest_region(self, lat: float, lng: float) -> str:
        reg_coords = self.df.groupby("권역")[["중심_위도", "중심_경도"]].mean()
        dists = (reg_coords["중심_위도"] - lat) ** 2 + (reg_coords["중심_경도"] - lng) ** 2
        return str(dists.idxmin())

    def recommend_categories_for_region(
        self,
        region: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """
        [방향 1: 지역 선택 -> 최적 업종 추천]
        """
        target_region = region
        if not target_region and lat is not None and lng is not None:
            target_region = self.find_nearest_region(lat, lng)
        if not target_region or target_region not in self.df["권역"].values:
            target_region = "동선동"

        sub = self.df[self.df["권역"] == target_region].sort_values(
            by="AI_예측_적합도", ascending=False
        )

        results: List[Dict[str, object]] = []
        for rank, (_, row) in enumerate(sub.head(top_k).iterrows(), start=1):
            results.append({
                "순위": rank,
                "권역": row["권역"],
                "추천_업종": row["표준업종"],
                "AI_상권적합도_점수": float(row["AI_예측_적합도"]),
                "공식_외식점포수": float(row["공식_외식점포수"]),
                "점포당_월매출_만원": float(row["점포당_월매출_만원"]),
                "공식_폐업률_퍼센트": float(row["공식_폐업률"]),
                "평균_신뢰도평점": float(row["평균_신뢰도평점"]),
                "리뷰_감성긍정도_퍼센트": round(float(row["평균_리뷰감성점수"]) * 100, 1),
                "추천_분석코멘트": (
                    f"{row['권역']} 내 '{row['표준업종']}' 업종은 점포당 월평균 추정매출 "
                    f"{row['점포당_월매출_만원']:,.0f}만 원(폐업률 {row['공식_폐업률']:.1f}%) 및 "
                    f"리뷰 평점 {row['평균_신뢰도평점']:.2f}점으로 수요 대비 창업 안정성이 높습니다."
                ),
            })
        return results

    def recommend_regions_for_category(
        self,
        category: str,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """
        [방향 2: 업종 선택 -> 최적 지역 추천]
        """
        matched_cat = next(
            (c for c in self.artifacts.categories if category in c or c in category),
            "카페/디저트",
        )
        sub = self.df[self.df["표준업종"] == matched_cat].sort_values(
            by="AI_예측_적합도", ascending=False
        )

        results: List[Dict[str, object]] = []
        for rank, (_, row) in enumerate(sub.head(top_k).iterrows(), start=1):
            results.append({
                "순위": rank,
                "희망_업종": matched_cat,
                "추천_권역": row["권역"],
                "AI_상권적합도_점수": float(row["AI_예측_적합도"]),
                "중심_위도": float(row["중심_위도"]),
                "중심_경도": float(row["중심_경도"]),
                "총_유동인구": int(row["총_유동인구"]),
                "청년층_2030_유동비중_퍼센트": round(float(row["청년층_2030_유동비중"]) * 100, 1),
                "점포당_월매출_만원": float(row["점포당_월매출_만원"]),
                "공식_외식점포수": float(row["공식_외식점포수"]),
                "평균_신뢰도평점": float(row["평균_신뢰도평점"]),
                "추천_분석코멘트": (
                    f"'{matched_cat}' 창업 시 {row['권역']}(분기 유동인구 {row['총_유동인구']:,.0f}명, "
                    f"2030 비중 {row['청년층_2030_유동비중']*100:.1f}%, 점포당 월매출 {row['점포당_월매출_만원']:,.0f}만 원)이 "
                    f"매출 잠재력과 상권 적합도 면에서 가장 우수합니다."
                ),
            })
        return results


if __name__ == "__main__":
    out_dir = "/working_dir/c_d0b86430e3c81b72/artifacts/file_generation/ttl=63d/output"
    features_csv = os.path.join(out_dir, "seongbuk_training_features.csv")

    artifacts = train_bidirectional_model(
        features_csv_path=features_csv,
        model_save_dir=out_dir,
        epochs=450,
        lr=0.008,
    )
    print(f"[모델 재학습 완료] Train RMSE: {artifacts.train_rmse}, R2 Score: {artifacts.train_r2}")

    recommender = DeepBiteRecommender(features_csv_path=features_csv, artifacts=artifacts)

    print("\n=== [방향 1 테스트] 지역 선택(안암동) -> 최적 창업 업종 TOP 3 ===")
    for item in recommender.recommend_categories_for_region(region="안암동", top_k=3):
        print(
            f"  {item['순위']}위: {item['추천_업종']} "
            f"(적합도 {item['AI_상권적합도_점수']}점 | 점포당 월매출 {item['점포당_월매출_만원']:,.0f}만원 | 폐업률 {item['공식_폐업률_퍼센트']}%)"
        )

    print("\n=== [방향 2 테스트] 업종 선택(카페/디저트) -> 성북구 최적 창업 지역 TOP 3 ===")
    for item in recommender.recommend_regions_for_category(category="카페/디저트", top_k=3):
        print(
            f"  {item['순위']}위: {item['추천_권역']} "
            f"(적합도 {item['AI_상권적합도_점수']}점 | 유동인구 {item['총_유동인구']:,}명 | 점포당 월매출 {item['점포당_월매출_만원']:,.0f}만원)"
        )