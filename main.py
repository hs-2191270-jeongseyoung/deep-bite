from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import math
from pyproj import Transformer

app = FastAPI()

# ---------------------------------------------------------
# 1. PyTorch AI 모델 정의
# ---------------------------------------------------------
class MarketEvaluationModel(nn.Module):
    def __init__(self):
        super(MarketEvaluationModel, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

ai_model = MarketEvaluationModel()
ai_model.eval()

# ---------------------------------------------------------
# 2. 공공데이터 및 TM -> 위경도(WGS84) 좌표 변환 파이프라인
# ---------------------------------------------------------
DATA_FILE = "data/seongbuk_market.csv"

# 중부원점 TM(EPSG:5181/2097 계열) -> GPS 위경도(EPSG:4326) 변환기
transformer = Transformer.from_crs("epsg:5181", "epsg:4326", always_xy=True)

cached_df = None

def load_and_preprocess_data():
    global cached_df
    if cached_df is not None:
        return cached_df

    if not os.path.exists(DATA_FILE):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(DATA_FILE, encoding='cp949', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_FILE, encoding='utf-8', low_memory=False)
    except Exception:
        return pd.DataFrame()

    # 좌표가 유효한 행만 필터링 후 위경도 컬럼 생성
    valid_coords = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)']).copy()
    
    def convert_coords(row):
        try:
            x, y = float(row['좌표정보(X)']), float(row['좌표정보(Y)'])
            lon, lat = transformer.transform(x, y)
            return pd.Series([lat, lon])
        except:
            return pd.Series([None, None])

    valid_coords[['lat', 'lng']] = valid_coords.apply(convert_coords, axis=1)
    cached_df = valid_coords.dropna(subset=['lat', 'lng'])
    return cached_df

# 두 위경도 좌표 간 거리 계산 (Haversine Formula, 단위: meter)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 지구 반지름 (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------------------------------------
# 3. 반경(m) 기반 API 엔드포인트
# ---------------------------------------------------------
@app.get("/api/radius-analysis")
def analyze_radius(lat: float, lng: float, radius_m: float = 300.0):
    df = load_and_preprocess_data()
    
    if df.empty:
        raise HTTPException(status_code=500, detail="데이터를 로드할 수 없습니다.")

    # 반경 거리 계산 및 필터링
    df['distance'] = df.apply(lambda r: haversine_distance(lat, lng, r['lat'], r['lng']), axis=1)
    nearby_df = df[df['distance'] <= radius_m]

    total_count = len(nearby_df)
    active_df = nearby_df[nearby_df['상세영업상태명'] == '영업']
    active_count = len(active_df)
    closed_count = total_count - active_count

    # 업태별 영업/전체 현황 집계 및 생존율 계산
    if not nearby_df.empty:
        industry_stats = nearby_df.groupby('업태구분명')['상세영업상태명'].agg(
            total='count',
            active=lambda x: (x == '영업').sum()
        )
        industry_stats['survival_rate'] = industry_stats['active'] / industry_stats['total']
        
        top_categories = active_df['업태구분명'].value_counts().head(3).to_dict()

        # 반경 내 생존율 높은 추천 업종 도출
        valid_ind = industry_stats[industry_stats['total'] >= 2]
        if not valid_ind.empty:
            best_row = valid_ind.sort_values(by=['survival_rate', 'active'], ascending=False).iloc[0]
            recommended_industry = best_row.name
            survival_pct = round(best_row['survival_rate'] * 100, 1)
            industry_reason = f"클릭 지점 반경 {int(radius_m)}m 내에서 생존율 **{survival_pct}%**를 기록하며 가장 안정적인 최적 추천 업종입니다."
        else:
            recommended_industry = active_df['업태구분명'].iloc[0] if not active_df.empty else "한식"
            survival_pct = 50.0
            industry_reason = f"반경 {int(radius_m)}m 내 점포 수가 적어 틈새시장 진입이 가능합니다."
    else:
        top_categories = {"데이터 없음": 0}
        recommended_industry = "카페/한식"
        survival_pct = 0.0
        industry_reason = f"선택하신 반경 {int(radius_m)}m 이내에 등록된 점포가 없어 신규 상권 개척 가능성이 높은 지역입니다."

    # PyTorch 모델 입력 (점포 밀도, 폐업 비율, 생존 지표)
    active_tensor_val = active_count / 100.0  # 반경 단위 스케일링
    closed_ratio = (closed_count / total_count) if total_count > 0 else 0.5
    diversity = (survival_pct / 100.0)

    input_tensor = torch.tensor([[active_tensor_val, closed_ratio, diversity]], dtype=torch.float32)

    with torch.no_grad():
        ai_raw_score = ai_model(input_tensor).item()

    ai_score = round(ai_raw_score * 100, 1)

    recommendation = f"🤖 [반경 {int(radius_m)}m 미세 상권 분석] 선택 좌표 기준 영업 점포는 {active_count}개이며, AI 창업 적합도는 {ai_score}점입니다."

    # 주변 점포 마커 표시용 간단 데이터 파싱 (최대 30개)
    nearby_stores = []
    for _, store in active_df.head(30).iterrows():
        nearby_stores.append({
            "name": str(store['사업장명']),
            "category": str(store['업태구분명']),
            "lat": store['lat'],
            "lng": store['lng']
        })

    return {
        "lat": lat,
        "lng": lng,
        "radius_m": radius_m,
        "total_store_count": total_count,
        "active_store_count": active_count,
        "closed_store_count": closed_count,
        "top_categories": top_categories,
        "recommended_industry": recommended_industry,
        "industry_reason": industry_reason,
        "estimated_score": ai_score,
        "recommendation": recommendation,
        "nearby_stores": nearby_stores
    }

# ---------------------------------------------------------
# 4. 프론트엔드 (클릭 타겟팅 + 반경 서클 시각화)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>딥바이트 (DeepBite) - 반경 타겟팅 AI 상권 분석</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
                body { font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; display: flex; height: 100vh; }
                .sidebar { width: 460px; background: white; padding: 25px; box-shadow: 4px 0 15px rgba(0,0,0,0.1); z-index: 1000; box-sizing: border-box; overflow-y: auto; }
                #map { flex-grow: 1; height: 100%; cursor: crosshair; }
                h2 { color: #2c3e50; margin-top: 0; }
                .badge { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; }
                .radius-control { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #e9ecef; }
                input[type=range] { width: 100%; margin-top: 8px; }
                #result { margin-top: 15px; background: #ebf5fb; padding: 15px; border-radius: 6px; border-left: 5px solid #2980b9; }
                .recommend-box { background: #fef9e7; border: 1px solid #f9e79f; padding: 12px; border-radius: 6px; margin-top: 15px; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2>🎯 딥바이트 <span class="badge">반경 분석</span></h2>
                <p>지도의 **특정 지점을 클릭**하여 반경 내 미세 상권을 분석하세요.</p>
                
                <div class="radius-control">
                    <label><b>📏 분석 반경 설정: <span id="radiusVal" style="color:#e67e22;">300</span>m</b></label>
                    <input type="range" id="radiusInput" min="100" max="1000" step="50" value="300" oninput="updateRadius(this.value)">
                </div>

                <div id="result">지도를 클릭하면 반경 내 상권 데이터가 실시간 집계됩니다.</div>
            </div>

            <div id="map"></div>

            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
                // 1. 성북구 중심 지도 설정
                const map = L.map('map').setView([37.6033, 127.0242], 14);

                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 19,
                    attribution: '© OpenStreetMap'
                }).addTo(map);

                let currentLat = 37.6033;
                let currentLng = 127.0242;
                let clickMarker = null;
                let radiusCircle = null;
                let storeMarkers = [];

                function updateRadius(val) {
                    document.getElementById('radiusVal').innerText = val;
                    if (clickMarker) {
                        getAnalysis(currentLat, currentLng);
                    }
                }

                // 지도 클릭 이벤트 핸들러
                map.on('click', function(e) {
                    currentLat = e.latlng.lat;
                    currentLng = e.latlng.lng;
                    getAnalysis(currentLat, currentLng);
                });

                async function getAnalysis(lat, lng) {
                    const radius = document.getElementById('radiusInput').value;
                    const res = await fetch(`/api/radius-analysis?lat=${lat}&lng=${lng}&radius_m=${radius}`);
                    const data = await res.json();

                    // 기존 마커 및 원 삭제
                    if (clickMarker) map.removeLayer(clickMarker);
                    if (radiusCircle) map.removeLayer(radiusCircle);
                    storeMarkers.forEach(m => map.removeLayer(m));
                    storeMarkers = [];

                    // 클릭 위치 마커 및 반경 원 그리기
                    clickMarker = L.marker([lat, lng]).addTo(map).bindPopup("<b>분석 타겟 지점</b>").openPopup();
                    radiusCircle = L.circle([lat, lng], {
                        color: '#e67e22',
                        fillColor: '#f39c12',
                        fillOpacity: 0.2,
                        radius: parseInt(radius)
                    }).addTo(map);

                    // 반경 내 실제 점포 소형 마커 찍기
                    if (data.nearby_stores) {
                        data.nearby_stores.forEach(s => {
                            let sm = L.circleMarker([s.lat, s.lng], {
                                radius: 4,
                                color: '#27ae60',
                                fillColor: '#2ecc71',
                                fillOpacity: 0.8
                            }).addTo(map).bindPopup(`<b>${s.name}</b><br>업태: ${s.category}`);
                            storeMarkers.push(sm);
                        });
                    }

                    // 사이드바 분석 결과 업데이트
                    let categoriesStr = "";
                    for (const [cat, cnt] of Object.entries(data.top_categories)) {
                        categoriesStr += `• ${cat}: ${cnt}개<br>`;
                    }

                    document.getElementById('result').innerHTML = `
                        <b>📍 선택 좌표:</b> ${lat.toFixed(4)}, ${lng.toFixed(4)}<br>
                        <b>🟢 반경 내 영업 점포:</b> ${data.active_store_count}개<br>
                        <b>🔴 반경 내 폐업 점포:</b> ${data.closed_store_count}개<br>
                        <hr style="border: 0.5px solid #aed6f1;">
                        <b>🏆 반경 내 업태 분포 TOP 3:</b><br>${categoriesStr}
                        
                        <div class="recommend-box">
                            <b>🎯 반경 맞춤 추천 업종:</b> <span style="color: #c0392b; font-size: 16px;"><b>${data.recommended_industry}</b></span><br>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #5d6d7e;">${data.industry_reason}</p>
                        </div>

                        <div style="margin-top: 15px;">
                            <b>⭐ PyTorch AI 적합도 점수:</b> <b style="font-size: 16px; color: #2980b9;">${data.estimated_score}점</b>
                        </div>
                        <p style="color: #1b4f72; font-weight: bold; margin-top:10px; margin-bottom:0;">${data.recommendation}</p>
                    `;
                }

                // 초기 로드 시 기본 위치 분석
                window.onload = () => { getAnalysis(currentLat, currentLng); };
            </script>
        </body>
    </html>
    """