from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import math
from pyproj import Transformer
from datetime import datetime

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
# 2. 공공데이터 로드 및 전처리 파이프라인
# ---------------------------------------------------------
DATA_FILE = "data/seongbuk_market.csv"
transformer = Transformer.from_crs("epsg:5181", "epsg:4326", always_xy=True)

COORDINATES = {
    "길음동": {"lat": 37.6033, "lng": 127.0242},
    "정릉동": {"lat": 37.6096, "lng": 127.0118},
    "장위동": {"lat": 37.6151, "lng": 127.0527},
    "석관동": {"lat": 37.6074, "lng": 127.0561},
    "하월곡동": {"lat": 37.6021, "lng": 127.0385},
    "종암동": {"lat": 37.5958, "lng": 127.0326},
    "동선동": {"lat": 37.5904, "lng": 127.0169},
    "안암동": {"lat": 37.5862, "lng": 127.0292},
    "성북동": {"lat": 37.5951, "lng": 126.9936},
    "삼선동": {"lat": 37.5878, "lng": 127.0119},
    "돈암동": {"lat": 37.5925, "lng": 127.0169},
}

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

    valid_coords = df.dropna(subset=['좌표정보(X)', '좌표정보(Y)']).copy()
    
    def convert_coords(row):
        try:
            x, y = float(row['좌표정보(X)']), float(row['좌표정보(Y)'])
            lon, lat = transformer.transform(x, y)
            return pd.Series([lat, lon])
        except:
            return pd.Series([None, None])

    valid_coords[['lat', 'lng']] = valid_coords.apply(convert_coords, axis=1)
    df_processed = valid_coords.dropna(subset=['lat', 'lng']).copy()

    # 법정동 파싱 함수
    def extract_dong(addr):
        if pd.isna(addr): return "기타"
        parts = str(addr).split()
        if len(parts) > 2:
            dong = parts[2]
            for key in COORDINATES.keys():
                if key in dong:
                    return key
        return "기타"

    df_processed['법정동'] = df_processed['지번주소'].apply(extract_dong)

    # 날짜 파싱 및 영업 유지 기간(Lifespan) 계산
    df_processed['인허가일자_dt'] = pd.to_datetime(df_processed['인허가일자'], errors='coerce')
    df_processed['폐업일자_dt'] = pd.to_datetime(df_processed['폐업일자'], errors='coerce')
    
    current_date = pd.to_datetime('2026-10-28')

    def calc_lifespan(row):
        if pd.notnull(row['폐업일자_dt']):
            delta = row['폐업일자_dt'] - row['인허가일자_dt']
        else:
            delta = current_date - row['인허가일자_dt']
        return max(delta.days, 0) / 365.25

    df_processed['영업유지기간_년'] = df_processed.apply(calc_lifespan, axis=1)
    df_processed['개업년도'] = df_processed['인허가일자_dt'].dt.year.fillna(0).astype(int)
    df_processed['폐업년도'] = df_processed['폐업일자_dt'].dt.year.fillna(0).astype(int)

    cached_df = df_processed
    return cached_df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------------------------------------
# 3. API 엔드포인트 1: 반경 미세 상권 분석
# ---------------------------------------------------------
@app.get("/api/radius-analysis")
def analyze_radius(lat: float, lng: float, radius_m: float = 300.0):
    df = load_and_preprocess_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="데이터 로드 실패")

    df['distance'] = df.apply(lambda r: haversine_distance(lat, lng, r['lat'], r['lng']), axis=1)
    nearby_df = df[df['distance'] <= radius_m]

    total_count = len(nearby_df)
    active_df = nearby_df[nearby_df['상세영업상태명'] == '영업']
    closed_df = nearby_df[nearby_df['상세영업상태명'] == '폐업']
    active_count = len(active_df)
    closed_count = len(closed_df)

    avg_active_lifespan = round(active_df['영업유지기간_년'].mean(), 1) if not active_df.empty else 0.0
    avg_closed_lifespan = round(closed_df['영업유지기간_년'].mean(), 1) if not closed_df.empty else 0.0

    recent_years = [2022, 2023, 2024, 2025, 2026]
    yearly_trend = {}
    for y in recent_years:
        op_cnt = len(nearby_df[nearby_df['개업년도'] == y])
        cl_cnt = len(nearby_df[nearby_df['폐업년도'] == y])
        yearly_trend[str(y)] = {"open": op_cnt, "close": cl_cnt}

    if not nearby_df.empty:
        industry_stats = nearby_df.groupby('업태구분명')['상세영업상태명'].agg(
            total='count',
            active=lambda x: (x == '영업').sum()
        )
        industry_stats['survival_rate'] = industry_stats['active'] / industry_stats['total']
        top_categories = active_df['업태구분명'].value_counts().head(3).to_dict()
        
        valid_ind = industry_stats[industry_stats['total'] >= 2]
        if not valid_ind.empty:
            best_row = valid_ind.sort_values(by=['survival_rate', 'active'], ascending=False).iloc[0]
            recommended_industry = best_row.name
            survival_pct = round(best_row['survival_rate'] * 100, 1)
            industry_reason = f"반경 내 생존율 **{survival_pct}%** (평균 운영 {avg_active_lifespan}년)를 기록하며 과당경쟁을 피한 최적 업종입니다."
        else:
            recommended_industry = "카페"
            survival_pct = 50.0
            industry_reason = "신규 틈새시장 진입이 유리합니다."
    else:
        top_categories = {"데이터 없음": 0}
        recommended_industry = "한식/카페"
        survival_pct = 0.0
        industry_reason = "반경 내 점포가 없어 상권 개척이 필요합니다."

    active_tensor_val = active_count / 100.0
    closed_ratio = (closed_count / total_count) if total_count > 0 else 0.5
    diversity = (survival_pct / 100.0)

    input_tensor = torch.tensor([[active_tensor_val, closed_ratio, diversity]], dtype=torch.float32)
    with torch.no_grad():
        ai_raw_score = ai_model(input_tensor).item()
    ai_score = round(ai_raw_score * 100, 1)

    recommendation = f"🤖 [시계열 및 생존력 분석] 추천 업종: {recommended_industry} (적합도 {ai_score}점)"

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
        "active_store_count": active_count,
        "closed_store_count": closed_count,
        "avg_active_lifespan": avg_active_lifespan,
        "avg_closed_lifespan": avg_closed_lifespan,
        "yearly_trend": yearly_trend,
        "top_categories": top_categories,
        "recommended_industry": recommended_industry,
        "industry_reason": industry_reason,
        "estimated_score": ai_score,
        "recommendation": recommendation,
        "nearby_stores": nearby_stores
    }

# ---------------------------------------------------------
# 4. API 엔드포인트 2: 업종별 최적 입지 추천 (신규 기능)
# ---------------------------------------------------------
@app.get("/api/industry-recommendation")
def industry_recommendation(industry: str = "한식"):
    df = load_and_preprocess_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="데이터 로드 실패")

    ind_df = df[df['업태구분명'] == industry]
    if ind_df.empty:
        return {"industry": industry, "recommendations": []}

    dong_stats = ind_df.groupby('법정동')['상세영업상태명'].agg(
        total='count',
        active=lambda x: (x == '영업').sum()
    )
    dong_stats['survival_rate'] = dong_stats['active'] / dong_stats['total']
    
    # 최소 점포수 5개 이상인 동 중에서 생존율과 영업점포수가 높은 순으로 정렬
    valid_dongs = dong_stats[dong_stats['total'] >= 5].sort_values(by=['survival_rate', 'active'], ascending=False).head(5)

    recommendations = []
    for dong, row in valid_dongs.iterrows():
        coord = COORDINATES.get(dong, {"lat": 37.6033, "lng": 127.0242})
        
        # PyTorch 모델 활용 점수 산출
        active_val = row['active'] / 200.0
        closed_ratio = (row['total'] - row['active']) / row['total']
        div = row['survival_rate']
        inp = torch.tensor([[active_val, closed_ratio, div]], dtype=torch.float32)
        with torch.no_grad():
            score = round(ai_model(inp).item() * 100, 1)

        recommendations.append({
            "district": dong,
            "lat": coord["lat"],
            "lng": coord["lng"],
            "active_count": int(row['active']),
            "total_count": int(row['total']),
            "survival_rate": round(row['survival_rate'] * 100, 1),
            "score": score
        })

    return {
        "industry": industry,
        "recommendations": recommendations
    }

# ---------------------------------------------------------
# 5. 프론트엔드 대시보드 (듀얼 모드: 반경 분석 + 업종별 입지 추천)
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>딥바이트 (DeepBite) - AI 상권 및 업종별 입지 추천</title>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
            <style>
                body { font-family: 'Malgun Gothic', sans-serif; margin: 0; padding: 0; background-color: #f4f4f9; display: flex; height: 100vh; }
                .sidebar { width: 480px; background: white; padding: 25px; box-shadow: 4px 0 15px rgba(0,0,0,0.1); z-index: 1000; box-sizing: border-box; overflow-y: auto; }
                #map { flex-grow: 1; height: 100%; cursor: crosshair; }
                h2 { color: #2c3e50; margin-top: 0; }
                .badge { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 4px; font-size: 12px; }
                .tab-menu { display: flex; margin-bottom: 15px; border-bottom: 2px solid #ddd; }
                .tab-btn { flex: 1; padding: 10px; text-align: center; cursor: pointer; font-weight: bold; color: #7f8c8d; background: #ecf0f1; border: none; }
                .tab-btn.active { background: #2980b9; color: white; }
                .mode-content { display: none; }
                .mode-content.active { display: block; }
                .radius-control, .industry-control { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #e9ecef; }
                input[type=range], select { width: 100%; margin-top: 8px; padding: 10px; box-sizing: border-box; }
                button { width: 100%; padding: 12px; background: #27ae60; color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 10px; }
                button:hover { background: #219653; }
                #result, #indResult { margin-top: 15px; background: #ebf5fb; padding: 15px; border-radius: 6px; border-left: 5px solid #2980b9; font-size: 14px; }
                .recommend-box { background: #fef9e7; border: 1px solid #f9e79f; padding: 12px; border-radius: 6px; margin-top: 15px; }
                .trend-table, .ind-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
                .trend-table th, .trend-table td, .ind-table th, .ind-table td { border: 1px solid #d4e6f1; padding: 6px; text-align: center; }
                .trend-table th, .ind-table th { background: #aed6f1; color: #1b4f72; }
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2>🎯 딥바이트 <span class="badge">AI 상권 분석</span></h2>
                
                <div class="tab-menu">
                    <button class="tab-btn active" onclick="switchMode('radius')">📍 지도 반경 분석</button>
                    <button class="tab-btn" onclick="switchMode('industry')">🏆 업종별 최적 입지</button>
                </div>

                <!-- 모드 1: 반경 분석 -->
                <div id="modeRadius" class="mode-content active">
                    <p>지도를 클릭하여 반경 내 점포의 **영업 유지 기간 및 시계열 트렌드**를 분석하세요.</p>
                    <div class="radius-control">
                        <label><b>📏 분석 반경 설정: <span id="radiusVal" style="color:#e67e22;">300</span>m</b></label>
                        <input type="range" id="radiusInput" min="100" max="1000" step="50" value="300" oninput="updateRadius(this.value)">
                    </div>
                    <div id="result">지도를 클릭하면 상세 분석 리포트가 생성됩니다.</div>
                </div>

                <!-- 모드 2: 업종별 최적 입지 추천 -->
                <div id="modeIndustry" class="mode-content">
                    <p>창업하고 싶은 업종을 선택하면 **성북구 내 생존율이 가장 높은 최적의 법정동 TOP 5**를 추천합니다.</p>
                    <div class="industry-control">
                        <label><b>🍜 창업 희망 업종 선택</b></label>
                        <select id="industrySelect">
                            <option value="한식">한식</option>
                            <option value="분식">분식</option>
                            <option value="호프/통닭">호프/통닭</option>
                            <option value="까페">까페</option>
                            <option value="일식">일식</option>
                            <option value="중국식">중국식</option>
                            <option value="경양식">경양식</option>
                            <option value="통닭(치킨)">통닭(치킨)</option>
                        </select>
                        <button onclick="getIndustryRecommendation()">🔍 업종별 최적 입지 검색</button>
                    </div>
                    <div id="indResult">업종을 선택하고 검색 버튼을 누르세요.</div>
                </div>
            </div>

            <div id="map"></div>

            <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
            <script>
                const map = L.map('map').setView([37.6033, 127.0242], 14);

                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    maxZoom: 19,
                    attribution: '© OpenStreetMap'
                }).addTo(map);

                let currentMode = 'radius';
                let currentLat = 37.6033;
                let currentLng = 127.0242;
                let clickMarker = null;
                let radiusCircle = null;
                let storeMarkers = [];
                let indMarkers = [];

                function switchMode(mode) {
                    currentMode = mode;
                    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                    document.querySelectorAll('.mode-content').forEach(c => c.classList.remove('active'));

                    if (mode === 'radius') {
                        event.target.classList.add('active');
                        document.getElementById('modeRadius').classList.add('active');
                        clearIndMarkers();
                        getAnalysis(currentLat, currentLng);
                    } else {
                        event.target.classList.add('active');
                        document.getElementById('modeIndustry').classList.add('active');
                        clearRadiusMarkers();
                        getIndustryRecommendation();
                    }
                }

                function clearRadiusMarkers() {
                    if (clickMarker) map.removeLayer(clickMarker);
                    if (radiusCircle) map.removeLayer(radiusCircle);
                    storeMarkers.forEach(m => map.removeLayer(m));
                    storeMarkers = [];
                }

                function clearIndMarkers() {
                    indMarkers.forEach(m => map.removeLayer(m));
                    indMarkers = [];
                }

                function updateRadius(val) {
                    document.getElementById('radiusVal').innerText = val;
                    if (currentMode === 'radius' && clickMarker) {
                        getAnalysis(currentLat, currentLng);
                    }
                }

                map.on('click', function(e) {
                    if (currentMode === 'radius') {
                        currentLat = e.latlng.lat;
                        currentLng = e.latlng.lng;
                        getAnalysis(currentLat, currentLng);
                    }
                });

                async function getAnalysis(lat, lng) {
                    const radius = document.getElementById('radiusInput').value;
                    const res = await fetch(`/api/radius-analysis?lat=${lat}&lng=${lng}&radius_m=${radius}`, {
    headers: {
        'ngrok-skip-browser-warning': '69420'
    }
});
                    const data = await res.json();

                    clearRadiusMarkers();

                    clickMarker = L.marker([lat, lng]).addTo(map).bindPopup("<b>분석 타겟 지점</b>").openPopup();
                    radiusCircle = L.circle([lat, lng], {
                        color: '#e67e22',
                        fillColor: '#f39c12',
                        fillOpacity: 0.2,
                        radius: parseInt(radius)
                    }).addTo(map);

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

                    let categoriesStr = "";
                    for (const [cat, cnt] of Object.entries(data.top_categories)) {
                        categoriesStr += `• ${cat}: ${cnt}개<br>`;
                    }

                    let trendRows = "";
                    for (const [year, stats] of Object.entries(data.yearly_trend)) {
                        trendRows += `<tr><td><b>${year}년</b></td><td style="color:green;">+${stats.open}건</td><td style="color:red;">-${stats.close}건</td></tr>`;
                    }

                    document.getElementById('result').innerHTML = `
                        <b>📍 선택 좌표:</b> ${lat.toFixed(4)}, ${lng.toFixed(4)}<br>
                        <b>🟢 영업 점포:</b> ${data.active_store_count}개 (평균 운영: <b>${data.avg_active_lifespan}년</b>)<br>
                        <b>🔴 폐업 점포:</b> ${data.closed_store_count}개 (평균 생존: <b>${data.avg_closed_lifespan}년</b>)<br>
                        <hr style="border: 0.5px solid #aed6f1;">
                        <b>📈 연도별 개폐업 트렌드 (최근 5년):</b>
                        <table class="trend-table">
                            <tr><th>연도</th><th>개업</th><th>폐업</th></tr>
                            ${trendRows}
                        </table>
                        <br>
                        <b>🏆 주요 업태 분포 TOP 3:</b><br>${categoriesStr}
                        
                        <div class="recommend-box">
                            <b>🎯 AI 추천 유망 업종:</b> <span style="color: #c0392b; font-size: 15px;"><b>${data.recommended_industry}</b></span><br>
                            <p style="margin: 5px 0 0 0; font-size: 12px; color: #5d6d7e;">${data.industry_reason}</p>
                        </div>

                        <div style="margin-top: 15px;">
                            <b>⭐ PyTorch AI 적합도 점수:</b> <b style="font-size: 16px; color: #2980b9;">${data.estimated_score}점</b>
                        </div>
                        <p style="color: #1b4f72; font-weight: bold; margin-top:10px; margin-bottom:0;">${data.recommendation}</p>
                    `;
                }

                async function getIndustryRecommendation() {
                    const industry = document.getElementById('industrySelect').value;
                    const res = await fetch(`/api/industry-recommendation?industry=${encodeURIComponent(industry)}`, {
    headers: {
        'ngrok-skip-browser-warning': '69420'
    }
});
                    const data = await res.json();

                    clearIndMarkers();

                    let rows = "";
                    if (data.recommendations.length > 0) {
                        data.recommendations.forEach((rec, idx) => {
                            rows += `<tr>
                                <td><b>${idx+1}위</b></td>
                                <td><b>${rec.district}</b></td>
                                <td style="color:#27ae60;">${rec.survival_rate}%</td>
                                <td style="color:#2980b9;"><b>${rec.score}점</b></td>
                            </tr>`;

                            // 지도에 순위 마커 표시
                            let m = L.marker([rec.lat, rec.lng]).addTo(map)
                                .bindPopup(`<b>[${idx+1}위] 성북구 ${rec.district}</b><br>업종: ${industry}<br>생존율: ${rec.survival_rate}%<br>적합도: ${rec.score}점`);
                            indMarkers.push(m);
                        });

                        // 첫 번째 추천 지역으로 지도 이동
                        map.setView([data.recommendations[0].lat, data.recommendations[0].lng], 14);
                    } else {
                        rows = `<tr><td colspan="4">해당 업종 데이터가 부족합니다.</td></tr>`;
                    }

                    document.getElementById('indResult').innerHTML = `
                        <b>🏆 성북구 '${data.industry}' 창업 유망 지역 TOP 5</b>
                        <table class="ind-table">
                            <tr><th>순위</th><th>법정동</th><th>생존율</th><th>AI 적합도</th></tr>
                            ${rows}
                        </table>
                        <p style="margin-top:10px; font-size:12px; color:#5d6d7e;">💡 지도의 마커를 클릭하여 해당 지역의 상세 위치를 확인하세요.</p>
                    `;
                }

                window.onload = () => { getAnalysis(currentLat, currentLng); };
            </script>
        </body>
    </html>
    """