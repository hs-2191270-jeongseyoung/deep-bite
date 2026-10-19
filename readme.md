# 🍽️ 딥바이트 (DeepBite)

> **공공데이터 및 PyTorch 기반 타겟 반경 상권 분석 & 유망 업종 추천 시스템**

`DeepBite`는 단순히 점포 수가 많은 과당경쟁 업종(레드오션)을 추천하는 한계를 극복하고, 성북구 공공데이터의 인허가·폐업 이력을 분석하여 **실제 생존율(Survival Rate)이 검증된 알짜 유망 업종** 및 **PyTorch 기반 상권 적합도**를 산출해 주는 미세 상권 분석 서비스입니다.

---

## 🚀 주요 기능

- **📍 지도 기반 반경 타겟팅 (GIS)**: 지도의 특정 위치를 클릭하고 반경(100m ~ 1000m)을 설정하여 해당 영역 내 점포 분포를 실시간 분석
- **💡 생존율(Survival Rate) 기반 업종 추천**: 단순 빈도수 1위가 아닌, 폐업률을 극복하고 높은 생존율을 유지하는 업종을 추려내어 과당경쟁 회피 전략 제시
- **🤖 PyTorch AI 상권 평가**: 점포 밀도, 폐업 비율, 생존 지표를 입력 텐서로 활용하여 AI 상권 적합도 점수(0~100점) 계산
- **🗺️ Leaflet.js 대시보드**: 반경 내 실제 영업 중인 점포의 위치를 마커로 시각화

---

## 🛠️ 기술 스택 (Tech Stack)

- **Backend**: Python 3.11, FastAPI, PyTorch, Pandas, PyProj
- **Frontend**: HTML5, JavaScript, Leaflet.js
- **Data Pipeline**: EPSG:5181(중부원점) 좌표계 ➡️ EPSG:4326(WGS84 GPS) 변환 및 Haversine 거리 계산

---

## 🏃 실행 방법

### 1. 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 필수 라이브러리 설치
```bash
pip install fastapi uvicorn pandas torch pyproj
```

### 3. 프로젝트 실행
```bash
uvicorn main:app --reload
```
---

## 📊 데이터 출처 및 라이선스 (Data Citation & License)

### 1. 공공데이터 출처 표기
본 프로젝트는 **[공공데이터포털(data.go.kr)]** 및 **[지방행정인허가데이터개방(localdata.go.kr)]**에서 제공하는 **'서울특별시 성북구 일반음식점 인허가 정보'** 공공데이터를 가공하여 활용하였습니다.

- **데이터명**: 서울특별시 성북구 일반음식점 인허가 정보
- **제공 기관**: 서울특별시 성북구
- **이용 조건**: 공공누리 제1유형 (출처표시 조건)

### 2. Software License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.