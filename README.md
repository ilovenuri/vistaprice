# 판매량 예측 앱 (Sales Forecasting App)

시계열 데이터를 기반으로 미래 판매량을 예측하는 Streamlit 웹 애플리케이션입니다.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vistaprice.streamlit.app)

## 데모

웹 브라우저에서 바로 사용해보실 수 있습니다: [Streamlit Cloud Demo](https://vistaprice.streamlit.app)
※ Gemini AI 기능은 로컬에서만 작동합니다.

## 주요 기능

- CSV 파일 업로드를 통한 데이터 분석
- Prophet 모델을 사용한 시계열 예측
- Gemini AI를 활용한 매출 전략 리포트 자동 생성 (경영 의사결정에 바로 활용 가능한 구체적 인사이트 제공)
- 7일 이동평균을 활용한 트렌드 분석
- 요일 및 휴일 효과를 고려한 예측
- 실제 데이터와 예측 결과의 시각화
- 계절성 분석 결과 제공

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/ilovenuri/vistaprice.git
cd sales_forecast_app
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
streamlit run app.py
```

## 실행 전 환경설정 (.env 파일)

Gemini AI 인사이트 기능을 사용하려면, 프로젝트 루트 또는 sales_forecast_app 폴더 내에 `.env` 파일을 생성하고 아래와 같이 본인의 Gemini API 키를 입력해야 합니다.

```
GOOGLE_API_KEY=여기에_본인의_Gemini_API_키_입력
```

## 입력 데이터 형식

CSV 파일은 다음과 같은 컬럼을 포함해야 합니다:

- `날짜`: YYYY-MM-DD 형식의 날짜
- `시간대`: "HH시" 형식의 시간
- `결제금액`: 숫자 (쉼표 포함 가능)
- `결제자수`: 숫자
- `결제수`: 숫자
- 기타 관련 지표들

## 주요 기술 스택

- Python 3.8+
- Streamlit 1.32.0
- Prophet 1.1.5
- Pandas 2.2.0
- Plotly 5.18.0

## 라이선스

MIT License

## 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 

예측 결과 하단의 "Gemini AI 인사이트 생성" 버튼을 누르면, 최근 매출 변화 원인, 사업적 영향, 실질적 개선 전략, 추가 분석 제안까지 포함된 전략 리포트가 자동 생성됩니다. 