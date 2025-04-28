# 판매량 예측 앱 (Sales Forecasting App)

시계열 데이터를 기반으로 미래 판매량을 예측하는 Streamlit 웹 애플리케이션입니다.

## 주요 기능

- CSV 파일 업로드를 통한 데이터 분석
- Prophet 모델을 사용한 시계열 예측
- 7일 이동평균을 활용한 트렌드 분석
- 요일 및 휴일 효과를 고려한 예측
- 실제 데이터와 예측 결과의 시각화
- 계절성 분석 결과 제공

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
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