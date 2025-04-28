import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 앱 제목 설정
st.title('판매량 예측 앱')

# 파일 업로드 섹션
st.header('1. 데이터 업로드')
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])

if uploaded_file is not None:
    try:
        # CSV 파일 읽기
        df = pd.read_csv(uploaded_file)
        
        # 컬럼명 매핑 정의
        expected_columns = [
            '날짜', '요일', '시간대', '결제자수', '결제수', 
            '모바일웹(결제건수)', '결제금액', '모바일웹(결제금액)', 
            '결제취소금액', '결제건수상세', '결제취소건수', 
            '환불건수', '환불금액', '환불비율(결제건수)', '환불비율(결제금액건수)'
        ]
        
        # 데이터 미리보기
        st.subheader('원본 데이터 미리보기')
        st.write(df.head())
        
        # 숫자 데이터 전처리 함수
        def clean_numeric(x):
            if pd.isna(x):
                return 0
            if isinstance(x, str):
                # 쉼표 제거하고 공백으로 분리된 경우 첫 번째 값만 사용
                return float(x.replace(',', '').split()[0])
            return float(x)
        
        # 숫자 컬럼들의 데이터 정제
        numeric_columns = ['결제금액', '결제자수', '결제수', '모바일웹(결제건수)', '모바일웹(결제금액)']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_numeric)
        
        # 데이터 전처리
        # 시간대에서 시간만 추출 (예: "11시" -> "11")
        df['시간'] = df['시간대'].str.extract('(\d+)').astype(int)
        
        # 날짜와 시간을 결합하여 datetime 형식으로 변환
        df['date'] = pd.to_datetime(df['날짜'].astype(str) + ' ' + df['시간'].astype(str) + ':00:00')
        
        # 분석에 사용할 지표 선택을 위한 사이드바 추가
        st.sidebar.header('분석 설정')
        target_column = st.sidebar.selectbox(
            '예측할 지표를 선택하세요',
            numeric_columns
        )
        
        # 선택된 지표로 일별 집계 및 이동평균 계산
        df_daily = df.groupby(df['date'].dt.date).agg({
            target_column: 'sum',
            '결제자수': 'sum',
            '결제수': 'sum',
            '결제금액': 'sum'
        }).reset_index()
        
        # date 컬럼을 datetime 형식으로 변환
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        # 7일 이동평균 계산
        df_daily['MA7'] = df_daily[target_column].rolling(window=7, center=True).mean()
        
        # 이상치 제거 (상위 99퍼센타일 초과 또는 하위 1퍼센타일 미만 제거)
        lower_bound = df_daily[target_column].quantile(0.01)
        upper_bound = df_daily[target_column].quantile(0.99)
        df_daily = df_daily[
            (df_daily[target_column] >= lower_bound) & 
            (df_daily[target_column] <= upper_bound)
        ]
        
        st.subheader('전처리된 데이터 미리보기')
        st.write(df_daily.head())
        
        # 기본 통계 정보 표시
        st.subheader('기본 통계 정보')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("평균 일별 결제금액", f"{df_daily['결제금액'].mean():,.0f}원")
        with col2:
            st.metric("평균 일별 결제자수", f"{df_daily['결제자수'].mean():,.0f}명")
        with col3:
            st.metric("평균 일별 결제수", f"{df_daily['결제수'].mean():,.0f}건")
        
        # Prophet 모델을 위한 데이터 전처리
        df_prophet = df_daily.rename(columns={
            'date': 'ds',
            target_column: 'y'
        })
        
        # 이동평균을 기반으로 한 최소/최대값 설정
        min_y = df_prophet['y'].rolling(window=30, min_periods=1).min().min()
        max_y = df_prophet['y'].rolling(window=30, min_periods=1).max().max()
        
        # Prophet 모델 학습
        st.subheader('2. 예측 모델 학습')
        model = Prophet(
            growth='flat',  # 선형 성장 대신 평탄화된 성장 사용
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,  # 일별 계절성 비활성화
            changepoint_prior_scale=0.001,  # 변화점 민감도를 매우 낮게 설정
            seasonality_prior_scale=0.1,    # 계절성 강도를 낮게 설정
            changepoint_range=0.95,         # 변화점 범위
            n_changepoints=15,              # 변화점 수를 줄임
            interval_width=0.95             # 예측 구간을 95%로 설정
        )
        
        # 휴일 효과 추가
        model.add_country_holidays(country_name='KR')
        
        # 요일 더미 변수 추가
        df_prophet['weekday'] = df_prophet['ds'].dt.dayofweek
        for i in range(7):
            df_prophet[f'weekday_{i}'] = (df_prophet['weekday'] == i).astype(int)
            model.add_regressor(f'weekday_{i}')
        
        # 이동평균을 리그레서로 추가
        df_prophet['MA7'] = df_prophet['y'].rolling(window=7, min_periods=1).mean()
        model.add_regressor('MA7')
        
        model.fit(df_prophet)
        
        # 예측 기간 설정
        forecast_period = st.sidebar.slider('예측 기간 (일)', 7, 90, 30)
        
        # 예측 수행
        future_dates = model.make_future_dataframe(periods=forecast_period)
        
        # 미래 데이터에 대한 리그레서 값 설정
        future_dates['weekday'] = future_dates['ds'].dt.dayofweek
        for i in range(7):
            future_dates[f'weekday_{i}'] = (future_dates['weekday'] == i).astype(int)
        
        # 이동평균의 마지막 값을 미래 데이터에 적용
        last_ma = df_prophet['MA7'].iloc[-1]
        future_dates['MA7'] = last_ma
        
        forecast = model.predict(future_dates)
        
        # 예측값이 음수인 경우 0으로 대체
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        
        # 극단적인 예측값 제한
        forecast['yhat'] = forecast['yhat'].clip(upper=max_y * 1.2)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_y * 1.5)
        
        # 예측 결과 시각화
        st.subheader('3. 예측 결과')
        
        # 실제 데이터와 예측 결과를 하나의 그래프로 표시
        fig = go.Figure()
        
        # 실제 데이터
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['y'],
            name='실제 데이터',
            line=dict(color='blue')
        ))
        
        # 7일 이동평균
        fig.add_trace(go.Scatter(
            x=df_prophet['ds'],
            y=df_prophet['MA7'],
            name='7일 이동평균',
            line=dict(color='green', dash='dash')
        ))
        
        # 예측 결과
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='예측값',
            line=dict(color='red')
        ))
        
        # 예측 구간
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='상위 예측 구간'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(255,0,0,0.2)',
            name='하위 예측 구간'
        ))
        
        # 평균선 추가
        mean_value = df_prophet['y'].mean()
        fig.add_hline(y=mean_value, line_dash="dash", line_color="gray", 
                     annotation_text=f"전체 평균: {mean_value:,.0f}")
        
        fig.update_layout(
            title=f'{target_column} 예측 결과',
            xaxis_title='날짜',
            yaxis_title=target_column,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # 예측 결과 테이블
        st.subheader('4. 예측 결과 상세')
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period)
        forecast_display.columns = ['날짜', '예측값', '최소 예측값', '최대 예측값']
        forecast_display = forecast_display.round(0)  # 소수점 제거
        st.dataframe(forecast_display)
        
        # 계절성 분석 결과
        st.subheader('5. 계절성 분석')
        fig_components = model.plot_components(forecast)
        st.pyplot(fig_components)
        
    except Exception as e:
        st.error(f'데이터 처리 중 오류가 발생했습니다: {str(e)}')
        st.error('올바른 형식의 CSV 파일을 업로드해주세요.') 