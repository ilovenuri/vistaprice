import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
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
        
        # 필수 컬럼 확인
        required_columns = ['날짜', '시간대', '결제금액', '결제자수', '결제수']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
            st.stop()
        
        # 숫자 컬럼들의 데이터 정제
        numeric_columns = ['결제금액', '결제자수', '결제수']
        for col in numeric_columns:
            df[col] = df[col].apply(clean_numeric)
        
        # 데이터 전처리
        # 시간대에서 시간만 추출 (예: "11시" -> "11")
        df['시간'] = df['시간대'].str.extract('(\d+)').astype(int)
        
        # 날짜와 시간을 결합하여 datetime 형식으로 변환
        df['date'] = pd.to_datetime(df['날짜'].astype(str) + ' ' + df['시간'].astype(str) + ':00:00')
        
        # 분석에 사용할 지표 선택을 위한 사이드바 추가
        st.sidebar.header('분석 설정')
        
        # 분석 가능한 지표 매핑 (기본)
        metric_mapping = {
            '결제 금액': '결제금액',
            '결제자 수': '결제자수',
            '결제 건수': '결제수'
        }
        
        # 모바일웹 지표가 있는 경우 추가
        if '모바일웹(결제금액)' in df.columns:
            metric_mapping['모바일웹 결제 금액'] = '모바일웹(결제금액)'
        if '모바일웹(결제건수)' in df.columns:
            metric_mapping['모바일웹 결제 건수'] = '모바일웹(결제건수)'
        
        selected_metric = st.sidebar.selectbox(
            '예측할 지표를 선택하세요',
            list(metric_mapping.keys())
        )
        
        target_column = metric_mapping[selected_metric]
        
        # 선택된 지표가 모바일웹 관련 컬럼인 경우 숫자 변환
        if target_column in ['모바일웹(결제금액)', '모바일웹(결제건수)']:
            df[target_column] = df[target_column].apply(clean_numeric)
        
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
            title=f'{selected_metric} 예측 결과',
            xaxis_title='날짜',
            yaxis_title=selected_metric,
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
        
        # 상관관계 분석
        st.subheader('6. 상관관계 분석')
        
        # 분석할 수치형 컬럼 선택
        numeric_cols = df_daily.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ['시간']]  # '시간' 컬럼 제외
        
        # 요일 더미 변수 추가
        df_daily['요일'] = pd.to_datetime(df_daily['date']).dt.dayofweek
        df_daily['주말'] = df_daily['요일'].isin([5, 6]).astype(int)  # 토,일 = 1, 평일 = 0
        
        # 공휴일 정보 추가
        from datetime import datetime
        import holidays
        kr_holidays = holidays.KR()
        df_daily['공휴일'] = df_daily['date'].apply(lambda x: 1 if x in kr_holidays else 0)
        
        # 시간대별 집계 추가
        df_daily['아침(6-11시)'] = df['시간'].between(6, 11).astype(int).groupby(df['date'].dt.date).sum()
        df_daily['점심(12-14시)'] = df['시간'].between(12, 14).astype(int).groupby(df['date'].dt.date).sum()
        df_daily['저녁(17-21시)'] = df['시간'].between(17, 21).astype(int).groupby(df['date'].dt.date).sum()
        df_daily['심야(22-5시)'] = ((df['시간'] >= 22) | (df['시간'] <= 5)).astype(int).groupby(df['date'].dt.date).sum()
        
        # 상관계수 히트맵
        correlation_cols = ['결제금액', '결제자수', '결제수', '주말', '공휴일', 
                          '아침(6-11시)', '점심(12-14시)', '저녁(17-21시)', '심야(22-5시)']
        corr_matrix = df_daily[correlation_cols].corr()
        
        # Plotly를 사용한 히트맵 생성
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig_heatmap.update_layout(
            title='지표 간 상관관계 히트맵',
            width=800,
            height=800
        )
        
        st.plotly_chart(fig_heatmap)
        
        # 히트맵 분석 설명
        st.markdown("""
        ### 히트맵 분석 결과
        
        위 히트맵은 각 지표 간의 상관관계를 보여줍니다:
        - **빨간색**: 양의 상관관계 (값이 1에 가까울수록 강한 양의 관계)
        - **파란색**: 음의 상관관계 (값이 -1에 가까울수록 강한 음의 관계)
        - **흰색**: 상관관계가 거의 없음 (값이 0에 가까움)
        
        주요 특징:
        - 대각선은 항상 1.0 (자기 자신과의 완벽한 상관관계)
        - 대칭적인 패턴 (A와 B의 상관관계 = B와 A의 상관관계)
        """)
        
        # 주요 상관관계 설명
        st.subheader("주요 상관관계 분석")
        target_correlations = corr_matrix[target_column].sort_values(ascending=False)
        
        # 자기 자신을 제외한 상위 3개 상관관계 추출
        top_correlations = target_correlations[target_correlations.index != target_column].head(3)
        
        st.write(f"**{selected_metric}**에 영향을 미치는 주요 요인:")
        for idx, corr in top_correlations.items():
            st.write(f"- {idx}: {corr:.2f}")
            
        # 상관관계 설명
        st.markdown("""
        ### 주요 상관관계 분석 결과
        
        위 결과는 선택한 지표(**{selected_metric}**)에 가장 큰 영향을 미치는 요인들을 보여줍니다:
        
        - 상관계수가 0.7 이상: 매우 강한 관계
        - 상관계수가 0.4~0.7: 강한 관계
        - 상관계수가 0.2~0.4: 중간 정도의 관계
        - 상관계수가 0.2 미만: 약한 관계
        
        이 분석을 통해 어떤 요인이 매출에 가장 큰 영향을 미치는지 파악할 수 있습니다.
        """)
            
        # 산점도 분석
        st.subheader("상세 산점도 분석")
        
        # 상관관계가 가장 높은 변수와의 산점도
        top_factor = top_correlations.index[0]
        
        fig_scatter = px.scatter(
            df_daily,
            x=top_factor,
            y=target_column,
            trendline="ols",
            title=f"{selected_metric}와 {top_factor} 간의 관계"
        )
        
        st.plotly_chart(fig_scatter)
        
        # 산점도 분석 설명
        st.markdown(f"""
        ### 산점도 분석 결과
        
        위 그래프는 **{selected_metric}**와 가장 강한 상관관계를 보이는 **{top_factor}**의 관계를 보여줍니다:
        
        - 각 점은 하루의 데이터를 나타냅니다
        - 파란색 선은 선형 회귀선으로, 두 변수 간의 일반적인 관계를 보여줍니다
        - 점들이 선에 가깝게 분포할수록 두 변수 간의 관계가 강합니다
        - 점들이 넓게 분포할수록 두 변수 간의 관계가 약합니다
        
        이 분석을 통해 두 변수 간의 관계가 선형적인지, 그리고 어떤 방향으로 영향을 미치는지 파악할 수 있습니다.
        """)
        
        # 시간대별 평균 매출 분석
        st.subheader("시간대별 평균 분석")
        
        time_periods = ['아침(6-11시)', '점심(12-14시)', '저녁(17-21시)', '심야(22-5시)']
        avg_by_period = df_daily[time_periods].mean()
        
        fig_time = go.Figure(data=[
            go.Bar(
                x=time_periods,
                y=avg_by_period.values,
                text=np.round(avg_by_period.values, 2),
                textposition='auto',
            )
        ])
        
        fig_time.update_layout(
            title=f"시간대별 평균 거래 비중",
            xaxis_title="시간대",
            yaxis_title="평균 거래 비중"
        )
        
        st.plotly_chart(fig_time)
        
        # 시간대별 분석 설명
        st.markdown("""
        ### 시간대별 분석 결과
        
        위 그래프는 하루 중 각 시간대별 거래 비중을 보여줍니다:
        
        - 막대의 높이는 해당 시간대의 평균 거래 비중을 나타냅니다
        - 숫자는 정확한 평균값을 보여줍니다
        - 가장 높은 막대가 가장 활발한 거래가 이루어지는 시간대입니다
        
        이 분석을 통해:
        - 하루 중 가장 활발한 거래 시간대 파악
        - 시간대별 마케팅 전략 수립
        - 인력 배치 최적화
        등의 인사이트를 얻을 수 있습니다.
        """)
        
    except Exception as e:
        st.error(f'데이터 처리 중 오류가 발생했습니다: {str(e)}')
        st.error('올바른 형식의 CSV 파일을 업로드해주세요.') 