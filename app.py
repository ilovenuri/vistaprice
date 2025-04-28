import streamlit as st
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def get_csv_download_link(csv_string, filename):
    """Generates a link to download the CSV file"""
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Sample {filename}</a>'
    return href

# Sample data for CSV templates
sample_sales_data = """date,brand,sales
2024-01-01,BrandA,100
2024-01-02,BrandA,150
2024-01-03,BrandA,120"""

sample_marketing_data = """date,marketing_cost,marketing_channel
2024-01-01,1000000,TV
2024-01-02,500000,SNS
2024-01-03,800000,Display"""

sample_promotion_data = """date,promotion_event,event_type,discount_rate
2024-01-01,New Year Discount,Discount,20
2024-01-02,1+1 Event,Bundle,50
2024-01-03,Brand Day,Brand,30"""

# Page config
st.set_page_config(page_title="마케팅 효과 분석 및 판매량 예측 앱", layout="wide")

# Custom CSS for better layout
st.markdown("""
    <style>
    .upload-section {
        padding: 20px;
        margin: 10px 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
        color: #0e1117;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("마케팅 효과 분석 및 판매량 예측 앱")

# Create three columns for the file uploaders
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">SALES DATA</p>', unsafe_allow_html=True)
    sales_file = st.file_uploader("Upload Sales Data", type=['csv'], key='sales_upload')
    st.markdown(get_csv_download_link(sample_sales_data, "sales_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">MARKETING DATA</p>', unsafe_allow_html=True)
    marketing_file = st.file_uploader("Upload Marketing Data", type=['csv'], key='marketing_upload')
    st.markdown(get_csv_download_link(sample_marketing_data, "marketing_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">PROMOTION DATA</p>', unsafe_allow_html=True)
    promotion_file = st.file_uploader("Upload Promotion Data", type=['csv'], key='promotion_upload')
    st.markdown(get_csv_download_link(sample_promotion_data, "promotion_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Data Processing
if sales_file and marketing_file and promotion_file:
    try:
        # Read the uploaded files
        sales_df = pd.read_csv(sales_file)
        marketing_df = pd.read_csv(marketing_file)
        promotion_df = pd.read_csv(promotion_file)
        
        # Convert date columns to datetime
        for df in [sales_df, marketing_df, promotion_df]:
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
        
        # Display the data in tabs
        tab1, tab2, tab3 = st.tabs(["Sales Data", "Marketing Data", "Promotion Data"])
        
        with tab1:
            st.subheader("Sales Data")
            st.write(sales_df)
            
            # Sales trend visualization
            fig = px.line(sales_df, x='date', y='sales', color='brand',
                         title='Sales Trend by Brand')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Marketing Data")
            st.write(marketing_df)
            
            # Marketing cost by channel visualization
            fig = px.bar(marketing_df, x='date', y='marketing_cost', color='marketing_channel',
                        title='Marketing Cost by Channel')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Promotion Data")
            st.write(promotion_df)
            
            # Promotion events visualization
            fig = px.scatter(promotion_df, x='date', y='discount_rate', 
                           color='event_type', size='discount_rate',
                           title='Promotion Events and Discount Rates')
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please make sure your CSV files match the template format.")

# 앱 제목 설정
st.title('마케팅 효과 분석 및 판매량 예측 앱')

# 파일 업로드 섹션
st.header('1. 데이터 업로드')

# 세션 상태 초기화
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'marketing_data' not in st.session_state:
    st.session_state.marketing_data = None
if 'promotion_data' not in st.session_state:
    st.session_state.promotion_data = None

# 각 데이터 타입별 파일 업로더
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("매출 데이터")
    sales_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'], key='sales')

with col2:
    st.subheader("마케팅 비용 데이터")
    marketing_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'], key='marketing')

with col3:
    st.subheader("판촉행사 데이터")
    promotion_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'], key='promotion')

# 데이터 로드 및 전처리
if sales_file:
    st.session_state.sales_data = pd.read_csv(sales_file)
    # 날짜 컬럼이 있는 경우 datetime으로 변환
    if '날짜' in st.session_state.sales_data.columns:
        st.session_state.sales_data['날짜'] = pd.to_datetime(st.session_state.sales_data['날짜'])

if marketing_file:
    st.session_state.marketing_data = pd.read_csv(marketing_file)
    if '날짜' in st.session_state.marketing_data.columns:
        st.session_state.marketing_data['날짜'] = pd.to_datetime(st.session_state.marketing_data['날짜'])

if promotion_file:
    st.session_state.promotion_data = pd.read_csv(promotion_file)
    if '날짜' in st.session_state.promotion_data.columns:
        st.session_state.promotion_data['날짜'] = pd.to_datetime(st.session_state.promotion_data['날짜'])

# 데이터가 모두 업로드되었는지 확인
if st.session_state.sales_data is not None:
    # 사이드바에 브랜드 필터 추가
    st.sidebar.header('브랜드 필터')
    
    # 브랜드 컬럼이 있는 경우
    if '브랜드' in st.session_state.sales_data.columns:
        brands = st.session_state.sales_data['브랜드'].unique()
        selected_brands = st.sidebar.multiselect(
            '분석할 브랜드를 선택하세요',
            options=brands,
            default=brands[:3] if len(brands) > 3 else brands
        )
        
        # 선택된 브랜드로 데이터 필터링
        filtered_sales = st.session_state.sales_data[st.session_state.sales_data['브랜드'].isin(selected_brands)]
    else:
        filtered_sales = st.session_state.sales_data
    
    # 데이터 미리보기
    st.subheader('원본 데이터 미리보기')
    st.write(filtered_sales.head())
    
    # 기본 통계 정보 표시
    st.subheader('기본 통계 정보')
    
    # 브랜드별 통계
    if '브랜드' in filtered_sales.columns:
        brand_stats = filtered_sales.groupby('브랜드').agg({
            '결제금액': ['mean', 'sum', 'count'],
            '결제자수': ['mean', 'sum'],
            '결제수': ['mean', 'sum']
        }).round(2)
        st.write('브랜드별 통계')
        st.dataframe(brand_stats)
    
    # 마케팅 비용과 판촉행사 데이터가 있는 경우 상관관계 분석
    if st.session_state.marketing_data is not None and st.session_state.promotion_data is not None:
        st.subheader('마케팅 효과 분석')
        
        # 마케팅 비용과 매출의 상관관계
        if '마케팅비용' in st.session_state.marketing_data.columns:
            marketing_corr = pd.merge(
                filtered_sales.groupby('날짜')['결제금액'].sum().reset_index(),
                st.session_state.marketing_data,
                on='날짜',
                how='inner'
            )
            
            # 상관계수 계산
            correlation = marketing_corr['결제금액'].corr(marketing_corr['마케팅비용'])
            
            # 상관관계 시각화
            fig = px.scatter(
                marketing_corr,
                x='마케팅비용',
                y='결제금액',
                title=f'마케팅 비용과 매출의 상관관계 (상관계수: {correlation:.2f})'
            )
            st.plotly_chart(fig)
            
            # 마케팅 ROI 계산
            marketing_corr['ROI'] = (marketing_corr['결제금액'] - marketing_corr['마케팅비용']) / marketing_corr['마케팅비용']
            st.write('마케팅 ROI 통계')
            st.write(marketing_corr['ROI'].describe())
        
        # 판촉행사 효과 분석
        if '판촉행사' in st.session_state.promotion_data.columns:
            promotion_effect = pd.merge(
                filtered_sales.groupby('날짜')['결제금액'].sum().reset_index(),
                st.session_state.promotion_data,
                on='날짜',
                how='left'
            )
            
            # 판촉행사가 있는 날과 없는 날의 매출 비교
            promotion_days = promotion_effect[promotion_effect['판촉행사'].notna()]
            non_promotion_days = promotion_effect[promotion_effect['판촉행사'].isna()]
            
            st.write('판촉행사 효과 분석')
            col1, col2 = st.columns(2)
            with col1:
                st.write('판촉행사일 평균 매출')
                st.write(promotion_days['결제금액'].mean())
            with col2:
                st.write('일반일 평균 매출')
                st.write(non_promotion_days['결제금액'].mean())
            
            # 판촉행사 효과 시각화
            fig = px.box(
                promotion_effect,
                x='판촉행사',
                y='결제금액',
                title='판촉행사 유무에 따른 매출 분포'
            )
            st.plotly_chart(fig)
    
    # 가격대별 분석
    if '가격' in filtered_sales.columns:
        st.subheader('가격대별 분석')
        
        # 가격대 구간 설정
        price_bins = [0, 10000, 20000, 30000, 40000, 50000, float('inf')]
        price_labels = ['1만원 미만', '1-2만원', '2-3만원', '3-4만원', '4-5만원', '5만원 이상']
        
        filtered_sales['가격대'] = pd.cut(filtered_sales['가격'], bins=price_bins, labels=price_labels)
        
        # 가격대별 판매량 및 매출 분석
        price_analysis = filtered_sales.groupby('가격대').agg({
            '결제수': 'sum',
            '결제금액': 'sum'
        }).reset_index()
        
        fig = px.bar(
            price_analysis,
            x='가격대',
            y='결제수',
            title='가격대별 판매량'
        )
        st.plotly_chart(fig)
        
        fig = px.bar(
            price_analysis,
            x='가격대',
            y='결제금액',
            title='가격대별 매출'
        )
        st.plotly_chart(fig)
    
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
    missing_columns = [col for col in required_columns if col not in filtered_sales.columns]
    if missing_columns:
        st.error(f"필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
        st.stop()
    
    # 숫자 컬럼들의 데이터 정제
    numeric_columns = ['결제금액', '결제자수', '결제수']
    for col in numeric_columns:
        filtered_sales[col] = filtered_sales[col].apply(clean_numeric)
    
    # 데이터 전처리
    # 시간대에서 시간만 추출 (예: "11시" -> "11")
    filtered_sales['시간'] = filtered_sales['시간대'].str.extract('(\d+)').astype(int)
    
    # 날짜와 시간을 결합하여 datetime 형식으로 변환
    filtered_sales['date'] = pd.to_datetime(filtered_sales['날짜'].astype(str) + ' ' + filtered_sales['시간'].astype(str) + ':00:00')
    
    # 분석에 사용할 지표 선택을 위한 사이드바 추가
    st.sidebar.header('분석 설정')
    
    # 목차를 사이드바에 추가
    st.sidebar.markdown("""
    ## 목차
    1. 데이터 전처리
    2. 예측 모델 학습
    3. 예측 결과
    4. 예측 결과 상세
    5. 계절성 분석
    """)
    
    # 분석 가능한 지표 매핑 (기본)
    metric_mapping = {
        '결제 금액': '결제금액',
        '결제자 수': '결제자수',
        '결제 건수': '결제수'
    }
    
    # 모바일웹 지표가 있는 경우 추가
    if '모바일웹(결제금액)' in filtered_sales.columns:
        metric_mapping['모바일웹 결제 금액'] = '모바일웹(결제금액)'
    if '모바일웹(결제건수)' in filtered_sales.columns:
        metric_mapping['모바일웹 결제 건수'] = '모바일웹(결제건수)'
    
    selected_metric = st.sidebar.selectbox(
        '예측할 지표를 선택하세요',
        list(metric_mapping.keys())
    )
    
    target_column = metric_mapping[selected_metric]
    
    # 선택된 지표가 모바일웹 관련 컬럼인 경우 숫자 변환
    if target_column in ['모바일웹(결제금액)', '모바일웹(결제건수)']:
        filtered_sales[target_column] = filtered_sales[target_column].apply(clean_numeric)
    
    # 선택된 지표로 일별 집계 및 이동평균 계산
    df_daily = filtered_sales.groupby(filtered_sales['date'].dt.date).agg({
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
    st.markdown("""
    ### Prophet 모델 학습 과정
    
    Prophet은 Facebook에서 개발한 시계열 예측 모델로, 다음과 같은 특징을 가지고 있습니다:
    
    - **계절성 분석**: 연간, 주간, 일간 패턴을 자동으로 감지
    - **휴일 효과**: 한국의 공휴일을 자동으로 반영
    - **이상치 처리**: 갑작스러운 변화나 특이점을 자동으로 감지
    - **불확실성 추정**: 예측값의 신뢰 구간을 제공
    
    현재 설정된 모델 파라미터:
    - **로지스틱 성장**: 매출의 상한선과 하한선을 고려한 성장 모델
    - **연간 계절성**: 연간 패턴 분석 활성화
    - **주간 계절성**: 요일별 패턴 분석 활성화
    - **변화점 감지**: 데이터의 급격한 변화를 자동으로 감지
    """)
    
    model = Prophet(
        growth='logistic',  # 로지스틱 성장 사용
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.01,  # 변화점 민감도 조정
        seasonality_prior_scale=0.1,  # 계절성 강도 조정
        changepoint_range=0.8,  # 변화점 범위 조정
        n_changepoints=15,  # 변화점 수 조정
        interval_width=0.95,
        seasonality_mode='multiplicative'
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
    
    # 로지스틱 성장을 위한 cap과 floor 설정
    df_prophet['cap'] = df_prophet['y'].max() * 1.5
    df_prophet['floor'] = df_prophet['y'].min() * 0.5
    
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
    
    # 로지스틱 성장을 위한 cap과 floor 설정
    future_dates['cap'] = df_prophet['cap'].iloc[-1]
    future_dates['floor'] = df_prophet['floor'].iloc[-1]
    
    forecast = model.predict(future_dates)
    
    # 예측값이 음수인 경우 최소값으로 대체
    min_value = df_prophet['y'].min() * 0.5
    forecast['yhat'] = forecast['yhat'].clip(lower=min_value)
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_value)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=min_value)
    
    # 극단적인 예측값 제한
    max_value = df_prophet['y'].max() * 1.2
    forecast['yhat'] = forecast['yhat'].clip(upper=max_value)
    forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_value * 1.5)
    
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
    
    # 계절성 분석 설명
    st.markdown("""
    ### 계절성 분석 결과 설명
    
    #### 1. 추세 (Trend) 분석
    이 그래프는 장기적인 매출 변화를 보여줍니다. 
    - 상승 추세: 매출이 꾸준히 증가하고 있음을 의미합니다. 이는 마케팅 전략이 효과적이거나, 고객 기반이 확대되고 있음을 시사합니다.
    - 하락 추세: 시장 상황이 악화되거나, 경쟁사가 강화되었을 수 있습니다.
    - 급격한 변화: 특별한 이벤트나 마케팅 캠페인의 영향일 수 있습니다.
    
    #### 2. 주간 계절성 (Weekly) 분석
    이 그래프는 요일별 매출 패턴을 보여줍니다.
    - 주말(토,일)이 높은 경우: 주말에 더 많은 고객이 방문하는 패턴입니다. 주말 특별 프로모션이나 이벤트를 고려해볼 수 있습니다.
    - 평일이 높은 경우: 직장인이나 학생들의 방문이 많을 수 있습니다. 평일 특별 메뉴나 할인을 고려해볼 수 있습니다.
    - 특정 요일이 높은 경우: 그 요일에 특별한 이벤트나 프로모션이 있을 수 있습니다.
    
    #### 3. 연간 계절성 (Yearly) 분석
    이 그래프는 연중 매출 패턴을 보여줍니다.
    - 계절별 패턴: 특정 계절에 매출이 높다면 계절성 상품이나 서비스의 영향일 수 있습니다.
    - 월별 변화: 특정 달에 급격한 변화가 있다면 휴일이나 특별한 이벤트의 영향일 수 있습니다.
    - 연말 효과: 12월에 매출이 급증한다면 연말 특별 프로모션의 효과일 수 있습니다.
    
    #### 4. 예측 구간 (Forecast) 분석
    - 넓은 예측 구간: 불확실성이 높다는 의미입니다. 이는 시장 상황이 불안정하거나, 데이터가 부족할 수 있습니다.
    - 좁은 예측 구간: 예측이 상대적으로 안정적이라는 의미입니다.
    - 예측 구간 이탈: 실제 매출이 예측 구간을 벗어난다면 예상치 못한 이벤트나 변화가 발생했을 수 있습니다.
    
    ### 구체적인 제안

    #### 1. 마케팅 전략 수립 시기
    - **매출 상승기**: 예측된 매출 상승 시기에 맞춰 신규 프로모션 출시
    - **매출 정체기**: 고객 유지 전략 강화 및 신규 고객 유치 캠페인
    - **매출 하락기**: 충성도 높은 고객 대상 특별 혜택 제공

    #### 2. 인력 배치 최적화
    - **주말 대응**: 주말 매출이 높은 경우, 주말 인력 수요 예측 및 스케줄링
    - **피크타임 대응**: 시간대별 매출 분석을 통한 인력 배치 최적화
    - **특별 이벤트**: 예측된 매출 급증 시기에 맞춰 임시 인력 채용 계획

    #### 3. 재고 관리 계획
    - **계절성 상품**: 계절별 매출 패턴에 맞춰 재고 수준 조정
    - **인기 상품**: 매출 예측을 기반으로 한 안전 재고 수준 설정
    - **신규 상품**: 예측된 매출 증가 시기에 맞춰 신규 상품 출시

    #### 4. 특별 프로모션 기획
    - **주말 프로모션**: 주말 매출이 높은 경우, 주말 특별 할인 이벤트
    - **계절성 프로모션**: 계절별 매출 패턴에 맞춰 시즌 특별 프로모션
    - **고객 세그먼트**: 요일별/시간대별 고객 특성에 맞춰 맞춤형 프로모션
    """)
    
else:
    st.error("데이터가 업로드되지 않았습니다. 데이터를 업로드하세요.") 