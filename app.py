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
import chardet
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Page config
st.set_page_config(page_title="Marketing Effect & Sales Forecast App", layout="wide")

load_dotenv()

def get_weekly_pattern(df):
    """주간 매출 패턴 분석"""
    df['day_of_week'] = pd.to_datetime(df['ds']).dt.day_name()
    weekly_avg = df.groupby('day_of_week')['y'].mean()
    peak_day = weekly_avg.idxmax()
    return f"주간 최고 매출일: {peak_day}"

def get_expected_events(forecast):
    """예상되는 주요 이벤트 분석 (정량+정성)"""
    forecast['date'] = pd.to_datetime(forecast['ds'])
    forecast['is_weekend'] = forecast['date'].dt.dayofweek >= 5
    weekend_sales = forecast[forecast['is_weekend']]['yhat'].mean()
    weekday_sales = forecast[~forecast['is_weekend']]['yhat'].mean()
    ratio = weekend_sales / weekday_sales if weekday_sales > 0 else 0
    if ratio < 0.5:
        return f"주말 매출이 주중 대비 매우 낮음 (주중 평균: {weekday_sales:,.0f}원, 주말 평균: {weekend_sales:,.0f}원)"
    elif ratio > 1.2:
        return f"주말 매출이 평일 대비 20% 이상 높음 (주중 평균: {weekday_sales:,.0f}원, 주말 평균: {weekend_sales:,.0f}원)"
    else:
        return f"주중/주말 매출이 비슷하게 유지됨 (주중 평균: {weekday_sales:,.0f}원, 주말 평균: {weekend_sales:,.0f}원)"

def get_csv_download_link(csv_string, filename):
    """Generates a link to download the CSV file"""
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">Download Sample</a>'
    return href

def prepare_data_for_prophet(df):
    # 일자별 실판매금액 합계로 Prophet용 데이터 생성
    prophet_df = df.groupby('date')['sales_amount'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

# Sample data for CSV templates
sample_sales_data = """date,store_name,product_name,quantity,sales_amount
2023-04-01,OnlineStore,Blouse,3,65000
2023-04-01,OfflineShop,Skirt,2,55000
2023-04-15,OnlineStore,Dress,4,88000
2023-04-15,OfflineShop,Blouse,3,62000
2023-05-01,OnlineStore,Skirt,4,78000
2023-05-01,OfflineShop,Dress,3,72000
2023-05-15,OnlineStore,Blouse,5,95000
2023-05-15,OfflineShop,Skirt,4,82000
2023-06-01,OnlineStore,Dress,6,115000
2023-06-01,OfflineShop,Blouse,4,85000
2023-06-15,OnlineStore,Skirt,5,92000
2023-06-15,OfflineShop,Dress,4,88000
2023-07-01,OnlineStore,Blouse,7,125000
2023-07-01,OfflineShop,Skirt,5,98000
2023-07-15,OnlineStore,Dress,6,118000
2023-07-15,OfflineShop,Blouse,5,102000
2023-08-01,OnlineStore,Skirt,8,142000
2023-08-01,OfflineShop,Dress,6,122000
2023-08-15,OnlineStore,Blouse,7,132000
2023-08-15,OfflineShop,Skirt,6,115000
2023-09-01,OnlineStore,Dress,6,120000
2023-09-01,OfflineShop,Blouse,5,105000
2023-09-15,OnlineStore,Skirt,5,98000
2023-09-15,OfflineShop,Dress,4,92000
2023-10-01,OnlineStore,Blouse,6,118000
2023-10-01,OfflineShop,Skirt,5,102000
2023-10-15,OnlineStore,Dress,7,128000
2023-10-15,OfflineShop,Blouse,6,115000
2023-11-01,OnlineStore,Skirt,8,145000
2023-11-01,OfflineShop,Dress,7,132000
2023-11-15,OnlineStore,Blouse,9,162000
2023-11-15,OfflineShop,Skirt,7,138000
2023-12-01,OnlineStore,Dress,10,180000
2023-12-01,OfflineShop,Blouse,8,155000
2023-12-15,OnlineStore,Skirt,9,165000
2023-12-15,OfflineShop,Dress,8,148000
2024-01-01,OnlineStore,Blouse,8,145000
2024-01-01,OfflineShop,Skirt,6,122000
2024-01-15,OnlineStore,Dress,7,135000
2024-01-15,OfflineShop,Blouse,6,118000
2024-02-01,OnlineStore,Skirt,7,132000
2024-02-01,OfflineShop,Dress,6,125000
2024-02-15,OnlineStore,Blouse,8,148000
2024-02-15,OfflineShop,Skirt,7,135000
2024-03-01,OnlineStore,Dress,9,165000
2024-03-01,OfflineShop,Blouse,7,142000
2024-03-15,OnlineStore,Skirt,8,152000
2024-03-15,OfflineShop,Dress,7,138000
"""

sample_marketing_data = """date,keyword,channel_name,channel_type,impressions,clicks,ctr,total_cost,avg_rank,conversions,conversion_rate,conversion_sales
2023-04-01,tumbler,NaverSearch,search,1000,50,5.0,50000,2.0,2,4.0,100000
2023-04-01,tumbler,NaverShopping,shopping,800,30,3.8,30000,2.2,1,3.3,50000
2023-05-01,tumbler,NaverSearch,search,1200,60,5.0,52000,1.9,3,5.0,140000
2023-05-01,tumbler,NaverShopping,shopping,900,35,3.9,32000,2.1,2,5.7,80000
2023-06-01,tumbler,NaverSearch,search,1400,70,5.0,55000,1.8,4,5.7,180000
2023-06-01,tumbler,NaverShopping,shopping,1000,40,4.0,35000,2.0,2,5.0,90000
2023-07-01,tumbler,NaverSearch,search,1600,85,5.3,58000,1.7,5,5.9,220000
2023-07-01,tumbler,NaverShopping,shopping,1200,48,4.0,38000,1.9,3,6.3,120000
2023-08-01,tumbler,NaverSearch,search,1800,95,5.3,62000,1.6,6,6.3,260000
2023-08-01,tumbler,NaverShopping,shopping,1400,58,4.1,42000,1.8,4,6.9,160000
2023-09-01,tumbler,NaverSearch,search,1600,80,5.0,58000,1.7,5,6.3,220000
2023-09-01,tumbler,NaverShopping,shopping,1200,45,3.8,38000,1.9,3,6.7,130000
2023-10-01,tumbler,NaverSearch,search,1800,90,5.0,60000,1.6,6,6.7,250000
2023-10-01,tumbler,NaverShopping,shopping,1400,55,3.9,40000,1.8,4,7.3,170000
2023-11-01,tumbler,NaverSearch,search,2000,105,5.3,65000,1.5,7,6.7,290000
2023-11-01,tumbler,NaverShopping,shopping,1600,65,4.1,45000,1.7,5,7.7,200000
2023-12-01,tumbler,NaverSearch,search,2500,135,5.4,72000,1.4,9,6.7,380000
2023-12-01,tumbler,NaverShopping,shopping,2000,85,4.3,52000,1.6,7,8.2,280000
2024-01-01,tumbler,NaverSearch,search,2200,115,5.2,68000,1.5,8,7.0,320000
2024-01-01,tumbler,NaverShopping,shopping,1800,72,4.0,48000,1.7,6,8.3,240000
2024-02-01,tumbler,NaverSearch,search,2000,105,5.3,65000,1.6,7,6.7,290000
2024-02-01,tumbler,NaverShopping,shopping,1600,65,4.1,45000,1.8,5,7.7,200000
2024-03-01,tumbler,NaverSearch,search,2200,118,5.4,70000,1.5,8,6.8,330000
2024-03-01,tumbler,NaverShopping,shopping,1800,75,4.2,50000,1.7,6,8.0,250000
"""

sample_promotion_data = """date,event_name,event_type,discount_rate
2023-04-01,SpringSale,seasonal,20
2023-04-15,EarthDay,holiday,15
2023-05-01,MayDay,holiday,20
2023-05-15,FamilyDay,holiday,25
2023-06-01,SummerSale,seasonal,30
2023-06-15,MidYearSale,discount,25
2023-07-01,VacationSale,seasonal,20
2023-07-15,HotSummerDeal,discount,15
2023-08-01,BackToSchool,seasonal,25
2023-08-15,IndependenceDay,holiday,20
2023-09-01,FallSale,seasonal,20
2023-09-15,ChuseokSale,holiday,30
2023-10-01,HarvestSale,seasonal,15
2023-10-31,HalloweenSale,holiday,25
2023-11-01,WinterSale,seasonal,20
2023-11-15,BlackFriday,discount,35
2023-12-01,YearEndSale,seasonal,30
2023-12-25,ChristmasSale,holiday,25
2024-01-01,NewYearSale,holiday,30
2024-01-15,WinterSpecial,seasonal,25
2024-02-01,LunarNewYear,holiday,30
2024-02-14,ValentineDay,holiday,15
2024-03-01,SpringSale,seasonal,20
2024-03-15,WhiteDay,holiday,15
"""

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-align: center;
    }
    .upload-section {
        background-color: transparent;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
    }
    .section-title {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .upload-text {
        color: #cccccc;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .download-link {
        color: #4CAF50;
        text-decoration: none;
        font-size: 0.9rem;
        padding: 0.3rem 0;
        display: inline-block;
    }
    .download-link:hover {
        color: #45a049;
        text-decoration: underline;
    }
    .stFileUploader {
        padding: 1rem 0;
    }
    /* Remove black background from file uploader */
    .uploadedFile {
        background-color: transparent !important;
    }
    .stFileUploader > div {
        background-color: transparent !important;
    }
    .stFileUploader > button {
        background-color: rgba(49, 51, 63, 0.2) !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
    }
    /* Style the drag and drop area */
    .stFileUploader > div[data-testid="stFileUploadDropzone"] {
        background-color: rgba(49, 51, 63, 0.1) !important;
        border: 1px dashed rgba(49, 51, 63, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Marketing Effect & Sales Forecast App</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Data upload section
st.markdown("### Data Upload")

# Create columns for file uploaders
col1, col2, col3 = st.columns(3)

# Sales data upload
with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sales</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">Upload your sales CSV file</div>', unsafe_allow_html=True)
    sales_file = st.file_uploader("", type=['csv'], key='sales_uploader')
    st.markdown(get_csv_download_link(sample_sales_data, "sales_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Marketing data upload
with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Marketing</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">Upload your marketing CSV file</div>', unsafe_allow_html=True)
    marketing_file = st.file_uploader("", type=['csv'], key='marketing_uploader')
    st.markdown(get_csv_download_link(sample_marketing_data, "marketing_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Promotion data upload
with col3:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Promotion</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">Upload your promotion CSV file</div>', unsafe_allow_html=True)
    promotion_file = st.file_uploader("", type=['csv'], key='promotion_uploader')
    st.markdown(get_csv_download_link(sample_promotion_data, "promotion_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Data Processing
if sales_file and marketing_file and promotion_file:
    try:
        def read_csv_auto_encoding(file):
            file.seek(0)
            raw = file.read(10000)
            if isinstance(raw, str):
                raw = raw.encode('utf-8')
            result = chardet.detect(raw)
            encoding = result['encoding'] if result['encoding'] else 'utf-8'
            file.seek(0)
            content = file.read()
            if isinstance(content, str):
                content = content.encode('utf-8')
            decoded = content.decode(encoding)
            return pd.read_csv(io.StringIO(decoded))

        # Read and clean files before any further processing
        sales_df = read_csv_auto_encoding(sales_file)
        marketing_df = read_csv_auto_encoding(marketing_file)
        promotion_df = read_csv_auto_encoding(promotion_file)

        def clean_columns(df):
            df.columns = (
                df.columns
                .str.replace('\ufeff', '', regex=True)  # BOM 제거
                .str.replace(' ', '', regex=True)       # 일반 공백 제거
                .str.replace('\xa0', '', regex=True)    # non-breaking space 제거
                .str.strip()                            # 앞뒤 공백 제거
            )
            return df

        sales_df = clean_columns(sales_df)
        marketing_df = clean_columns(marketing_df)
        promotion_df = clean_columns(promotion_df)
        
        # Convert date columns to datetime
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        marketing_df['date'] = pd.to_datetime(marketing_df['date'])
        promotion_df['date'] = pd.to_datetime(promotion_df['date'])
        
        # sales_amount에서 쉼표 제거 및 숫자형 변환
        sales_df['sales_amount'] = sales_df['sales_amount'].astype(str).str.replace(',', '').astype(float)
        
        # Display the data in tabs
        tab4, tab1, tab2, tab3 = st.tabs(["Forecast", "Sales", "Marketing", "Promotion"])
        
        with tab4:
            st.subheader("Sales Forecast")
            st.markdown("""
                **분석 방법:**<br>
                본 예측은 Facebook Prophet 시계열 모델을 사용하여, 일별 실판매금액의 추세와 계절성을 반영해 향후 매출을 예측합니다.<br>
                예측 결과는 아래 그래프와 표로 확인할 수 있습니다. (예상 매출이 음수로 예측될 경우 0으로 보정하여 표시합니다.)
            """, unsafe_allow_html=True)

            # Forecast period selection
            forecast_period_options = {
                "30일": 30,
                "60일": 60,
                "90일": 90
            }
            selected_period = st.selectbox(
                "예측 기간을 선택하세요",
                options=list(forecast_period_options.keys()),
                index=0
            )
            forecast_days = forecast_period_options[selected_period]

            # Train Prophet model
            sales_prophet = prepare_data_for_prophet(sales_df)
            
            # 데이터 기간이 충분한지 확인
            date_range = (sales_prophet['ds'].max() - sales_prophet['ds'].min()).days
            
            # 이상치 제거 (1~99퍼센타일로 clip)
            q_low = sales_prophet['y'].quantile(0.01)
            q_high = sales_prophet['y'].quantile(0.99)
            sales_prophet['y'] = sales_prophet['y'].clip(lower=q_low, upper=q_high)
            
            # Prophet 모델 파라미터 조정 (신뢰구간 개선)
            if date_range < 30:  # 데이터가 30일 미만인 경우
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    growth='linear',
                    changepoint_prior_scale=0.01,   # 더 작게
                    seasonality_prior_scale=2.0,    # 더 작게
                    seasonality_mode='multiplicative',
                    interval_width=0.80             # 더 좁게
                )
            else:
                model = Prophet(
                    yearly_seasonality=True if date_range > 365 else False,
                    weekly_seasonality=True,
                    daily_seasonality=True if date_range > 7 else False,
                    growth='linear',
                    changepoint_prior_scale=0.01,   # 더 작게
                    seasonality_prior_scale=2.0,    # 더 작게
                    seasonality_mode='multiplicative',
                    interval_width=0.80             # 더 좁게
                )
            
            # 과거 데이터 통계 계산
            min_sales = sales_prophet['y'].quantile(0.25)  # 25퍼센타일
            max_sales = sales_prophet['y'].quantile(0.95)  # 95퍼센타일
            mean_sales = sales_prophet['y'].mean()
            std_sales = sales_prophet['y'].std()
            
            # 예측 상한값 설정 (평균 + 2 표준편차 또는 95퍼센타일 중 큰 값)
            upper_bound = max(mean_sales + 2 * std_sales, max_sales)
            
            model.fit(sales_prophet)
            
            # 예측 기간 설정 (사용자가 선택한 기간 그대로 사용)
            forecast_periods = forecast_days
            
            # Prophet의 make_future_dataframe() 함수 사용
            future = model.make_future_dataframe(periods=forecast_periods, freq='D')
            
            # 디버깅을 위한 future 데이터프레임 정보 출력
            st.write("Future DataFrame Info:")
            st.write(f"Shape: {future.shape}")
            st.write(f"Date Range: {future['ds'].min()} to {future['ds'].max()}")
            st.write(f"NaN values in ds: {future['ds'].isna().sum()}")
            
            forecast = model.predict(future)
            
            # 예측값 범위 조정
            forecast['yhat'] = forecast['yhat'].clip(lower=min_sales, upper=upper_bound)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_sales)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=forecast['yhat'], upper=upper_bound)

            # Add warning for short data periods
            if date_range < 30:
                st.warning("""
                ⚠️ 주의: 현재 데이터가 30일 미만입니다. 정확한 예측을 위해서는 최소 30일 이상의 데이터가 필요합니다.
                현재 예측은 제한된 데이터를 기반으로 한 참고용 수치입니다.
                """)

            # Calculate trend analysis
            current_avg = sales_prophet['y'].mean()
            future_avg = forecast['yhat'].tail(forecast_days).mean()
            trend_direction = "상승" if future_avg > current_avg else "하락"
            trend_percentage = abs((future_avg - current_avg) / current_avg * 100)

            # Generate insights
            st.markdown("### 📊 매출 분석 및 인사이트")
            
            # Current performance analysis
            st.markdown("#### 1. 현재 매출 현황")
            st.markdown(f"""
            - **평균 일일 매출**: {current_avg:,.0f}원
            - **향후 {forecast_days}일 매출 추세**: {trend_direction}세 ({trend_percentage:.1f}%)
            - **주간 패턴**: {get_weekly_pattern(sales_prophet)}
            """)

            # Future outlook
            st.markdown(f"#### 2. 향후 {forecast_days}일 전망")
            st.markdown(f"""
            - **예상 평균 일일 매출**: {future_avg:,.0f}원
            - **예상 매출 범위**: {forecast['yhat_lower'].tail(forecast_days).mean():,.0f}원 ~ {forecast['yhat_upper'].tail(forecast_days).mean():,.0f}원
            - **주요 예상 이벤트**: {get_expected_events(forecast)}
            """)

            # Improvement recommendations
            st.markdown("#### 3. 매출 개선 방안")
            if trend_direction == "하락":
                st.markdown("""
                - **프로모션 전략 강화**
                  - 주말/휴일 특별 할인 이벤트
                  - VIP 고객 대상 맞춤형 프로모션
                  - 신규 고객 유치를 위한 입문 패키지
                
                - **마케팅 채널 다각화**
                  - 소셜 미디어 광고 집행
                  - 이메일 마케팅 캠페인 강화
                  - 인플루언서 마케팅 도입 검토
                
                - **고객 경험 개선**
                  - 리워드 프로그램 도입
                  - 구매 후 리뷰 이벤트
                  - 맞춤형 추천 시스템 도입
                """)
            else:
                st.markdown("""
                - **현재 성공 요인 강화**
                  - 인기 상품 재고 확보
                  - 고객 만족도 높은 서비스 유지
                  - 성공적인 프로모션 전략 지속
                
                - **신규 기회 포착**
                  - 신규 시장 진출 검토
                  - 제품 라인업 확장
                  - 고객 세그먼트 확대
                """)

            # --- 모든 분석/표/프롬프트/AI 인사이트 코드를 try 블록 안에만 위치 ---
            forecast['date'] = pd.to_datetime(forecast['ds'])
            forecast['is_weekend'] = forecast['date'].dt.dayofweek >= 5
            weekend_avg = forecast[forecast['is_weekend']]['yhat'].mean()
            weekday_avg = forecast[~forecast['is_weekend']]['yhat'].mean()
            weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 0
            weekend_diff = weekend_avg - weekday_avg
            future_avg = forecast['yhat'].tail(forecast_days).mean()
            trend_direction = "상승" if future_avg > current_avg else "하락"
            trend_percentage = abs((future_avg - current_avg) / current_avg * 100)
            st.markdown(f"""
            ### 📊 매출 분석 및 인사이트
            - **평균 일일 매출**: {current_avg:,.0f}원
            - **향후 {forecast_days}일 매출 추세**: {trend_direction}세 ({trend_percentage:.1f}%)
            - **주간 패턴**: {get_weekly_pattern(sales_prophet)}
            - **예상 평균 일일 매출**: {future_avg:,.0f}원
            - **예상 매출 범위**: {forecast['yhat_lower'].tail(forecast_days).mean():,.0f}원 ~ {forecast['yhat_upper'].tail(forecast_days).mean():,.0f}원
            - **주요 예상 이벤트**: {get_expected_events(forecast)}
            #### 📊 주중/주말 매출 비교 (예측값 기준)
            | 구분 | 평균 매출(원) |
            |------|--------------|
            | 주중 | {weekday_avg:,.0f} |
            | 주말 | {weekend_avg:,.0f} |
            - **주말/주중 비율**: {weekend_ratio:.2%}
            - **차이**: {weekend_diff:,.0f}원
            향후 {forecast_days}일 예측 평균 매출: {future_avg:,.0f}원
            주중 평균 매출: {weekday_avg:,.0f}원
            주말 평균 매출: {weekend_avg:,.0f}원
            주말/주중 비율: {weekend_ratio:.2%}
            매출 추세: {trend_direction} ({trend_percentage:.1f}%)
            """)
            if weekend_ratio < 0.5:
                st.markdown("""
                **해석:**  
                - 주말 매출이 주중 대비 매우 낮습니다.  
                - 주말 프로모션, 이벤트, 광고 강화 필요
                """)
            elif weekend_ratio > 1.2:
                st.markdown("""
                **해석:**  
                - 주말 매출이 주중 대비 20% 이상 높습니다.  
                - 주말 집중 마케팅, 인기 상품 재고 확보가 중요
                """)
            else:
                st.markdown("""
                **해석:**  
                - 주중/주말 매출이 비슷하게 유지되고 있습니다.  
                - 전체적인 마케팅 균형 유지
                """)
            # --- Gemini AI 인사이트 생성 기능 전체 try 블록 안에 위치 ---
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            def get_gemini_insight(prompt, api_key):
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                return response.text
            analysis_prompt = f"""
            [매출 데이터 요약]
            - 최근 30일 평균 매출: {current_avg:,.0f}원
            - 향후 {forecast_days}일 예측 평균 매출: {future_avg:,.0f}원
            - 주중 평균 매출: {weekday_avg:,.0f}원
            - 주말 평균 매출: {weekend_avg:,.0f}원
            - 주말/주중 비율: {weekend_ratio:.2%}
            - 매출 추세: {trend_direction} ({trend_percentage:.1f}%)
            - 주요 이벤트: {get_expected_events(forecast)}

            [요구사항]
            위 데이터를 바탕으로, 아래 4가지 항목을 포함한 전략 리포트 형식의 인사이트를 10문장 이상으로 작성해줘.

            1. 최근 매출 변화의 주요 원인 분석 (패턴, 이상점, 고객 행동 등)
            2. 사업적 영향 및 리스크 평가 (지속 시나리오, 현금흐름, 브랜드 영향 등)
            3. 실질적 개선 전략 제안 (프로모션, 마케팅, 신제품, 타깃 마케팅 등)
            4. 추가 데이터 수집/분석 제안 (고객 피드백, 경쟁사, 외부 데이터 등)

            각 항목별로 구체적이고 실질적인 내용을 포함해, 경영자가 바로 실행에 옮길 수 있도록 작성해줘.
            """
            if st.button("Gemini AI 인사이트 생성"):
                with st.spinner("Gemini가 분석 중입니다..."):
                    ai_insight = get_gemini_insight(analysis_prompt, gemini_api_key)
                st.markdown("#### 🤖 Gemini 기반 매출 인사이트")
                st.write(ai_insight)

            # Plot forecast
            fig = go.Figure()

            # 전체 데이터 준비 (실제 데이터 + 예측 데이터)
            all_dates = pd.concat([sales_prophet['ds'], forecast['ds']]).unique()
            date_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')

            # Actual values
            fig.add_trace(go.Scatter(x=sales_prophet['ds'], 
                                   y=sales_prophet['y'],
                                   name='Actual Sales',
                                   mode='markers+lines'))

            # Predicted values (전체 예측 기간 표시)
            fig.add_trace(go.Scatter(x=forecast['ds'],
                                   y=forecast['yhat'],
                                   name='Forecast Sales',
                                   mode='lines',
                                   line=dict(dash='dash')))

            # Confidence interval
            fig.add_trace(go.Scatter(x=forecast['ds'],
                                   y=forecast['yhat_upper'],
                                   fill=None,
                                   mode='lines',
                                   line=dict(width=0),
                                   showlegend=False))

            fig.add_trace(go.Scatter(x=forecast['ds'],
                                   y=forecast['yhat_lower'],
                                   fill='tonexty',
                                   mode='lines',
                                   line=dict(width=0),
                                   name='95% CI'))

            # Update layout with extended date range
            fig.update_layout(
                title=f'Sales Forecast (Next {forecast_days} Days)',
                xaxis_title='Date',
                yaxis_title='Sales Amount',
                hovermode='x unified',
                xaxis=dict(
                    range=[date_range.min(), date_range.max()],
                    type='date'
                ),
                showlegend=True
            )

            # 그래프 범위 강제 설정
            fig.update_xaxes(range=[date_range.min(), date_range.max()])
            
            st.plotly_chart(fig, use_container_width=True)

            # Show forecast table (future only)
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_days)
            forecast_table = forecast_table.rename(columns={
                'ds': '날짜',
                'yhat': '예상 매출',
                'yhat_lower': '최소 예측값',
                'yhat_upper': '최대 예측값'
            })
            
            # 날짜 형식 변경
            forecast_table['날짜'] = forecast_table['날짜'].dt.strftime('%Y-%m-%d')
            
            # 숫자에 천 단위 구분자 추가
            for col in ['예상 매출', '최소 예측값', '최대 예측값']:
                forecast_table[col] = forecast_table[col].apply(lambda x: f"{x:,.0f}")
            
            st.markdown(f"#### 4. 예측 결과 테이블 (향후 {forecast_days}일)")
            st.dataframe(forecast_table, use_container_width=True)

        with tab1:
            st.subheader("Sales Data")
            st.write(sales_df)
            
            # Sales trend visualization
            fig = px.line(sales_df, x='date', y='sales_amount', color='store_name',
                         title='Sales Trend by Store')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Marketing Data")
            st.write(marketing_df)
            
            # Marketing cost by channel visualization
            fig = px.bar(marketing_df, x='date', y='total_cost', color='channel_name',
                        title='Marketing Cost by Channel')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Promotion Data")
            st.write(promotion_df)
            
            # Promotion events visualization
            fig = px.bar(promotion_df, x='date', y='discount_rate', color='event_name',
                        title='Discount Rate by Event')
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.data_loaded = True

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please make sure your CSV files match the sample template format.")
        if 'sales_prophet' in locals():
            st.write("[디버깅] sales_prophet.head():", sales_prophet.head())
            st.write("[디버깅] sales_prophet NaN count:", sales_prophet.isna().sum())
        if 'future' in locals():
            st.write("[디버깅] future.head():", future.head())
            st.write("[디버깅] future NaN count:", future.isna().sum())
        if 'forecast' in locals():
            st.write("[디버깅] forecast.head():", forecast.head())
            st.write("[디버깅] forecast NaN count:", forecast.isna().sum())
        st.stop()
