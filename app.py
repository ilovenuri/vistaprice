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

# Page config
st.set_page_config(page_title="Marketing Effect & Sales Forecast App", layout="wide")

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
2024-01-01,OnlineStore,Blouse,2,40500
2024-01-01,OfflineShop,Skirt,1,49500
2024-01-02,OnlineStore,Dress,3,67500
"""

sample_marketing_data = """date,keyword,channel_name,channel_type,impressions,clicks,ctr,total_cost,avg_rank,conversions,conversion_rate,conversion_sales
2024-01-01,tumbler,NaverSearch,search,1000,50,5.0,50000,1.8,2,4.0,100000
2024-01-01,tumbler,NaverShopping,shopping,800,30,3.75,30000,2.1,1,3.3,50000
"""

sample_promotion_data = """date,event_name,event_type,discount_rate
2024-01-01,NewYearSale,discount,20
2024-01-02,1plus1Event,bundle,50
2024-01-03,BrandDay,brand,30
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

        # Debug: Print cleaned column names
        st.write("Cleaned Sales DataFrame columns:", sales_df.columns.tolist())
        st.write("Cleaned Marketing DataFrame columns:", marketing_df.columns.tolist())
        st.write("Cleaned Promotion DataFrame columns:", promotion_df.columns.tolist())
        
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
                본 예측은 Facebook Prophet 시계열 모델을 사용하여, 일별 실판매금액의 추세와 계절성을 반영해 향후 30일간의 매출을 예측합니다.<br>
                예측 결과는 아래 그래프와 표로 확인할 수 있습니다. (예상 매출이 음수로 예측될 경우 0으로 보정하여 표시합니다.)
            """, unsafe_allow_html=True)

            # Train Prophet model
            sales_prophet = prepare_data_for_prophet(sales_df)
            model = Prophet(yearly_seasonality=True, 
                          weekly_seasonality=True, 
                          daily_seasonality=True)
            model.fit(sales_prophet)
            # Create future dates for prediction
            future_dates = model.make_future_dataframe(periods=30)
            forecast = model.predict(future_dates)

            # Clip negative predictions to zero
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)

            # Calculate trend analysis
            current_avg = sales_prophet['y'].mean()
            future_avg = forecast['yhat'].tail(30).mean()
            trend_direction = "상승" if future_avg > current_avg else "하락"
            trend_percentage = abs((future_avg - current_avg) / current_avg * 100)

            # Generate insights
            st.markdown("### 📊 매출 분석 및 인사이트")
            
            # Current performance analysis
            st.markdown("#### 1. 현재 매출 현황")
            st.markdown(f"""
            - **평균 일일 매출**: {current_avg:,.0f}원
            - **최근 30일 매출 추세**: {trend_direction}세 ({trend_percentage:.1f}%)
            - **주간 패턴**: {get_weekly_pattern(sales_prophet)}
            """)

            # Future outlook
            st.markdown("#### 2. 향후 30일 전망")
            st.markdown(f"""
            - **예상 평균 일일 매출**: {future_avg:,.0f}원
            - **예상 매출 범위**: {forecast['yhat_lower'].tail(30).mean():,.0f}원 ~ {forecast['yhat_upper'].tail(30).mean():,.0f}원
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

            # Plot forecast
            fig = go.Figure()
            # Actual values
            fig.add_trace(go.Scatter(x=sales_prophet['ds'], 
                                   y=sales_prophet['y'],
                                   name='Actual Sales',
                                   mode='markers+lines'))
            # Predicted values
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
            fig.update_layout(title='Sales Forecast (Next 30 Days)',
                            xaxis_title='Date',
                            yaxis_title='Sales Amount',
                            hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Show forecast table (future only)
            forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
            forecast_table = forecast_table.rename(columns={
                'ds': 'Date',
                'yhat': 'Predicted Sales',
                'yhat_lower': 'Lower Bound',
                'yhat_upper': 'Upper Bound'
            })
            st.markdown("#### 4. 예측 결과 테이블 (향후 30일)")
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
        st.info("Please make sure your CSV files match the sample template format.") # force update from desktop copy

def get_weekly_pattern(df):
    """주간 매출 패턴 분석"""
    df['day_of_week'] = pd.to_datetime(df['ds']).dt.day_name()
    weekly_avg = df.groupby('day_of_week')['y'].mean()
    peak_day = weekly_avg.idxmax()
    return f"주간 최고 매출일: {peak_day}"

def get_expected_events(forecast):
    """예상되는 주요 이벤트 분석"""
    # 주말/휴일 식별
    forecast['date'] = pd.to_datetime(forecast['ds'])
    forecast['is_weekend'] = forecast['date'].dt.dayofweek >= 5
    weekend_sales = forecast[forecast['is_weekend']]['yhat'].mean()
    weekday_sales = forecast[~forecast['is_weekend']]['yhat'].mean()
    
    if weekend_sales > weekday_sales * 1.2:
        return "주말 매출이 평일 대비 20% 이상 높을 것으로 예상"
    else:
        return "주중/주말 매출이 안정적으로 유지될 것으로 예상"
