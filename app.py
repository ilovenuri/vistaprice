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
st.set_page_config(page_title="마케팅 효과 분석 및 판매량 예측 앱", layout="wide")

def get_csv_download_link(csv_string, filename):
    """Generates a link to download the CSV file"""
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">샘플 다운로드</a>'
    return href

def prepare_data_for_prophet(df):
    # 일자별 실판매금액 합계로 Prophet용 데이터 생성
    prophet_df = df.groupby('판매일자')['실판매금액'].sum().reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

# Sample data for CSV templates
sample_sales_data = """판매일자,매장명,상품명,판매수량,실판매금액
2024-01-01,온라인몰,블라우스,2,40500
2024-01-01,오프라인점,스커트,1,49500
2024-01-02,온라인몰,원피스,3,67500
"""

sample_marketing_data = """판매일자,키워드,매체이름,검색/콘텐츠매체,노출수,클릭수,클릭률(%),총비용(VAT포함,원),평균노출순위,전환수,전환율(%),전환매출액(원)
2024-01-01,텀블러,네이버통합검색,검색,1000,50,5.0,50000,1.8,2,4.0,100000
2024-01-01,텀블러,네이버쇼핑검색,쇼핑,800,30,3.75,30000,2.1,1,3.3,50000
"""

sample_promotion_data = """판매일자,이벤트명,종류,할인율(%)
2024-01-01,신년할인,할인,20
2024-01-02,1+1이벤트,번들,50
2024-01-03,브랜드데이,브랜드,30
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
st.markdown('<h1 class="main-header">마케팅 효과 분석 및 판매량 예측 앱</h1>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Data upload section
st.markdown("### 데이터 업로드")

# Create columns for file uploaders
col1, col2, col3 = st.columns(3)

# Sales data upload
with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">매출</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">CSV 파일을 업로드하세요</div>', unsafe_allow_html=True)
    sales_file = st.file_uploader("", type=['csv'], key='sales_uploader')
    st.markdown(get_csv_download_link(sample_sales_data, "sales_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Marketing data upload
with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">마케팅 비용</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">CSV 파일을 업로드하세요</div>', unsafe_allow_html=True)
    marketing_file = st.file_uploader("", type=['csv'], key='marketing_uploader')
    st.markdown(get_csv_download_link(sample_marketing_data, "marketing_template.csv"), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Promotion data upload
with col3:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">프로모션</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">CSV 파일을 업로드하세요</div>', unsafe_allow_html=True)
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
        sales_df['판매일자'] = pd.to_datetime(sales_df['판매일자'])
        marketing_df['판매일자'] = pd.to_datetime(marketing_df['판매일자'])
        promotion_df['판매일자'] = pd.to_datetime(promotion_df['판매일자'])
        
        # Display the data in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["매출", "마케팅", "프로모션", "예측"])
        
        with tab1:
            st.subheader("매출 데이터")
            st.write(sales_df)
            
            # Sales trend visualization
            fig = px.line(sales_df, x='판매일자', y='실판매금액', color='매장명',
                         title='매장별 매출 추이')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("마케팅 데이터")
            st.write(marketing_df)
            
            # Marketing cost by channel visualization
            fig = px.bar(marketing_df, x='판매일자', y='총비용(VAT포함,원)', color='매체이름',
                        title='매체별 마케팅 비용')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("프로모션 데이터")
            st.write(promotion_df)
            
            # Promotion events visualization
            fig = px.bar(promotion_df, x='판매일자', y='할인율(%)', color='이벤트명',
                        title='이벤트별 할인율')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            st.subheader("매출 예측")
            
            # Train Prophet model
            sales_prophet = prepare_data_for_prophet(sales_df)
            model = Prophet(yearly_seasonality=True, 
                          weekly_seasonality=True, 
                          daily_seasonality=True)
            model.fit(sales_prophet)
            
            # Create future dates for prediction
            future_dates = model.make_future_dataframe(periods=30)
            forecast = model.predict(future_dates)
            
            # Plot forecast
            fig = go.Figure()
            
            # Actual values
            fig.add_trace(go.Scatter(x=sales_prophet['ds'], 
                                   y=sales_prophet['y'],
                                   name='실제 매출',
                                   mode='markers+lines'))
            
            # Predicted values
            fig.add_trace(go.Scatter(x=forecast['ds'],
                                   y=forecast['yhat'],
                                   name='예측 매출',
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
                                   name='95% 신뢰구간'))
            
            fig.update_layout(title='매출 예측 (향후 30일)',
                            xaxis_title='날짜',
                            yaxis_title='매출액',
                            hovermode='x unified')
            
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.data_loaded = True

    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("CSV 파일이 샘플 템플릿 형식과 일치하는지 확인해주세요.") 