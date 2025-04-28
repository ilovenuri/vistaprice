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

# Page config
st.set_page_config(page_title="마케팅 효과 분석 및 판매량 예측 앱", layout="wide")

def get_csv_download_link(csv_string, filename):
    """Generates a link to download the CSV file"""
    b64 = base64.b64encode(csv_string.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-link">샘플 다운로드</a>'
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
    st.markdown('<div class="section-title">판촉행사</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-text">CSV 파일을 업로드하세요</div>', unsafe_allow_html=True)
    promotion_file = st.file_uploader("", type=['csv'], key='promotion_uploader')
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
        sales_df['판매일자'] = pd.to_datetime(sales_df['판매일자'])
        if 'date' in marketing_df.columns:
            marketing_df['date'] = pd.to_datetime(marketing_df['date'])
        if 'date' in promotion_df.columns:
            promotion_df['date'] = pd.to_datetime(promotion_df['date'])
        
        # Display the data in tabs
        tab1, tab2, tab3 = st.tabs(["매출", "마케팅", "판촉행사"])
        
        with tab1:
            st.subheader("매출")
            st.write(sales_df)
            
            # Sales trend visualization
            fig = px.line(sales_df, x='판매일자', y='실판매금액', color='매장구분',
                         title='매장별 매출 추이')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("마케팅")
            st.write(marketing_df)
            
            # Marketing cost by channel visualization
            fig = px.bar(marketing_df, x='date', y='marketing_cost', color='marketing_channel',
                        title='채널별 마케팅 비용')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("판촉행사")
            st.write(promotion_df)
            
            # Promotion events visualization
            fig = px.scatter(promotion_df, x='date', y='discount_rate', 
                           color='event_type', size='discount_rate',
                           title='판촉행사 및 할인율')
            st.plotly_chart(fig, use_container_width=True)

        st.session_state.data_loaded = True

    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.info("CSV 파일이 샘플 템플릿 형식과 일치하는지 확인해주세요.") 