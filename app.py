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
        tab1, tab2, tab3, tab4 = st.tabs(["Sales", "Marketing", "Promotion", "Forecast"])
        
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
            
        with tab4:
            st.subheader("Sales Forecast")
            
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

        st.session_state.data_loaded = True

    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.info("Please make sure your CSV files match the sample template format.") 