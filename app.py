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

def prepare_data_for_prophet(df, date_column, value_column):
    """Prophet 모델을 위한 데이터 전처리"""
    prophet_df = df.groupby(date_column)[value_column].sum().reset_index()
    prophet_df.columns = ['ds', 'y']  # Prophet requires these specific column names
    return prophet_df

# Sample data for CSV templates
sample_sales_data = """⇅,판매일자,매장구분,매장코드,매장명,영수증번호,기획구분,판매유형,팀구분,상품코드,상품명,칼라,칼라명,사이즈,사이즈명,자사바코드,판매구분,매출구분,TAG가,사전원가,평균원가,현재가,판매예정단가,판매수량,할인(T.실)%,에누리,마일리지,상품권,즉시환급,판매단가,판매금액,실판매금액,순판매(할인제외),순판매(할인포함),수수료,TAG가합,사전원가합,평균원가합,행사구분,마진(%),중간관리(%),고객코드,고객명,판매원,수취인/주문인,특이사항,특이사항2,영수증특이사항,MD Concept Story,원구매정보,입력시간,영수증번호.1
1,2024-01-01,온라인,ST001,온라인스토어,RN001,일반,정상판매,여성의류,P001,블라우스,BK,블랙,S,Small,B001,판매,일반,50000,30000,32000,45000,45000,1,10,0,0,0,0,40500,40500,40500,45000,40500,2000,50000,30000,32000,일반,19.0,5.0,C001,홍길동,판매자1,수취인1,,,,,,,09:00,RN001
1,2024-01-01,온라인,ST001,온라인스토어,RN002,일반,정상판매,여성의류,P002,스커트,NV,네이비,M,Medium,B002,판매,일반,60000,35000,37000,55000,55000,1,10,0,0,0,0,49500,49500,49500,55000,49500,2500,60000,35000,37000,일반,18.0,5.0,C002,김철수,판매자1,수취인2,,,,,,,10:30,RN002
1,2024-01-01,온라인,ST001,온라인스토어,RN003,일반,정상판매,여성의류,P003,원피스,WH,화이트,L,Large,B003,판매,일반,80000,45000,47000,75000,75000,1,10,0,0,0,0,67500,67500,67500,75000,67500,3000,80000,45000,47000,일반,20.0,5.0,C003,이영희,판매자2,수취인3,,,,,,,14:15,RN003"""

sample_marketing_data = """판매일자,url,impressions,clicks,click_rate,cost,vat_included_cost,avg_cpc,conversions,conversion_amount
2024-01-01,https://smart.example.com/ad1,1000,50,5.0,50000,55000,1000,2,100000
2024-01-02,https://smart.example.com/ad2,1500,75,5.0,75000,82500,1000,3,150000
2024-01-03,https://smart.example.com/ad3,2000,100,5.0,100000,110000,1000,4,200000"""

sample_promotion_data = """판매일자,promotion_event,event_type,discount_rate
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
        # Read the uploaded files with automatic encoding detection
        def read_csv_auto_encoding(file):
            file.seek(0)
            try:
                return pd.read_csv(file, encoding='utf-8')
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding='cp949')
        
        sales_df = read_csv_auto_encoding(sales_file)
        marketing_df = read_csv_auto_encoding(marketing_file)
        promotion_df = read_csv_auto_encoding(promotion_file)
        
        # Debug: Print column names
        st.write("Sales DataFrame columns:", sales_df.columns.tolist())
        st.write("Marketing DataFrame columns:", marketing_df.columns.tolist())
        st.write("Promotion DataFrame columns:", promotion_df.columns.tolist())
        
        # Convert date columns to datetime
        sales_df['판매일자'] = pd.to_datetime(sales_df['판매일자'])
        
        # Check if '판매일자' exists in marketing_df
        if '판매일자' not in marketing_df.columns and 'date' in marketing_df.columns:
            marketing_df = marketing_df.rename(columns={'date': '판매일자'})
        marketing_df['판매일자'] = pd.to_datetime(marketing_df['판매일자'])
        
        # Check if '판매일자' exists in promotion_df
        if '판매일자' not in promotion_df.columns and 'date' in promotion_df.columns:
            promotion_df = promotion_df.rename(columns={'date': '판매일자'})
        promotion_df['판매일자'] = pd.to_datetime(promotion_df['판매일자'])
        
        # Prepare data for Prophet
        sales_prophet = prepare_data_for_prophet(sales_df, '판매일자', '실판매금액')
        
        # Display the data in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["매출", "마케팅", "판촉행사", "예측"])
        
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
            fig = px.bar(marketing_df, x='판매일자', y='cost', color='url',
                        title='채널별 마케팅 비용')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("판촉행사")
            st.write(promotion_df)
            
            # Promotion events visualization
            fig = px.scatter(promotion_df, x='판매일자', y='discount_rate', 
                           color='event_type', size='discount_rate',
                           title='판촉행사 및 할인율')
            st.plotly_chart(fig, use_container_width=True)
            
        with tab4:
            st.subheader("매출 예측")
            
            # Train Prophet model
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