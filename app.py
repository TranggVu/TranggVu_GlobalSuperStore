import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import plotly.graph_objects as go

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Superstore Forecast", layout="wide")


@st.cache_resource
def load_assets():
    # Load mô hình
    model = xgb.XGBRegressor()
    model.load_model("model_monthly.json")
    # Load features & regions
    with open('features.json', 'r') as f:
        features = json.load(f)
    with open('regions.json', 'r') as f:
        regions = json.load(f)
    return model, features, regions


model, features, regions = load_assets()
history_df = pd.read_csv("monthly_history.csv")
history_df['ds'] = pd.to_datetime(history_df['ds'])

# --- GIAO DIỆN SIDEBAR ---
st.sidebar.header("🕹️ Bảng điều khiển")
selected_region = st.sidebar.selectbox("Chọn khu vực kinh doanh:", regions)
forecast_date = st.sidebar.date_input("Chọn tháng muốn dự báo:", pd.to_datetime("2015-01-01"))


# --- XỬ LÝ LOGIC DỰ BÁO ---
def prepare_input(region, date, hist):
    # Lấy dữ liệu lịch sử của vùng đó
    df_reg = hist[hist['region'] == region].sort_values('ds')

    # Tạo row mới cho tháng dự báo
    new_row = {'ds': pd.to_datetime(date), 'region': region, 'month': date.month}

    # Tính toán các Feature giống hệt lúc Train
    month = date.month
    input_data = {
        'month': month,
        'quarter': (month - 1) // 3 + 1,
        'year': date.year,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12)
    }

    # Lấy các giá trị Lag từ lịch sử
    # Lưu ý: Trong thực tế bạn cần lấy y_log của các tháng gần nhất trong hist
    last_y_logs = np.log1p(df_reg['y'].tail(12).values)

    input_data['lag_1'] = last_y_logs[-1]
    input_data['lag_2'] = last_y_logs[-2]
    input_data['lag_3'] = last_y_logs[-3]
    input_data['lag_6'] = last_y_logs[-6]
    input_data['lag_12'] = last_y_logs[-12]

    input_data['roll_mean_3'] = np.mean(last_y_logs[-3:])
    input_data['roll_mean_6'] = np.mean(last_y_logs[-6:])
    input_data['roll_std_3'] = np.std(last_y_logs[-3:])
    input_data['diff_1_3'] = last_y_logs[-1] - last_y_logs[-3]
    input_data['diff_1_12'] = last_y_logs[-1] - last_y_logs[-12]

    # Xử lý One-hot Encoding cho Region
    for feat in features:
        if feat.startswith('region_'):
            region_name = feat.replace('region_', '')
            input_data[feat] = 1 if region == region_name else 0

    return pd.DataFrame([input_data])[features]


# --- HIỂN THỊ KẾT QUẢ ---
st.title("📈 Dự báo Doanh thu Tháng - Superstore")
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("🚀 Chạy dự báo"):
        X_val = prepare_input(selected_region, forecast_date, history_df)
        pred_log = model.predict(X_val)
        prediction = np.expm1(pred_log)[0]

        st.metric(label=f"Doanh thu dự kiến {selected_region}", value=f"${prediction:,.2f}")
        st.info(f"Tháng dự báo: {forecast_date.strftime('%m/%Y')}")

with col2:
    # Vẽ biểu đồ lịch sử của vùng đó
    df_plot = history_df[history_df['region'] == selected_region].tail(12)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['ds'], y=df_plot['y'], name='Lịch sử 12 tháng'))
    fig.update_layout(title=f"Diễn biến doanh thu tại {selected_region}", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)