import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =========================================================
# 1. CONFIG
# =========================================================
st.set_page_config(
    page_title="Retail Forecast & Profit Optimization",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --bg: #f4f7fb;
    --card: #ffffff;
    --ink: #1f2937;
    --muted: #6b7280;
    --brand: #0f4c81;
    --brand-2: #1f78b4;
    --accent: #10b981;
    --warn: #f59e0b;
    --danger: #ef4444;
    --line: #e5e7eb;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #f8fbff 0%, #f4f7fb 100%);
}

.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid var(--line);
}

.hero-card {
    background: linear-gradient(135deg, #0f4c81 0%, #1f78b4 100%);
    color: white;
    border-radius: 18px;
    padding: 24px 28px;
    box-shadow: 0 10px 28px rgba(15, 76, 129, 0.18);
    margin-bottom: 20px;
}

.hero-title {
    font-size: 15px;
    opacity: 0.85;
    margin-bottom: 8px;
}

.hero-value {
    font-size: 40px;
    font-weight: 700;
    line-height: 1.1;
    margin: 0;
}

.hero-sub {
    margin-top: 8px;
    font-size: 14px;
    opacity: 0.9;
}

.section-title {
    font-size: 28px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 6px;
}



.soft-card {
    background: var(--card);
    border: 1px solid rgba(229, 231, 235, 0.8);
    border-radius: 18px;
    padding: 18px 18px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
}

.kpi-card {
    background: var(--card);
    border: 1px solid rgba(229, 231, 235, 0.9);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
    min-height: 132px;
}

.kpi-label {
    color: var(--muted);
    font-size: 14px;
    margin-bottom: 10px;
}

.kpi-value {
    color: var(--ink);
    font-size: 28px;
    font-weight: 700;
    line-height: 1.1;
}

.kpi-badge-good, .kpi-badge-warn, .kpi-badge-bad {
    display: inline-block;
    margin-top: 12px;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
}

.kpi-badge-good {
    background: rgba(16, 185, 129, 0.12);
    color: #047857;
}

.kpi-badge-warn {
    background: rgba(245, 158, 11, 0.12);
    color: #b45309;
}

.kpi-badge-bad {
    background: rgba(239, 68, 68, 0.12);
    color: #b91c1c;
}

.note-card {
    background: #f8fbff;
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 14px 16px;
    color: #1e3a8a;
    font-size: 14px;
}

.success-note {
    background: #ecfdf5;
    border: 1px solid #bbf7d0;
    color: #047857;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 14px;
}

.warn-note {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #c2410c;
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 14px;
}

.insight-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}

.small-muted {
    color: var(--muted);
    font-size: 13px;
}

div[data-testid="stMetric"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background: #ffffff !important;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: var(--brand);
    color: white;
}

hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 18px 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. LOAD ASSETS
# =========================================================
@st.cache_resource
def load_assets():
    # Lấy đường dẫn thư mục chứa file webdemo.py
    # Giúp app load file đúng dù bạn chạy Streamlit từ thư mục nào
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_DIR = BASE_DIR / "models"
    CONFIG_DIR = BASE_DIR / "configs"
    DATA_DIR = BASE_DIR / "data"

    # Load mô hình Sales
    m_sales = xgb.XGBRegressor()
    m_sales.load_model(str(MODEL_DIR / "model_monthly.json"))

    # Load mô hình Margin
    m_margin = xgb.XGBRegressor()
    m_margin.load_model(str(MODEL_DIR / "model_margin.json"))

    # Load danh sách features và cấu hình giao diện
    with open(CONFIG_DIR / "features.json", "r", encoding="utf-8") as f:
        f_sales = json.load(f)

    with open(CONFIG_DIR / "features_margin.json", "r", encoding="utf-8") as f:
        f_margin = json.load(f)

    with open(CONFIG_DIR / "regions.json", "r", encoding="utf-8") as f:
        regions = json.load(f)

    with open(CONFIG_DIR / "ship_modes.json", "r", encoding="utf-8") as f:
        ship_modes = json.load(f)

    with open(CONFIG_DIR / "categories.json", "r", encoding="utf-8") as f:
        categories = json.load(f)

    # Load dữ liệu lịch sử và các bảng tra cứu
    history = pd.read_csv(DATA_DIR / "monthly_history.csv")
    history["ds"] = pd.to_datetime(history["ds"])

    if "y_log" in history.columns:
        history["Sales"] = np.expm1(history["y_log"])

    ship_ratio_df = pd.read_csv(DATA_DIR / "ship_ratio_profile.csv")
    ship_money_df = pd.read_csv(DATA_DIR / "ship_money_profile.csv")
    cat_sales_avg_df = pd.read_csv(DATA_DIR / "cat_sales_avg.csv")

    # Giải mã region từ các cột one-hot region_...
    region_cols = [c for c in history.columns if c.startswith("region_")]

    def decode_region(row):
        for col in region_cols:
            if row[col] == 1:
                return col.replace("region_", "")
        return regions[0]

    history["region_display"] = history.apply(decode_region, axis=1)

    return (
        m_sales, m_margin, f_sales, f_margin,
        regions, ship_modes, categories,
        history, ship_ratio_df, ship_money_df, cat_sales_avg_df
    )


(
    m_sales, m_margin, f_sales, f_margin,
    regions, ship_modes, categories,
    history, ship_ratio_df, ship_money_df, cat_sales_avg_df
) = load_assets()

# =========================================================
# 3. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## Điều khiển mô hình")
    st.caption("Chọn khu vực và thời điểm để xem dự báo và mô phỏng chiến lược.")

    region_user = st.selectbox("Khu vực", regions)

    date_options = pd.date_range(
        start="2013-01-01",
        end="2016-12-01",
        freq="MS"
    ).strftime("%Y-%m").tolist()

    sel_date_str = st.select_slider(
        "Tháng dự báo",
        options=date_options,
        value="2016-12"
    )
    target_date = pd.to_datetime(f"{sel_date_str}-01")

    st.markdown("---")


# =========================================================
# 4. PAGE HEADER
# =========================================================
st.markdown("""
<div class="section-title">Retail Forecast & Profit Optimization</div>

""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Dự báo doanh thu", "Mô phỏng lợi nhuận"])

# =========================================================
# 5. TAB 1 - SALES FORECAST
# =========================================================
with tab1:
    df_reg = history[history["region_display"] == region_user].sort_values("ds").copy()
    last_hist_date = df_reg["ds"].max()
    plot_df = df_reg.copy()
    pred_log = None

    if target_date > last_hist_date:
        steps = (target_date.year - last_hist_date.year) * 12 + (target_date.month - last_hist_date.month)
        curr_row = df_reg.iloc[-1:].copy()

        for _ in range(steps):
            next_dt = pd.to_datetime(curr_row["ds"].values[0]) + pd.DateOffset(months=1)
            curr_row["ds"] = next_dt
            curr_row["month"] = next_dt.month
            curr_row["year"] = next_dt.year
            curr_row["month_sin"] = np.sin(2 * np.pi * next_dt.month / 12)
            curr_row["month_cos"] = np.cos(2 * np.pi * next_dt.month / 12)
            curr_row["lag_1"] = curr_row["y_log"].values[0]

            ly_dt = next_dt - pd.DateOffset(years=1)
            ly_data = plot_df[plot_df["ds"] == ly_dt]
            if not ly_data.empty:
                curr_row["lag_12"] = ly_data["y_log"].values[0]

            X_input = curr_row[f_sales].astype(float)
            pred_log = m_sales.predict(X_input)[0]
            curr_row["y_log"] = pred_log
            curr_row["Sales"] = np.expm1(pred_log)
            plot_df = pd.concat([plot_df, curr_row], ignore_index=True)

        sales_final = float(np.expm1(pred_log))
        forecast_type = "Dự báo từ mô hình"
    else:
        match_data = df_reg[df_reg["ds"] == target_date]
        sales_final = float(match_data["Sales"].values[0]) if not match_data.empty else 0.0
        forecast_type = "Giá trị từ dữ liệu lịch sử"

    st.markdown(f"""
    <div class="hero-card">
        <div class="hero-title">Doanh thu tháng {sel_date_str} | Khu vực {region_user}</div>
        <p class="hero-value">${sales_final:,.2f}</p>
        <div class="hero-sub">{forecast_type}</div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_side = st.columns([2.2, 1])

    with col_chart:
        st.markdown("### Xu hướng doanh thu theo mùa")
        st.caption("So sánh cùng tháng qua các năm để nhìn rõ chu kỳ doanh thu.")

        seasonal_df = plot_df.copy()
        seasonal_df["Năm"] = seasonal_df["ds"].dt.year.astype(str)
        seasonal_df["Tháng"] = seasonal_df["ds"].dt.strftime("%b")

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig = px.line(
            seasonal_df,
            x="Tháng",
            y="Sales",
            color="Năm",
            markers=True,
            category_orders={"Tháng": month_order},
            template="plotly_white",
            color_discrete_sequence=["#94a3b8", "#60a5fa", "#2563eb", "#0f4c81", "#10b981"]
        )

        fig.for_each_trace(
            lambda t: t.update(line=dict(width=4)) if t.name == str(target_date.year) else t.update(line=dict(width=2))
        )

        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=20, b=10),
            legend_title_text="Năm",
            xaxis_title="Tháng",
            yaxis_title="Doanh thu",
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
        st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown("### Tóm tắt nhanh")

        same_month_hist = df_reg[
            (df_reg["ds"].dt.month == target_date.month) &
            (df_reg["ds"] < target_date)
        ].copy()

        same_month_avg = same_month_hist["Sales"].mean() if not same_month_hist.empty else 0.0

        growth_vs_same_month_avg = (
            (sales_final - same_month_avg) / same_month_avg * 100
            if same_month_avg > 0 else 0.0
        )

        st.markdown(f"""
        <div class="soft-card">
            <div class="kpi-label">So với trung bình cùng tháng các năm trước</div>
            <div class="kpi-value">{growth_vs_same_month_avg:+.1f}%</div>
            <div class="small-muted">Đối chiếu với trung bình các kỳ cùng tháng trong lịch sử của khu vực {region_user}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        insight_text = (
            f"Doanh thu tháng {sel_date_str} đang cao hơn mức trung bình của các tháng cùng kỳ trước đây."
            if growth_vs_same_month_avg >= 0
            else f"Doanh thu tháng {sel_date_str} đang thấp hơn mức trung bình của các tháng cùng kỳ trước đây."
        )

        st.markdown(f"""
        <div class="insight-card">
            <div style="font-weight:700; color:#1f2937; margin-bottom:6px;">Nhận định nhanh</div>
            <div class="small-muted">{insight_text}</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# 6. TAB 2 - PROFIT SCENARIO
# =========================================================
with tab2:
    forecast_rev = sales_final if "sales_final" in locals() else 1000.0

    st.markdown("### Mô phỏng hỗ trợ tối ưu hóa biên lợi nhuận")
    st.caption("Điều chỉnh discount, shipping và cấu hình vận hành để đánh giá tác động lên margin và lợi nhuận ròng.")

    st.markdown(f"""
    <div class="note-card">
        Doanh thu dự báo dùng cho mô phỏng tại <b>{region_user}</b> trong tháng <b>{sel_date_str}</b> là <b>${forecast_rev:,.2f}</b>.
    </div>
    """, unsafe_allow_html=True)

    col_ctrl, col_viz = st.columns([1, 1.35], gap="large")

    with col_ctrl:
        st.markdown("### Thiết lập kịch bản")

        cat_user = st.selectbox("Ngành hàng mục tiêu", categories)

        m_money = ship_money_df[
            (ship_money_df["Region"] == region_user) &
            (ship_money_df["Category"] == cat_user)
        ]
        m_ratio = ship_ratio_df[
            (ship_ratio_df["Region"] == region_user) &
            (ship_ratio_df["Category"] == cat_user)
        ]
        m_sales_avg = cat_sales_avg_df[
            (cat_sales_avg_df["Region"] == region_user) &
            (cat_sales_avg_df["Category"] == cat_user)
        ]

        region_total_avg = cat_sales_avg_df[
            cat_sales_avg_df["Region"] == region_user
        ]["Avg_Monthly_Sales"].sum()

        cat_avg = float(m_sales_avg["Avg_Monthly_Sales"].values[0]) if not m_sales_avg.empty else 0.0
        share_ratio = cat_avg / region_total_avg if region_total_avg > 0 else 0.33
        sim_sales_cat = forecast_rev * share_ratio

        avg_ship_ratio = float(m_ratio["avg_ship_ratio"].values[0]) if not m_ratio.empty else 0.12
        avg_ship_money = float(m_money["Avg_Ship_Cost_Money"].values[0]) if not m_money.empty else 0.0

        st.markdown(f"""
        <div class="success-note">
            <b>Hồ sơ lịch sử</b><br>
            Phí ship trung bình: <b>${avg_ship_money:,.2f}</b><br>
            Tỷ lệ ship/doanh thu: <b>{avg_ship_ratio*100:.1f}%</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="note-card">
            <b>Doanh thu dự báo {cat_user}</b><br>
            ${sim_sales_cat:,.2f}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        discount_sim = st.slider(
            "Mức chiết khấu áp dụng (%)",
            min_value=0.0,
            max_value=0.8,
            value=0.15,
            step=0.05
        )

        ship_mode_user = st.selectbox("Hình thức vận chuyển", ship_modes)
        ship_enc = ship_modes.index(ship_mode_user)

        ship_ratio_sim = avg_ship_ratio
        qty_sim = 5

        with st.expander("Tùy chỉnh thêm"):
            ship_ratio_sim = st.number_input(
                "Tỷ lệ phí ship",
                min_value=0.0,
                max_value=1.0,
                value=float(avg_ship_ratio),
                step=0.01
            )
            qty_sim = st.number_input(
                "Số lượng sản phẩm/đơn",
                min_value=1,
                value=5,
                step=1
            )

    with col_viz:
        st.markdown("### Dự báo lợi nhuận và điểm hòa vốn")

        in_m = pd.DataFrame(0.0, index=[0], columns=f_margin)
        in_m["Discount"] = discount_sim
        in_m["ship_mode_enc"] = ship_enc
        in_m["ship_ratio"] = ship_ratio_sim
        in_m["Quantity"] = qty_sim

        if f"cat_{cat_user}" in in_m.columns:
            in_m[f"cat_{cat_user}"] = 1.0
        if f"region_{region_user}" in in_m.columns:
            in_m[f"region_{region_user}"] = 1.0

        margin_pred = float(m_margin.predict(in_m.astype(float))[0])
        net_profit = float(sim_sales_cat * margin_pred)

        k1, k2 = st.columns(2)

        badge_html = (
            '<span class="kpi-badge-good">Có lãi</span>' if net_profit > 0 and margin_pred >= 0.10 else
            '<span class="kpi-badge-warn">Biên mỏng</span>' if net_profit > 0 else
            '<span class="kpi-badge-bad">Lỗ ròng</span>'
        )

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Biên lợi nhuận dự kiến</div>
                <div class="kpi-value">{margin_pred*100:.2f}%</div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Lợi nhuận ròng dự kiến</div>
                <div class="kpi-value">${net_profit:,.2f}</div>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        d_range = np.linspace(0, 0.8, 50)
        m_range = [float(m_margin.predict(in_m.assign(Discount=d).astype(float))[0]) for d in d_range]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=d_range * 100,
            y=m_range,
            mode="lines",
            line=dict(color="#0f4c81", width=4),
            name="Biên lợi nhuận"
        ))

        fig.add_trace(go.Scatter(
            x=[discount_sim * 100],
            y=[margin_pred],
            mode="markers+text",
            marker=dict(
                color="#ef4444",
                size=13,
                symbol="diamond",
                line=dict(width=2, color="white")
            ),
            text=["Kịch bản hiện tại"],
            textposition="top center",
            name="Kịch bản chọn"
        ))

        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="Hòa vốn",
            annotation_position="top right"
        )

        fig.update_layout(
            template="plotly_white",
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Mức discount (%)",
            yaxis_title="Biên lợi nhuận dự báo",
            showlegend=False,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
        st.plotly_chart(fig, use_container_width=True)

        if margin_pred < 0:
            st.markdown("""
            <div class="warn-note">
                <b>Cảnh báo:</b> Với cấu hình hiện tại, mô hình dự báo biên lợi nhuận âm. Nên giảm discount hoặc kiểm soát chi phí ship trước khi triển khai.
            </div>
            """, unsafe_allow_html=True)
        elif margin_pred < 0.10:
            st.markdown("""
            <div class="warn-note">
                <b>Lưu ý:</b> Kịch bản vẫn có lãi nhưng biên mỏng. Doanh nghiệp sẽ nhạy với biến động của phí vận chuyển và mix sản phẩm.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-note">
                <b>Đánh giá:</b> Kịch bản hiện tại có vùng an toàn tốt hơn cho lợi nhuận ròng và phù hợp để cân nhắc triển khai.
            </div>
            """, unsafe_allow_html=True)

        breakeven_discount = None
        for d, m in zip(d_range, m_range):
            if m <= 0:
                breakeven_discount = d * 100
                break

        insight_text = (
            f"Ngưỡng discount hòa vốn xấp xỉ khoảng {breakeven_discount:.1f}%."
            if breakeven_discount is not None
            else "Trong dải khảo sát hiện tại, mô hình chưa đi xuống dưới ngưỡng hòa vốn."
        )

        st.markdown(f"""
        <div class="insight-card">
            <div style="font-weight:700; color:#1f2937; margin-bottom:6px;">Doanh nghiệp cần chú ý </div>
            <div class="small-muted">{insight_text}</div>
        </div>
        """, unsafe_allow_html=True)
