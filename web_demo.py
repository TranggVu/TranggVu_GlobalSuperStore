from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastResult:
    metrics: dict
    monthly_full: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    history_plot_df: pd.DataFrame
    feature_importance: pd.DataFrame
    regions: list[str]
    best_iteration: int


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = {
        col: col.strip().lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    }
    return df.rename(columns=normalized)


def read_dataset(uploaded_file, sheet_name: str | None) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file, sheet_name=sheet_name or 0)
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    raise ValueError("Chỉ hỗ trợ file .csv, .xlsx hoặc .xls")


def find_column(columns: list[str], candidates: list[str]) -> str:
    lookup = set(columns)
    for candidate in candidates:
        if candidate in lookup:
            return candidate
    raise KeyError(f"Không tìm thấy cột phù hợp. Cần một trong các cột: {candidates}")


def prepare_monthly_region_data(raw_df: pd.DataFrame, test_cutoff: str) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    date_col = find_column(df.columns.tolist(), ["order_date", "orderdate", "date", "ngay_dat_hang"])
    sales_col = find_column(df.columns.tolist(), ["sales", "revenue", "doanh_thu"])
    region_col = find_column(df.columns.tolist(), ["region", "khu_vuc", "vung"])

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")
    df = df.dropna(subset=[date_col, sales_col, region_col]).copy()

    sales_limit = df.loc[df[date_col] < test_cutoff, sales_col].quantile(0.95)
    df["sales_cleaned"] = df[sales_col].clip(upper=sales_limit)

    monthly = (
        df.groupby([pd.Grouper(key=date_col, freq="MS"), region_col])["sales_cleaned"]
        .sum()
        .reset_index()
        .rename(columns={date_col: "ds", region_col: "region", "sales_cleaned": "y"})
    )

    all_regions = monthly["region"].sort_values().unique()
    all_months = pd.date_range(monthly["ds"].min(), monthly["ds"].max(), freq="MS")
    full_index = pd.MultiIndex.from_product([all_months, all_regions], names=["ds", "region"])

    monthly_full = (
        monthly.set_index(["ds", "region"])
        .reindex(full_index)
        .reset_index()
    )
    monthly_full["y"] = monthly_full["y"].fillna(0)
    return monthly_full


def make_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy().sort_values(["region", "ds"])
    featured["region_name"] = featured["region"]

    featured["month"] = featured["ds"].dt.month
    featured["quarter"] = featured["ds"].dt.quarter
    featured["year"] = featured["ds"].dt.year
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12)
    featured["y_log"] = np.log1p(featured["y"])

    for lag in [1, 2, 3, 6, 12]:
        featured[f"lag_{lag}"] = featured.groupby("region")["y_log"].shift(lag)

    grouped_lag_1 = featured.groupby("region")["lag_1"]
    featured["roll_mean_3"] = grouped_lag_1.transform(lambda x: x.rolling(3).mean())
    featured["roll_mean_6"] = grouped_lag_1.transform(lambda x: x.rolling(6).mean())
    featured["roll_std_3"] = grouped_lag_1.transform(lambda x: x.rolling(3).std())
    featured["diff_1_3"] = featured["lag_1"] - featured["lag_3"]
    featured["diff_1_12"] = featured["lag_1"] - featured["lag_12"]

    featured = pd.get_dummies(featured, columns=["region"], drop_first=True)
    return featured.dropna().reset_index(drop=True)


def wape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denominator * 100)


def train_monthly_xgboost(
    raw_df: pd.DataFrame,
    valid_cutoff: str,
    test_cutoff: str,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
) -> ForecastResult:
    monthly_full = prepare_monthly_region_data(raw_df, test_cutoff=test_cutoff)
    data_model = make_monthly_features(monthly_full)

    train_df = data_model[data_model["ds"] < valid_cutoff].copy()
    valid_df = data_model[(data_model["ds"] >= valid_cutoff) & (data_model["ds"] < test_cutoff)].copy()
    test_df = data_model[data_model["ds"] >= test_cutoff].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Không đủ dữ liệu để tách train/validation/test với các mốc thời gian hiện tại.")

    cols_to_drop = ["ds", "y", "y_log", "region_name"]
    features = [col for col in train_df.columns if col not in cols_to_drop]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1,
        reg_lambda=8,
        random_state=42,
        early_stopping_rounds=60,
        n_jobs=-1,
    )

    model.fit(
        train_df[features],
        train_df["y_log"],
        eval_set=[(valid_df[features], valid_df["y_log"])],
        verbose=False,
    )

    y_pred = np.expm1(model.predict(test_df[features]))
    metrics = {
        "R2": float(r2_score(test_df["y"], y_pred)),
        "MAE": float(mean_absolute_error(test_df["y"], y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test_df["y"], y_pred))),
        "WAPE": wape(test_df["y"], y_pred),
    }

    history_plot_df = test_df[["ds", "y", "region_name"]].copy()
    history_plot_df["predicted"] = y_pred
    history_plot_df["segment"] = "Test"

    feature_importance = pd.DataFrame(
        {
            "feature": features,
            "gain": model.feature_importances_,
        }
    ).sort_values("gain", ascending=False)

    region_columns = [col for col in monthly_full.columns if col == "region"]
    regions = sorted(monthly_full[region_columns[0]].unique().tolist()) if region_columns else []

    return ForecastResult(
        metrics=metrics,
        monthly_full=monthly_full,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df.assign(predicted=y_pred).rename(columns={"region_name": "region"}),
        history_plot_df=history_plot_df,
        feature_importance=feature_importance,
        regions=regions,
        best_iteration=int(model.best_iteration),
    )


def build_overall_chart(test_df: pd.DataFrame) -> go.Figure:
    monthly_actual = test_df.groupby("ds", as_index=False)["y"].sum()
    monthly_pred = test_df.groupby("ds", as_index=False)["predicted"].sum()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=monthly_actual["ds"],
            y=monthly_actual["y"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#0f766e", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_pred["ds"],
            y=monthly_pred["predicted"],
            mode="lines+markers",
            name="Forecast",
            line=dict(color="#f97316", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Month",
        yaxis_title="Revenue",
        height=420,
    )
    return fig


def build_region_chart(test_df: pd.DataFrame, monthly_full: pd.DataFrame, region: str) -> go.Figure:
    region_actual = monthly_full[monthly_full["region"] == region][["ds", "y"]].copy()
    region_test = test_df[test_df["region"] == region][["ds", "y", "predicted"]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=region_actual["ds"],
            y=region_actual["y"],
            mode="lines",
            name="History",
            line=dict(color="#94a3b8", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=region_test["ds"],
            y=region_test["y"],
            mode="lines+markers",
            name="Actual test",
            line=dict(color="#1d4ed8", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=region_test["ds"],
            y=region_test["predicted"],
            mode="lines+markers",
            name="Forecast test",
            line=dict(color="#ea580c", width=3, dash="dash"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Month",
        yaxis_title="Revenue",
        height=420,
    )
    return fig


def style_app() -> None:
    st.set_page_config(page_title="Revenue Forecast Studio", page_icon="📈", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(15,118,110,0.14), transparent 32%),
                    radial-gradient(circle at top right, rgba(249,115,22,0.16), transparent 28%),
                    linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            }
            .hero {
                padding: 28px 32px;
                border-radius: 24px;
                background: linear-gradient(135deg, #0f172a 0%, #134e4a 52%, #f97316 100%);
                color: white;
                box-shadow: 0 20px 40px rgba(15, 23, 42, 0.18);
                margin-bottom: 22px;
            }
            .hero h1 {
                margin: 0 0 8px 0;
                font-size: 2.4rem;
                letter-spacing: -0.04em;
            }
            .hero p {
                margin: 0;
                max-width: 760px;
                line-height: 1.55;
                color: rgba(255,255,255,0.9);
            }
            .metric-card {
                border-radius: 22px;
                padding: 18px 20px;
                background: rgba(255, 255, 255, 0.82);
                box-shadow: 0 18px 30px rgba(15, 23, 42, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.18);
                backdrop-filter: blur(10px);
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 700;
                margin: 6px 0 12px 0;
                color: #0f172a;
                letter-spacing: -0.02em;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 999px;
                background: rgba(255,255,255,0.7);
                border: 1px solid rgba(148, 163, 184, 0.18);
                padding: 8px 18px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str, hint: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size:0.85rem;color:#475569;margin-bottom:6px;">{label}</div>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a;line-height:1;">{value}</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:8px;">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    style_app()

    st.markdown(
        """
        <div class="hero">
            <h1>Revenue Forecast Studio</h1>
            <p>
                Demo dự báo doanh thu theo tháng và vùng bằng XGBoost. Ứng dụng cho phép tải dữ liệu,
                huấn luyện mô hình trực tiếp, xem chỉ số đánh giá, biểu đồ dự báo và các đặc trưng quan trọng nhất.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Cấu hình")
        uploaded_file = st.file_uploader("Tải dữ liệu bán hàng", type=["csv", "xlsx", "xls"])
        sheet_name = st.text_input("Tên sheet Excel", value="Orders")
        valid_cutoff = st.date_input("Mốc validation", value=pd.Timestamp("2014-07-01"))
        test_cutoff = st.date_input("Mốc test", value=pd.Timestamp("2015-01-01"))
        learning_rate = st.slider("Learning rate", min_value=0.01, max_value=0.10, value=0.03, step=0.01)
        max_depth = st.slider("Max depth", min_value=2, max_value=6, value=3)
        n_estimators = st.slider("Số cây tối đa", min_value=300, max_value=2000, value=1200, step=100)
        run_button = st.button("Huấn luyện và dự báo", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption(
            "Yêu cầu dữ liệu có ít nhất cột ngày, doanh thu và vùng. "
            "App tự dò các tên cột phổ biến như Order Date, Sales, Region."
        )

    if not uploaded_file:
        st.info("Tải file `.csv` hoặc `.xlsx` để bắt đầu demo.")
        return

    if not run_button:
        st.warning("Thiết lập tham số ở thanh bên trái rồi bấm `Huấn luyện và dự báo`.")
        return

    try:
        with st.spinner("Đang đọc dữ liệu, huấn luyện mô hình và dựng dashboard..."):
            raw_df = read_dataset(uploaded_file, sheet_name.strip() or None)
            result = train_monthly_xgboost(
                raw_df=raw_df,
                valid_cutoff=str(valid_cutoff),
                test_cutoff=str(test_cutoff),
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators,
            )
    except Exception as exc:
        st.error(f"Không thể chạy mô hình: {exc}")
        return

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric("R²", f"{result.metrics['R2']:.3f}", "Khả năng giải thích biến thiên")
    with metric_cols[1]:
        render_metric("WAPE", f"{result.metrics['WAPE']:.2f}%", "Sai số tương đối trên tổng doanh thu")
    with metric_cols[2]:
        render_metric("MAE", f"{result.metrics['MAE']:,.0f}", "Sai số tuyệt đối trung bình")
    with metric_cols[3]:
        render_metric("RMSE", f"{result.metrics['RMSE']:,.0f}", "Phạt mạnh hơn với sai số lớn")
    with metric_cols[4]:
        render_metric("Best Iteration", str(result.best_iteration), "Điểm dừng sớm tốt nhất")

    tabs = st.tabs(["Tổng quan", "Theo vùng", "Đặc trưng", "Dữ liệu test"])

    with tabs[0]:
        st.markdown('<div class="section-title">Tổng doanh thu thực tế vs dự báo trên tập test</div>', unsafe_allow_html=True)
        st.plotly_chart(build_overall_chart(result.test_df), use_container_width=True)

        summary_cols = st.columns([1.1, 1])
        with summary_cols[0]:
            actual_summary = result.test_df.groupby("ds", as_index=False)["y"].sum().rename(columns={"y": "actual"})
            forecast_summary = result.test_df.groupby("ds", as_index=False)["predicted"].sum().rename(columns={"predicted": "forecast"})
            compare_df = actual_summary.merge(forecast_summary, on="ds")
            compare_df["abs_error"] = (compare_df["actual"] - compare_df["forecast"]).abs()
            st.dataframe(compare_df, use_container_width=True, hide_index=True)
        with summary_cols[1]:
            top_region_error = (
                result.test_df.assign(abs_error=lambda df: (df["y"] - df["predicted"]).abs())
                .groupby("region", as_index=False)["abs_error"]
                .sum()
                .sort_values("abs_error", ascending=False)
            )
            fig = px.bar(
                top_region_error.head(8),
                x="abs_error",
                y="region",
                orientation="h",
                color="abs_error",
                color_continuous_scale=["#cbd5e1", "#0f766e", "#f97316"],
            )
            fig.update_layout(
                template="plotly_white",
                margin=dict(l=20, r=20, t=20, b=20),
                coloraxis_showscale=False,
                yaxis_title="",
                xaxis_title="Tổng sai số tuyệt đối",
                height=360,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="section-title">Diễn biến theo từng vùng</div>', unsafe_allow_html=True)
        selected_region = st.selectbox("Chọn vùng để xem chi tiết", result.regions)
        st.plotly_chart(
            build_region_chart(result.test_df, result.monthly_full, selected_region),
            use_container_width=True,
        )
        region_table = result.test_df[result.test_df["region"] == selected_region][["ds", "y", "predicted"]].copy()
        region_table["abs_error"] = (region_table["y"] - region_table["predicted"]).abs()
        st.dataframe(region_table, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown('<div class="section-title">Top feature importance</div>', unsafe_allow_html=True)
        fig = px.bar(
            result.feature_importance.head(15).sort_values("gain"),
            x="gain",
            y="feature",
            orientation="h",
            color="gain",
            color_continuous_scale=["#dbeafe", "#2563eb", "#f97316"],
        )
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            yaxis_title="",
            xaxis_title="Importance",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Biểu đồ này cho biết biến nào được mô hình dùng nhiều nhất để giảm lỗi dự báo, "
            "không nên diễn giải như quan hệ nhân quả tuyệt đối."
        )

    with tabs[3]:
        st.markdown('<div class="section-title">Tập test chi tiết</div>', unsafe_allow_html=True)
        detail_df = result.test_df[["ds", "region", "y", "predicted"]].copy()
        detail_df["abs_error"] = (detail_df["y"] - detail_df["predicted"]).abs()
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
