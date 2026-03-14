import re
from itertools import combinations

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None


st.set_page_config(page_title="SKU 属性销售分析", layout="wide")

st.title("SKU 属性销售分析仪表盘")
st.caption(
    "上传两个销售文件和一个 SKU 属性对照表后，自动完成 SKU 映射、动态时间区间识别、单属性分析、两两属性 Heat Map、退款风险分析、款均分析与中文摘要。"
)


def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return x if x != "" else np.nan


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def normalize_base_sku(sku: str):
    if pd.isna(sku):
        return np.nan
    sku = str(sku).strip().upper()
    sku = sku.split("-")[0]

    m = re.search(r"N[A-Z]{1,2}[FXJS]\d{3}", sku)
    if m:
        return m.group(0)

    m = re.search(r"N[A-Z0-9]{2,4}\d{3}", sku)
    if m:
        return m.group(0)

    return sku


def split_multi_value(x):
    if pd.isna(x):
        return []
    text = str(x).strip()
    if text == "":
        return []

    parts = re.split(r"[;,，、/|]+", text)
    cleaned = []
    for p in parts:
        item = str(p).strip()
        if item and item.lower() not in {"nan", "none", "null"}:
            cleaned.append(item)

    seen = set()
    deduped = []
    for item in cleaned:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def format_pct(x):
    if pd.isna(x):
        return "无法计算"
    return f"{x * 100:.1f}%"


def detect_date_column(df):
    candidates = [
        "Order Created Time",
        "Order Create Time",
        "Created Time",
        "Create Time",
        "Order Date",
        "Created At",
        "order_created_time",
        "create_time",
        "date",
    ]

    for col in candidates:
        if col in df.columns:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() > 0:
                return col

    for col in df.columns:
        col_name = str(col).lower()
        if "date" in col_name or "time" in col_name:
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().sum() > 0:
                return col

    return None


def build_period_label(dt_series, fallback_label):
    dt_series = pd.to_datetime(dt_series, errors="coerce").dropna()
    if len(dt_series) == 0:
        return fallback_label, None, None

    start_dt = dt_series.min().normalize()
    end_dt = dt_series.max().normalize()
    label = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}"
    return label, start_dt, end_dt


def load_sales(file, fallback_label: str):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Seller SKU", "Quantity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{fallback_label} 缺少必要列: {missing}")

    df["Seller SKU"] = df["Seller SKU"].astype(str).str.strip()
    df["Base SKU"] = df["Seller SKU"].apply(normalize_base_sku)
    df["Quantity"] = safe_numeric(df["Quantity"])

    if "SKU Subtotal After Discount" in df.columns:
        df["Sales"] = safe_numeric(df["SKU Subtotal After Discount"])
    elif "Order Amount" in df.columns:
        df["Sales"] = safe_numeric(df["Order Amount"])
    else:
        df["Sales"] = 0

    if "Order Refund Amount" in df.columns:
        df["Refund Amount"] = safe_numeric(df["Order Refund Amount"])
    else:
        df["Refund Amount"] = 0

    if "Order Status" not in df.columns:
        df["Order Status"] = "Unknown"

    df["Has Refund"] = np.where(df["Refund Amount"] > 0, 1, 0)

    date_col = detect_date_column(df)
    if date_col is not None:
        df["Order Date Parsed"] = pd.to_datetime(df[date_col], errors="coerce")
        period_label, period_start, period_end = build_period_label(df["Order Date Parsed"], fallback_label)
    else:
        df["Order Date Parsed"] = pd.NaT
        period_label, period_start, period_end = fallback_label, None, None

    df["Period"] = period_label
    df.attrs["period_label"] = period_label
    df.attrs["period_start"] = period_start
    df.attrs["period_end"] = period_end
    df.attrs["date_col"] = date_col

    return df


@st.cache_data
def load_attr(file):
    attr = pd.read_csv(file)
    attr.columns = [str(c).strip() for c in attr.columns]

    if "SKU" not in attr.columns:
        raise ValueError("属性对照表缺少必要列: SKU")

    for col in attr.columns:
        attr[col] = attr[col].apply(clean_text)

    attr["SKU"] = attr["SKU"].astype(str).str.strip().str.upper()

    if "上架时间" in attr.columns:
        attr["上架时间"] = pd.to_datetime(attr["上架时间"], errors="coerce")

        attr["上架月份"] = attr["上架时间"].dt.strftime("%Y-%m")
        attr["上架月份"] = attr["上架月份"].fillna("(空值)")

        def format_quarter(dt):
            if pd.isna(dt):
                return "(空值)"
            return f"{dt.year}Q{dt.quarter}"

        attr["上架季度"] = attr["上架时间"].apply(format_quarter)

        today = pd.Timestamp.today().normalize()
        attr["上架天数"] = (today - attr["上架时间"]).dt.days

        def classify_launch_stage(days):
            if pd.isna(days):
                return "(空值)"
            if days <= 30:
                return "新品"
            elif days <= 90:
                return "次新品"
            elif days <= 180:
                return "常规款"
            else:
                return "老款"

        attr["上新阶段"] = attr["上架天数"].apply(classify_launch_stage)
    else:
        attr["上架月份"] = "(空值)"
        attr["上架季度"] = "(空值)"
        attr["上新阶段"] = "(空值)"
        attr["上架天数"] = np.nan

    attr = attr.drop_duplicates(subset=["SKU"], keep="first").copy()
    return attr


def merge_sales_attr(sales_df, attr_df):
    merged = sales_df.merge(attr_df, how="left", left_on="Base SKU", right_on="SKU")
    merged["Mapped"] = merged["SKU"].notna()
    return merged


def choose_attribute_columns(df):
    excluded = {
        "SKU",
        "中文名称",
        "款式英文名称",
        "图片",
        "上架时间",
        "上架天数",
        "Seller SKU",
        "Base SKU",
        "Period",
        "Mapped"
    }

    candidates = []
    for col in df.columns:
        if col in excluded:
            continue
        if col in ["Quantity", "Sales", "Refund Amount", "Has Refund"]:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        non_null = df[col].notna().sum()
        nunique = df[col].nunique(dropna=True)
        if non_null > 0 and nunique >= 2:
            candidates.append(col)

    return candidates


def apply_filters(df, selected_status, mapped_only):
    out = df.copy()
    if selected_status:
        out = out[out["Order Status"].isin(selected_status)]
    if mapped_only:
        out = out[out["Mapped"]]
    return out


def explode_attribute(df, attr_col):
    tmp = df.copy()
    tmp["_attr_item"] = tmp[attr_col].apply(split_multi_value)
    tmp["_attr_item"] = tmp["_attr_item"].apply(lambda x: x if len(x) > 0 else ["(空值)"])
    tmp = tmp.explode("_attr_item")
    tmp["_attr_item"] = tmp["_attr_item"].fillna("(空值)")
    return tmp


def aggregate_by_attribute(df, attr_col):
    tmp = explode_attribute(df, attr_col)

    g = tmp.groupby("_attr_item", dropna=False).agg(
        order_lines=("Seller SKU", "count"),
        units=("Quantity", "sum"),
        sales=("Sales", "sum"),
        refund=("Refund Amount", "sum"),
        refund_orders=("Has Refund", "sum"),
        sku_count=("Base SKU", "nunique")
    ).reset_index().rename(columns={"_attr_item": attr_col})

    g["refund_rate_by_orders"] = np.where(g["order_lines"] == 0, np.nan, g["refund_orders"] / g["order_lines"])
    g["refund_to_sales_ratio"] = np.where(g["sales"] == 0, np.nan, g["refund"] / g["sales"])

    g["sales_per_sku"] = np.where(g["sku_count"] == 0, np.nan, g["sales"] / g["sku_count"])
    g["units_per_sku"] = np.where(g["sku_count"] == 0, np.nan, g["units"] / g["sku_count"])
    g["refund_per_sku"] = np.where(g["sku_count"] == 0, np.nan, g["refund"] / g["sku_count"])
    g["order_lines_per_sku"] = np.where(g["sku_count"] == 0, np.nan, g["order_lines"] / g["sku_count"])

    return g.sort_values("sales", ascending=False)


def compare_periods(cur_df, prev_df, attr_col):
    cur = aggregate_by_attribute(cur_df, attr_col).rename(columns={
        "order_lines": "order_lines_cur",
        "units": "units_cur",
        "sales": "sales_cur",
        "refund": "refund_cur",
        "refund_orders": "refund_orders_cur",
        "refund_rate_by_orders": "refund_rate_by_orders_cur",
        "refund_to_sales_ratio": "refund_to_sales_ratio_cur",
        "sales_per_sku": "sales_per_sku_cur",
        "units_per_sku": "units_per_sku_cur",
        "refund_per_sku": "refund_per_sku_cur",
        "order_lines_per_sku": "order_lines_per_sku_cur",
        "sku_count": "sku_count_cur",
    })

    prev = aggregate_by_attribute(prev_df, attr_col).rename(columns={
        "order_lines": "order_lines_prev",
        "units": "units_prev",
        "sales": "sales_prev",
        "refund": "refund_prev",
        "refund_orders": "refund_orders_prev",
        "refund_rate_by_orders": "refund_rate_by_orders_prev",
        "refund_to_sales_ratio": "refund_to_sales_ratio_prev",
        "sales_per_sku": "sales_per_sku_prev",
        "units_per_sku": "units_per_sku_prev",
        "refund_per_sku": "refund_per_sku_prev",
        "order_lines_per_sku": "order_lines_per_sku_prev",
        "sku_count": "sku_count_prev",
    })

    merged = cur.merge(prev, on=attr_col, how="outer").fillna(0)

    metrics = [
        "order_lines", "units", "sales", "refund", "refund_orders",
        "refund_rate_by_orders", "refund_to_sales_ratio",
        "sales_per_sku", "units_per_sku", "refund_per_sku", "order_lines_per_sku",
        "sku_count"
    ]

    for metric in metrics:
        merged[f"{metric}_diff"] = merged[f"{metric}_cur"] - merged[f"{metric}_prev"]
        merged[f"{metric}_pct"] = np.where(
            merged[f"{metric}_prev"] == 0,
            np.nan,
            merged[f"{metric}_diff"] / merged[f"{metric}_prev"]
        )

    return merged.sort_values("sales_cur", ascending=False)


def build_heatmap_source(df, attr_x, attr_y, metric):
    tmp = df.copy()

    tmp["_x_items"] = tmp[attr_x].apply(split_multi_value)
    tmp["_y_items"] = tmp[attr_y].apply(split_multi_value)

    tmp["_x_items"] = tmp["_x_items"].apply(lambda x: x if len(x) > 0 else ["(空值)"])
    tmp["_y_items"] = tmp["_y_items"].apply(lambda x: x if len(x) > 0 else ["(空值)"])

    tmp = tmp.explode("_x_items")
    tmp = tmp.explode("_y_items")

    tmp["_x_items"] = tmp["_x_items"].fillna("(空值)")
    tmp["_y_items"] = tmp["_y_items"].fillna("(空值)")

    grp = tmp.groupby(["_y_items", "_x_items"], dropna=False).agg(
        sales=("Sales", "sum"),
        units=("Quantity", "sum"),
        order_lines=("Seller SKU", "count"),
        refund=("Refund Amount", "sum"),
        refund_orders=("Has Refund", "sum"),
        sku_count=("Base SKU", "nunique")
    ).reset_index()

    grp["refund_rate"] = np.where(grp["order_lines"] == 0, np.nan, grp["refund_orders"] / grp["order_lines"])
    grp["refund_to_sales_ratio"] = np.where(grp["sales"] == 0, np.nan, grp["refund"] / grp["sales"])

    grp["sales_per_sku"] = np.where(grp["sku_count"] == 0, np.nan, grp["sales"] / grp["sku_count"])
    grp["units_per_sku"] = np.where(grp["sku_count"] == 0, np.nan, grp["units"] / grp["sku_count"])
    grp["refund_per_sku"] = np.where(grp["sku_count"] == 0, np.nan, grp["refund"] / grp["sku_count"])
    grp["order_lines_per_sku"] = np.where(grp["sku_count"] == 0, np.nan, grp["order_lines"] / grp["sku_count"])

    metric_map = {
        "sales": "sales",
        "sales_per_sku": "sales_per_sku",
        "units": "units",
        "units_per_sku": "units_per_sku",
        "order_lines": "order_lines",
        "order_lines_per_sku": "order_lines_per_sku",
        "refund": "refund",
        "refund_per_sku": "refund_per_sku",
        "refund_orders": "refund_orders",
        "refund_rate": "refund_rate",
        "refund_to_sales_ratio": "refund_to_sales_ratio",
    }

    value_col = metric_map.get(metric, "sales")
    src = grp[["_y_items", "_x_items", value_col]].copy()
    src = src.rename(columns={
        "_x_items": attr_x,
        "_y_items": attr_y,
        value_col: "value"
    })
    return src


def keep_top_labels(src_df, attr_x, attr_y, value_col="value", top_n=12):
    if len(src_df) == 0:
        return src_df.copy()

    temp = src_df.copy()
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce").fillna(0)

    top_x = (
        temp.groupby(attr_x)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    top_y = (
        temp.groupby(attr_y)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    out = temp[temp[attr_x].isin(top_x) & temp[attr_y].isin(top_y)].copy()
    return out


def sort_matrix(matrix):
    row_order = matrix.fillna(0).sum(axis=1).sort_values(ascending=False).index
    col_order = matrix.fillna(0).sum(axis=0).sort_values(ascending=False).index
    return matrix.loc[row_order, col_order]


def render_bar_chart(plot_df, x_col, title, y_label):
    if px is not None:
        fig = px.bar(
            plot_df,
            x=x_col,
            y="value",
            color="period",
            barmode="group",
            title=title,
            text_auto=".2s",
        )
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_label,
            xaxis_tickangle=-35,
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        pivot_bar = plot_df.pivot(index=x_col, columns="period", values="value").fillna(0)
        st.bar_chart(pivot_bar)
        st.caption("当前环境未安装 plotly，已自动切换为 Streamlit 原生柱状图。")


def render_heatmap(src_df, attr_x, attr_y, value_col, title, diff=False, percent=False):
    if len(src_df) == 0:
        st.info("该组合没有数据。")
        return

    matrix = src_df.pivot(index=attr_y, columns=attr_x, values=value_col)
    matrix = sort_matrix(matrix)

    st.markdown(f"**{title}**")

    def format_cell(v):
        if pd.isna(v):
            return ""
        if percent:
            return "0%" if abs(float(v)) < 1e-12 else f"{float(v):.1%}"
        return "0" if abs(float(v)) < 1e-12 else f"{float(v):,.2f}"

    text_matrix = matrix.copy().astype(object)
    for r in text_matrix.index:
        for c in text_matrix.columns:
            text_matrix.loc[r, c] = format_cell(text_matrix.loc[r, c])

    if px is not None:
        plot_matrix = matrix.copy()

        if diff:
            vals = plot_matrix.to_numpy(dtype=float)
            if np.isfinite(vals).any():
                max_abs = float(np.nanmax(np.abs(vals)))
            else:
                max_abs = 1.0
            if max_abs == 0 or np.isnan(max_abs):
                max_abs = 1.0

            fig = px.imshow(
                plot_matrix,
                aspect="auto",
                color_continuous_scale="RdYlGn",
                range_color=[-max_abs, max_abs],
            )
        else:
            fig = px.imshow(
                plot_matrix,
                aspect="auto",
                color_continuous_scale="YlOrRd",
            )

        fig.update_traces(
            text=text_matrix.values,
            texttemplate="%{text}",
            textfont_size=12,
            hovertemplate=f"{attr_y}: %{{y}}<br>{attr_x}: %{{x}}<br>值: %{{z}}<extra></extra>"
        )

        fig.update_layout(
            title=title,
            xaxis_title=attr_x,
            yaxis_title=attr_y,
            xaxis_tickangle=-40,
            height=max(420, 42 * len(matrix.index)),
            margin=dict(l=20, r=20, t=60, b=20),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        def style_formatter(v):
            if pd.isna(v):
                return ""
            if percent:
                return "0%" if abs(float(v)) < 1e-12 else f"{float(v):.1%}"
            return "0" if abs(float(v)) < 1e-12 else f"{float(v):,.2f}"

        styled = matrix.style.format(style_formatter)

        if diff:
            styled = styled.background_gradient(cmap="RdYlGn", axis=None)
        else:
            styled = styled.background_gradient(cmap="YlOrRd", axis=None)

        st.dataframe(styled, use_container_width=True)
        st.caption("当前环境未安装 plotly，已自动切换为表格热力图。")


def show_kpi_row(cur_df, prev_df, cur_label, prev_label):
    cur_sales = cur_df["Sales"].sum()
    prev_sales = prev_df["Sales"].sum()

    cur_units = cur_df["Quantity"].sum()
    prev_units = prev_df["Quantity"].sum()

    cur_lines = len(cur_df)
    prev_lines = len(prev_df)

    cur_map = cur_df["Mapped"].mean() if len(cur_df) else 0
    prev_map = prev_df["Mapped"].mean() if len(prev_df) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"{cur_label} 销售额",
        f"${cur_sales:,.2f}",
        f"${cur_sales - prev_sales:,.2f}",
        delta_color="normal",
    )
    c2.metric(
        f"{cur_label} 销量",
        f"{cur_units:,.0f}",
        f"{cur_units - prev_units:,.0f}",
        delta_color="normal",
    )
    c3.metric(
        f"{cur_label} 订单行",
        f"{cur_lines:,}",
        f"{cur_lines - prev_lines:,}",
        delta_color="normal",
    )
    c4.metric(
        "SKU映射率",
        f"{cur_map:.1%}",
        f"{(cur_map - prev_map):.1%}",
        delta_color="normal",
    )


def generate_overall_summary(cur_df, prev_df, cur_label, prev_label):
    cur_sales = cur_df["Sales"].sum()
    prev_sales = prev_df["Sales"].sum()
    cur_units = cur_df["Quantity"].sum()
    prev_units = prev_df["Quantity"].sum()
    cur_lines = len(cur_df)
    prev_lines = len(prev_df)

    cur_refund = cur_df["Refund Amount"].sum()
    prev_refund = prev_df["Refund Amount"].sum()

    cur_refund_orders = cur_df["Has Refund"].sum()
    prev_refund_orders = prev_df["Has Refund"].sum()

    sales_diff = cur_sales - prev_sales
    units_diff = cur_units - prev_units
    lines_diff = cur_lines - prev_lines
    refund_diff = cur_refund - prev_refund
    refund_orders_diff = cur_refund_orders - prev_refund_orders

    sales_pct = sales_diff / prev_sales if prev_sales != 0 else np.nan
    units_pct = units_diff / prev_units if prev_units != 0 else np.nan
    lines_pct = lines_diff / prev_lines if prev_lines != 0 else np.nan
    refund_pct = refund_diff / prev_refund if prev_refund != 0 else np.nan
    refund_orders_pct = refund_orders_diff / prev_refund_orders if prev_refund_orders != 0 else np.nan

    cur_map = cur_df["Mapped"].mean() if len(cur_df) else 0

    texts = [
        f"{cur_label} 整体销售额为 ${cur_sales:,.2f}，相较 {prev_label} 变化了 ${sales_diff:,.2f}，变化幅度为 {format_pct(sales_pct)}。",
        f"{cur_label} 总销量为 {cur_units:,.0f} 件，相较 {prev_label} 变化了 {units_diff:,.0f} 件，变化幅度为 {format_pct(units_pct)}。",
        f"{cur_label} 订单行数为 {cur_lines:,}，相较 {prev_label} 变化了 {lines_diff:,}，变化幅度为 {format_pct(lines_pct)}。",
        f"{cur_label} 退款金额为 ${cur_refund:,.2f}，相较 {prev_label} 变化了 ${refund_diff:,.2f}，变化幅度为 {format_pct(refund_pct)}。",
        f"{cur_label} 发生退款的订单行数为 {cur_refund_orders:,.0f}，相较 {prev_label} 变化了 {refund_orders_diff:,.0f}，变化幅度为 {format_pct(refund_orders_pct)}。",
        f"当前 SKU 属性映射率为 {cur_map:.1%}。"
    ]

    if sales_diff > 0 and units_diff > 0:
        texts.append("这说明当前区间整体销售表现偏强，销售额和销量同步改善。")
    elif sales_diff > 0 and units_diff <= 0:
        texts.append("这说明当前区间销售额提升更多可能来自高客单价款式，而不是纯粹靠销量扩大。")
    elif sales_diff < 0 and units_diff < 0:
        texts.append("这说明当前区间整体成交表现走弱，建议进一步拆解属性结构和主力 SKU 的变化。")
    else:
        texts.append("当前区间销售额与销量的变化方向并不完全一致，建议结合属性和退款表现做进一步分析。")

    if refund_diff > 0:
        texts.append("退款金额上升意味着当前区间售后压力有所变大，建议重点观察哪些属性和组合更容易产生退款。")
    elif refund_diff < 0:
        texts.append("退款金额下降说明当前区间售后表现有所改善，可以继续观察改善是否来自某些属性组合结构优化。")

    return texts


def generate_attribute_summary(comp_df, attr_col, top_n=5):
    df = comp_df.copy()
    if len(df) == 0:
        return [f"{attr_col} 维度当前没有可分析的数据。"]

    top_sales = df.sort_values("sales_cur", ascending=False).head(top_n)
    top_sales_per_sku = df.sort_values("sales_per_sku_cur", ascending=False).head(top_n)
    top_growth = df.sort_values("sales_diff", ascending=False).head(top_n)

    texts = []

    if len(top_sales) > 0:
        names = "、".join(top_sales[attr_col].astype(str).tolist())
        texts.append(f"从总销售额来看，当前区间贡献最高的 {attr_col} 属性词主要包括：{names}。")

    if len(top_sales_per_sku) > 0:
        names = "、".join(top_sales_per_sku[attr_col].astype(str).tolist())
        texts.append(f"从款均销售额来看，单款效率最高的 {attr_col} 属性词主要包括：{names}。这组指标更适合判断哪些属性是真正“单款能打”。")

    if len(top_growth) > 0:
        names = "、".join(top_growth[attr_col].astype(str).tolist())
        texts.append(f"从区间增长角度看，提升最明显的属性词包括：{names}。")

    leader = top_sales.iloc[0] if len(top_sales) > 0 else None
    if leader is not None:
        texts.append(
            f"其中，{leader[attr_col]} 当前区间总销售额为 ${leader['sales_cur']:,.2f}，款数为 {leader['sku_count_cur']:,.0f}，款均销售额为 ${leader['sales_per_sku_cur']:,.2f}。"
        )

    return texts


def generate_refund_summary(comp_df, attr_col, top_n=5):
    df = comp_df.copy()
    if len(df) == 0:
        return [f"{attr_col} 维度当前没有足够退款数据。"]

    high_refund_amt = df.sort_values("refund_cur", ascending=False).head(top_n)
    high_refund_rate = df.sort_values("refund_rate_by_orders_cur", ascending=False).head(top_n)
    high_refund_ratio = df.sort_values("refund_to_sales_ratio_cur", ascending=False).head(top_n)
    high_refund_per_sku = df.sort_values("refund_per_sku_cur", ascending=False).head(top_n)

    texts = []

    if len(high_refund_amt) > 0:
        names = "、".join(high_refund_amt[attr_col].astype(str).tolist())
        texts.append(f"从退款金额来看，当前区间退款金额较高的属性词主要包括：{names}。")

    if len(high_refund_rate) > 0:
        names = "、".join(high_refund_rate[attr_col].astype(str).tolist())
        texts.append(f"从退款订单占比来看，更容易发生退款的属性词包括：{names}。")

    if len(high_refund_ratio) > 0:
        names = "、".join(high_refund_ratio[attr_col].astype(str).tolist())
        texts.append(f"从退款金额占销售额比例来看，风险更高的属性词包括：{names}。")

    if len(high_refund_per_sku) > 0:
        names = "、".join(high_refund_per_sku[attr_col].astype(str).tolist())
        texts.append(f"从款均退款金额来看，单款售后压力更高的属性词包括：{names}。")

    return texts


def generate_heatmap_summary(cur_src, prev_src, attr_x, attr_y, metric, top_n=5):
    if len(cur_src) == 0:
        return [f"{attr_x} × {attr_y} 当前没有可分析的数据。"]

    cur_top = cur_src.sort_values("value", ascending=False).head(top_n).copy()

    diff_src = cur_src.merge(
        prev_src,
        on=[attr_x, attr_y],
        how="outer",
        suffixes=("_cur", "_prev"),
    ).fillna(0)
    diff_src["diff"] = diff_src["value_cur"] - diff_src["value_prev"]

    growth_top = diff_src.sort_values("diff", ascending=False).head(top_n).copy()

    texts = []

    if len(cur_top) > 0:
        combos = [f"{r[attr_x]} × {r[attr_y]}" for _, r in cur_top.iterrows()]
        texts.append(f"从 {attr_x} × {attr_y} 的组合关系来看，当前区间在指标 {metric} 下表现最强的组合主要有：{'、'.join(combos)}。")

    if len(growth_top) > 0:
        combos = [f"{r[attr_x]} × {r[attr_y]}" for _, r in growth_top.iterrows()]
        texts.append(f"从区间变化来看，提升最明显的组合包括：{'、'.join(combos)}。")

    texts.append("组合分析更适合观察属性之间的协同效果，而不是只看单一属性本身。")
    return texts


def generate_business_suggestions(comp_df, attr_col, top_n=3):
    if len(comp_df) == 0:
        return [f"{attr_col} 当前没有足够数据，暂时无法形成业务建议。"]

    high_sales = comp_df.sort_values("sales_cur", ascending=False).head(top_n)
    high_efficiency = comp_df.sort_values("sales_per_sku_cur", ascending=False).head(top_n)
    high_refund = comp_df.sort_values("refund_to_sales_ratio_cur", ascending=False).head(top_n)

    texts = []

    if len(high_sales) > 0:
        names = "、".join(high_sales[attr_col].astype(str).tolist())
        texts.append(f"建议优先围绕 {names} 这类总销售贡献高的属性做重点补货和内容投放。")

    if len(high_efficiency) > 0:
        names = "、".join(high_efficiency[attr_col].astype(str).tolist())
        texts.append(f"建议重点观察 {names} 这类款均效率高的属性，因为它们更能代表真正的单款竞争力。")

    if len(high_refund) > 0:
        names = "、".join(high_refund[attr_col].astype(str).tolist())
        texts.append(f"对于 {names} 这类退款风险相对更高的属性，建议优先复盘产品质量、展示素材、尺码预期和包装交付问题。")

    return texts


def prepare_display_table(comp_df):
    out = comp_df.copy()

    pct_cols = [
        "sales_pct", "units_pct", "order_lines_pct", "refund_pct", "refund_orders_pct",
        "refund_rate_by_orders_cur", "refund_rate_by_orders_prev", "refund_rate_by_orders_diff", "refund_rate_by_orders_pct",
        "refund_to_sales_ratio_cur", "refund_to_sales_ratio_prev", "refund_to_sales_ratio_diff", "refund_to_sales_ratio_pct",
        "sales_per_sku_pct", "units_per_sku_pct", "refund_per_sku_pct", "order_lines_per_sku_pct",
        "sku_count_pct"
    ]

    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "N/A")

    money_cols = [
        "sales_cur", "sales_prev", "sales_diff",
        "refund_cur", "refund_prev", "refund_diff",
        "sales_per_sku_cur", "sales_per_sku_prev", "sales_per_sku_diff",
        "refund_per_sku_cur", "refund_per_sku_prev", "refund_per_sku_diff",
    ]
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"${x:,.2f}")

    num_cols = [
        "units_cur", "units_prev", "units_diff",
        "order_lines_cur", "order_lines_prev", "order_lines_diff",
        "refund_orders_cur", "refund_orders_prev", "refund_orders_diff",
        "sku_count_cur", "sku_count_prev", "sku_count_diff",
        "units_per_sku_cur", "units_per_sku_prev", "units_per_sku_diff",
        "order_lines_per_sku_cur", "order_lines_per_sku_prev", "order_lines_per_sku_diff",
    ]
    for col in num_cols:
        if col in out.columns:
            out[col] = out[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")

    return out


st.sidebar.header("上传文件")
current_file = st.sidebar.file_uploader("当前对比文件（CSV）", type=["csv"], key="current")
previous_file = st.sidebar.file_uploader("基准对比文件（CSV）", type=["csv"], key="previous")
attr_file = st.sidebar.file_uploader("SKU 属性对照表（CSV）", type=["csv"], key="attr")

st.sidebar.header("分析设置")
selected_status = st.sidebar.multiselect(
    "纳入哪些订单状态",
    options=["Shipped", "Completed", "To ship", "Canceled", "Unknown"],
    default=["Shipped", "Completed"],
)

mapped_only = st.sidebar.checkbox("只分析成功映射属性的记录", value=True)

metric_choice = st.sidebar.selectbox(
    "Heat Map 指标",
    options=[
        "sales",
        "sales_per_sku",
        "units",
        "units_per_sku",
        "order_lines",
        "order_lines_per_sku",
        "refund",
        "refund_per_sku",
        "refund_orders",
        "refund_rate",
        "refund_to_sales_ratio",
    ],
    index=0,
)

max_pair_count = st.sidebar.slider("自动模式最多展示多少组属性组合", min_value=1, max_value=15, value=6)
heatmap_top_n = st.sidebar.slider("Heat Map 每轴最多保留多少个属性词", min_value=5, max_value=30, value=12)
top_summary_n = st.sidebar.slider("自动摘要最多引用多少个属性/组合", min_value=3, max_value=10, value=5)

if not (current_file and previous_file and attr_file):
    st.info("请先在左侧上传 3 个 CSV 文件。")
    st.stop()

try:
    sales_cur = load_sales(current_file, "当前文件")
    sales_prev = load_sales(previous_file, "对比文件")

    cur_label = sales_cur.attrs.get("period_label", "当前文件")
    prev_label = sales_prev.attrs.get("period_label", "对比文件")

    attr_df = load_attr(attr_file)

    merged_cur = merge_sales_attr(sales_cur, attr_df)
    merged_prev = merge_sales_attr(sales_prev, attr_df)

    merged_cur = apply_filters(merged_cur, selected_status, mapped_only)
    merged_prev = apply_filters(merged_prev, selected_status, mapped_only)

    attribute_cols = choose_attribute_columns(attr_df)
    if len(attribute_cols) < 2:
        st.error("属性对照表里可用于分析的属性列不足 2 个。")
        st.stop()

    selected_attrs = st.multiselect(
        "选择要纳入分析的属性列",
        options=attribute_cols,
        default=attribute_cols,
    )

    if len(selected_attrs) == 0:
        st.warning("请至少选择 1 个属性列。")
        st.stop()

    show_kpi_row(merged_cur, merged_prev, cur_label, prev_label)

    st.info(f"当前对比区间：{cur_label} vs {prev_label}")

    st.info(
        "说明：当前分析中，“上架月份 / 上架季度 / 上新阶段”属于根据上架时间自动生成的客观时间属性；"
        "而你表中原有的季节字段（例如 25C、26X 等）属于主观企划属性。"
        "两者可以配合使用，用来判断你的季节判断是否和实际销售节奏一致。"
    )

    st.markdown("---")
    st.header("自动分析摘要")
    overall_texts = generate_overall_summary(merged_cur, merged_prev, cur_label, prev_label)
    for t in overall_texts:
        st.write("• " + t)

    with st.expander("映射质量检查", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{cur_label} 未映射 SKU")
            unmapped_cur = (
                merged_cur.loc[~merged_cur["Mapped"], ["Seller SKU", "Base SKU"]]
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(unmapped_cur, use_container_width=True, hide_index=True)

        with col2:
            st.subheader(f"{prev_label} 未映射 SKU")
            unmapped_prev = (
                merged_prev.loc[~merged_prev["Mapped"], ["Seller SKU", "Base SKU"]]
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(unmapped_prev, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("单属性销售分析")

    focus_attr = st.selectbox("选择一个属性看区间对比", options=selected_attrs)

    comp_df = compare_periods(merged_cur, merged_prev, focus_attr)

    display_cols = [
        focus_attr,

        "sku_count_cur", "sku_count_prev", "sku_count_diff", "sku_count_pct",

        "sales_cur", "sales_prev", "sales_diff", "sales_pct",
        "sales_per_sku_cur", "sales_per_sku_prev", "sales_per_sku_diff", "sales_per_sku_pct",

        "units_cur", "units_prev", "units_diff", "units_pct",
        "units_per_sku_cur", "units_per_sku_prev", "units_per_sku_diff", "units_per_sku_pct",

        "order_lines_cur", "order_lines_prev", "order_lines_diff", "order_lines_pct",
        "order_lines_per_sku_cur", "order_lines_per_sku_prev", "order_lines_per_sku_diff", "order_lines_per_sku_pct",

        "refund_cur", "refund_prev", "refund_diff", "refund_pct",
        "refund_per_sku_cur", "refund_per_sku_prev", "refund_per_sku_diff", "refund_per_sku_pct",

        "refund_orders_cur", "refund_orders_prev", "refund_orders_diff", "refund_orders_pct",
        "refund_rate_by_orders_cur", "refund_rate_by_orders_prev", "refund_rate_by_orders_diff",
        "refund_to_sales_ratio_cur", "refund_to_sales_ratio_prev", "refund_to_sales_ratio_diff",
    ]

    comp_df_display = prepare_display_table(comp_df[display_cols].copy())
    st.dataframe(comp_df_display, use_container_width=True, hide_index=True)

    st.markdown("#### 单属性分析解读")
    attr_texts = generate_attribute_summary(comp_df, focus_attr, top_n=top_summary_n)
    for t in attr_texts:
        st.write("• " + t)

    st.markdown("#### 退款分析解读")
    refund_texts = generate_refund_summary(comp_df, focus_attr, top_n=top_summary_n)
    for t in refund_texts:
        st.write("• " + t)

    st.markdown("#### 单属性业务建议")
    suggestion_texts = generate_business_suggestions(comp_df, focus_attr, top_n=3)
    for t in suggestion_texts:
        st.write("• " + t)

    chart_metric = st.radio(
        "单属性柱状图指标",
        options=[
            "sales",
            "sales_per_sku",
            "units",
            "units_per_sku",
            "order_lines",
            "order_lines_per_sku",
            "refund",
            "refund_per_sku",
            "refund_orders",
            "refund_rate_by_orders",
            "refund_to_sales_ratio"
        ],
        horizontal=True,
        index=0,
    )

    plot_df = comp_df[[focus_attr, f"{chart_metric}_cur", f"{chart_metric}_prev"]].copy()
    plot_df = plot_df.melt(id_vars=[focus_attr], var_name="period", value_name="value")
    plot_df["period"] = plot_df["period"].map({
        f"{chart_metric}_cur": cur_label,
        f"{chart_metric}_prev": prev_label,
    })

    render_bar_chart(
        plot_df=plot_df,
        x_col=focus_attr,
        title=f"{focus_attr} - {cur_label} vs {prev_label}",
        y_label=chart_metric,
    )

    st.markdown("---")
    st.header("两两属性 Heat Map")
    st.caption("这里会先把一个单元格中的多标签属性拆开，再按单个属性词统计，所以坐标轴只会显示单个标签。")

    if len(selected_attrs) < 2:
        st.warning("至少需要 2 个属性列才能生成 Heat Map。")
        st.stop()

    heatmap_mode = st.radio(
        "Heat Map 模式",
        options=["自动推荐 Top 组合", "手动指定属性组合"],
        horizontal=True,
    )

    pair_options = list(combinations(selected_attrs, 2))
    display_pairs = []

    if heatmap_mode == "自动推荐 Top 组合":
        pair_scores = []
        for attr_x, attr_y in pair_options:
            src = build_heatmap_source(merged_cur, attr_x, attr_y, metric_choice)
            score = pd.to_numeric(src["value"], errors="coerce").fillna(0).sum() if len(src) else 0
            pair_scores.append(((attr_x, attr_y), score))

        pair_scores = sorted(pair_scores, key=lambda x: x[1], reverse=True)
        display_pairs = [p for p, _ in pair_scores[:max_pair_count]]
        st.write(f"当前自动展示前 {len(display_pairs)} 组组合。")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            manual_x = st.selectbox("选择 X 轴属性", options=selected_attrs, key="manual_x")
        with col_b:
            manual_y_options = [x for x in selected_attrs if x != manual_x]
            manual_y = st.selectbox("选择 Y 轴属性", options=manual_y_options, key="manual_y")
        display_pairs = [(manual_x, manual_y)]

    for idx, (attr_x, attr_y) in enumerate(display_pairs, start=1):
        st.subheader(f"{idx}. {attr_x} × {attr_y}")

        cur_src = build_heatmap_source(merged_cur, attr_x, attr_y, metric_choice)
        prev_src = build_heatmap_source(merged_prev, attr_x, attr_y, metric_choice)

        heatmap_texts = generate_heatmap_summary(cur_src, prev_src, attr_x, attr_y, metric_choice, top_n=top_summary_n)
        st.markdown("#### 组合分析解读")
        for t in heatmap_texts:
            st.write("• " + t)

        cur_src_small = keep_top_labels(cur_src, attr_x, attr_y, value_col="value", top_n=heatmap_top_n)
        prev_src_small = keep_top_labels(prev_src, attr_x, attr_y, value_col="value", top_n=heatmap_top_n)

        diff_src = cur_src.merge(
            prev_src,
            on=[attr_x, attr_y],
            how="outer",
            suffixes=("_cur", "_prev"),
        ).fillna(0)
        diff_src["value"] = diff_src["value_cur"] - diff_src["value_prev"]
        diff_src = diff_src[[attr_x, attr_y, "value"]]
        diff_src_small = keep_top_labels(diff_src, attr_x, attr_y, value_col="value", top_n=heatmap_top_n)

        percent_metric = metric_choice in {"refund_rate", "refund_to_sales_ratio"}

        tab1, tab2, tab3 = st.tabs([cur_label, prev_label, "差异"])

        with tab1:
            render_heatmap(
                cur_src_small,
                attr_x=attr_x,
                attr_y=attr_y,
                value_col="value",
                title=f"{cur_label} - {metric_choice}",
                diff=False,
                percent=percent_metric,
            )

        with tab2:
            render_heatmap(
                prev_src_small,
                attr_x=attr_x,
                attr_y=attr_y,
                value_col="value",
                title=f"{prev_label} - {metric_choice}",
                diff=False,
                percent=percent_metric,
            )

        with tab3:
            render_heatmap(
                diff_src_small,
                attr_x=attr_x,
                attr_y=attr_y,
                value_col="value",
                title=f"差异（{cur_label} - {prev_label}）- {metric_choice}",
                diff=True,
                percent=percent_metric,
            )

        with st.expander(f"查看 {attr_x} × {attr_y} 原始明细表", expanded=False):
            col1, col2, col3 = st.columns(3)

            def format_detail_df(df_show, percent=False):
                out = df_show.copy()
                if percent:
                    out["value"] = out["value"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                else:
                    out["value"] = out["value"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                return out

            with col1:
                st.markdown(f"**{cur_label} 明细**")
                st.dataframe(
                    format_detail_df(cur_src_small.sort_values("value", ascending=False), percent_metric),
                    use_container_width=True,
                    hide_index=True,
                )

            with col2:
                st.markdown(f"**{prev_label} 明细**")
                st.dataframe(
                    format_detail_df(prev_src_small.sort_values("value", ascending=False), percent_metric),
                    use_container_width=True,
                    hide_index=True,
                )

            with col3:
                st.markdown("**差异明细**")
                st.dataframe(
                    format_detail_df(diff_src_small.sort_values("value", ascending=False), percent_metric),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    st.header("退款风险专项分析")

    refund_attr = st.selectbox("选择一个属性看退款风险", options=selected_attrs, key="refund_attr")
    refund_df = compare_periods(merged_cur, merged_prev, refund_attr)

    risk_view = refund_df[[
        refund_attr,
        "sku_count_cur",
        "sales_cur",
        "sales_per_sku_cur",
        "refund_cur",
        "refund_per_sku_cur",
        "refund_orders_cur",
        "refund_rate_by_orders_cur",
        "refund_to_sales_ratio_cur",
        "sales_diff",
        "refund_diff",
        "refund_orders_diff",
        "refund_rate_by_orders_diff",
        "refund_to_sales_ratio_diff",
    ]].copy()

    risk_view_display = risk_view.copy()
    for col in ["sales_cur", "sales_per_sku_cur", "refund_cur", "refund_per_sku_cur", "sales_diff", "refund_diff"]:
        risk_view_display[col] = risk_view_display[col].apply(lambda x: f"${x:,.2f}")
    for col in ["sku_count_cur", "refund_orders_cur", "refund_orders_diff"]:
        risk_view_display[col] = risk_view_display[col].apply(lambda x: f"{x:,.0f}")
    for col in ["refund_rate_by_orders_cur", "refund_to_sales_ratio_cur", "refund_rate_by_orders_diff", "refund_to_sales_ratio_diff"]:
        risk_view_display[col] = risk_view_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")

    st.dataframe(risk_view_display, use_container_width=True, hide_index=True)

    top_refund_amt = refund_df.sort_values("refund_cur", ascending=False).head(10)
    top_refund_rate = refund_df.sort_values("refund_rate_by_orders_cur", ascending=False).head(10)
    top_refund_ratio = refund_df.sort_values("refund_to_sales_ratio_cur", ascending=False).head(10)
    top_refund_per_sku = refund_df.sort_values("refund_per_sku_cur", ascending=False).head(10)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**退款金额最高的属性**")
        st.dataframe(
            top_refund_amt[[refund_attr, "refund_cur", "sales_cur"]].assign(
                refund_cur=lambda d: d["refund_cur"].map(lambda x: f"${x:,.2f}"),
                sales_cur=lambda d: d["sales_cur"].map(lambda x: f"${x:,.2f}")
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c2:
        st.markdown("**退款订单占比最高的属性**")
        st.dataframe(
            top_refund_rate[[refund_attr, "refund_rate_by_orders_cur", "refund_orders_cur", "order_lines_cur"]].assign(
                refund_rate_by_orders_cur=lambda d: d["refund_rate_by_orders_cur"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"),
                refund_orders_cur=lambda d: d["refund_orders_cur"].map(lambda x: f"{x:,.0f}"),
                order_lines_cur=lambda d: d["order_lines_cur"].map(lambda x: f"{x:,.0f}")
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c3:
        st.markdown("**退款金额占销售额比例最高的属性**")
        st.dataframe(
            top_refund_ratio[[refund_attr, "refund_to_sales_ratio_cur", "refund_cur", "sales_cur"]].assign(
                refund_to_sales_ratio_cur=lambda d: d["refund_to_sales_ratio_cur"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"),
                refund_cur=lambda d: d["refund_cur"].map(lambda x: f"${x:,.2f}"),
                sales_cur=lambda d: d["sales_cur"].map(lambda x: f"${x:,.2f}")
            ),
            use_container_width=True,
            hide_index=True,
        )

    with c4:
        st.markdown("**款均退款金额最高的属性**")
        st.dataframe(
            top_refund_per_sku[[refund_attr, "refund_per_sku_cur", "sku_count_cur"]].assign(
                refund_per_sku_cur=lambda d: d["refund_per_sku_cur"].map(lambda x: f"${x:,.2f}"),
                sku_count_cur=lambda d: d["sku_count_cur"].map(lambda x: f"{x:,.0f}")
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.header("明细数据导出")

    export_df = pd.concat([merged_cur, merged_prev], ignore_index=True)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="下载映射后的明细 CSV",
        data=csv_bytes,
        file_name="mapped_sales_with_attributes.csv",
        mime="text/csv",
    )

except Exception as e:
    st.error(f"程序运行出错：{e}")
    st.exception(e)
