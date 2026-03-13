import re
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="SKU 属性销售分析", layout="wide")

st.title("SKU 属性销售分析仪表盘")
st.caption("上传近7天销售数据、过去7天销售数据、SKU属性对照表后，自动完成属性映射、周对比与两两属性 Heat Map 分析。")


# =========================
# 工具函数
# =========================
def clean_text(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return x if x != "" else np.nan


def normalize_base_sku(sku: str):
    """把销售表里的 Seller SKU 标准化为基础 SKU。
    例：NPF015-M -> NPF015
        NPJ005-S -> NPJ005
    """
    if pd.isna(sku):
        return np.nan
    sku = str(sku).strip().upper()
    sku = sku.split("-")[0]

    # 优先匹配常见结构：N + 1~2个字母 + [F/X/J] + 3位数字
    m = re.search(r"N[A-Z]{1,2}[FXJ]\d{3}", sku)
    if m:
        return m.group(0)

    # 兜底：N + 2~3位字母/数字组合 + 3位数字
    m = re.search(r"N[A-Z0-9]{2,3}\d{3}", sku)
    if m:
        return m.group(0)

    return sku


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce").fillna(0)


def load_sales(file, label: str):
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Seller SKU", "Quantity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} 缺少必要列: {missing}")

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

    df["Period"] = label
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
    attr = attr.drop_duplicates(subset=["SKU"], keep="first").copy()
    return attr


def merge_sales_attr(sales_df, attr_df):
    merged = sales_df.merge(attr_df, how="left", left_on="Base SKU", right_on="SKU")
    merged["Mapped"] = merged["SKU"].notna()
    return merged


def choose_attribute_columns(df):
    excluded = {
        "SKU", "中文名称", "款式英文名称", "图片", "上架时间",
        "Seller SKU", "Base SKU", "Period", "Mapped"
    }
    candidates = []
    for col in df.columns:
        if col in excluded:
            continue
        if col in ["Quantity", "Sales", "Refund Amount"]:
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


def aggregate_by_attribute(df, attr_col):
    tmp = df.copy()
    tmp[attr_col] = tmp[attr_col].fillna("(空值)")
    g = tmp.groupby(attr_col, dropna=False).agg(
        order_lines=("Seller SKU", "count"),
        units=("Quantity", "sum"),
        sales=("Sales", "sum"),
        refund=("Refund Amount", "sum"),
        sku_count=("Base SKU", "nunique")
    ).reset_index()
    return g.sort_values("sales", ascending=False)


def compare_periods(cur_df, prev_df, attr_col):
    cur = aggregate_by_attribute(cur_df, attr_col).rename(columns={
        "order_lines": "order_lines_cur",
        "units": "units_cur",
        "sales": "sales_cur",
        "refund": "refund_cur",
        "sku_count": "sku_count_cur",
    })
    prev = aggregate_by_attribute(prev_df, attr_col).rename(columns={
        "order_lines": "order_lines_prev",
        "units": "units_prev",
        "sales": "sales_prev",
        "refund": "refund_prev",
        "sku_count": "sku_count_prev",
    })
    merged = cur.merge(prev, on=attr_col, how="outer").fillna(0)

    for metric in ["order_lines", "units", "sales", "refund", "sku_count"]:
        merged[f"{metric}_diff"] = merged[f"{metric}_cur"] - merged[f"{metric}_prev"]
        merged[f"{metric}_pct"] = np.where(
            merged[f"{metric}_prev"] == 0,
            np.nan,
            merged[f"{metric}_diff"] / merged[f"{metric}_prev"]
        )

    return merged.sort_values("sales_diff", ascending=False)


def build_heatmap_source(df, attr_x, attr_y, metric):
    tmp = df.copy()
    tmp[attr_x] = tmp[attr_x].fillna("(空值)")
    tmp[attr_y] = tmp[attr_y].fillna("(空值)")

    metric_map = {
        "units": "Quantity",
        "sales": "Sales",
        "order_lines": "Seller SKU",
        "refund": "Refund Amount",
    }

    if metric == "order_lines":
        pivot = tmp.groupby([attr_y, attr_x]).size().reset_index(name="value")
    else:
        pivot = tmp.groupby([attr_y, attr_x], dropna=False)[metric_map[metric]].sum().reset_index(name="value")

    return pivot


def show_kpi_row(cur_df, prev_df):
    cur_sales = cur_df["Sales"].sum()
    prev_sales = prev_df["Sales"].sum()
    cur_units = cur_df["Quantity"].sum()
    prev_units = prev_df["Quantity"].sum()
    cur_lines = len(cur_df)
    prev_lines = len(prev_df)
    cur_map = cur_df["Mapped"].mean() if len(cur_df) else 0
    prev_map = prev_df["Mapped"].mean() if len(prev_df) else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("近7天销售额", f"${cur_sales:,.2f}", f"${cur_sales - prev_sales:,.2f}")
    c2.metric("近7天销量", f"{cur_units:,.0f}", f"{cur_units - prev_units:,.0f}")
    c3.metric("近7天订单行", f"{cur_lines:,}", f"{cur_lines - prev_lines:,}")
    c4.metric("SKU映射率", f"{cur_map:.1%}", f"{(cur_map - prev_map):.1%}")


# =========================
# 侧边栏上传
# =========================
st.sidebar.header("上传文件")
current_file = st.sidebar.file_uploader("近 7 天销售数据（CSV）", type=["csv"], key="current")
previous_file = st.sidebar.file_uploader("过去 7 天销售数据（CSV）", type=["csv"], key="previous")
attr_file = st.sidebar.file_uploader("SKU 属性对照表（CSV）", type=["csv"], key="attr")

st.sidebar.header("分析设置")
default_status = ["Shipped", "Completed"]
selected_status = st.sidebar.multiselect(
    "纳入哪些订单状态",
    options=["Shipped", "Completed", "To ship", "Canceled", "Unknown"],
    default=default_status,
)
mapped_only = st.sidebar.checkbox("只分析成功映射属性的记录", value=True)
metric_choice = st.sidebar.selectbox(
    "Heat Map 指标",
    options=["sales", "units", "order_lines", "refund"],
    index=0,
)
max_pair_count = st.sidebar.slider("最多展示多少组属性组合", min_value=1, max_value=15, value=6)


if not (current_file and previous_file and attr_file):
    st.info("请先在左侧上传 3 个 CSV 文件。")
    st.stop()


# =========================
# 主流程
# =========================
try:
    sales_cur = load_sales(current_file, "近7天")
    sales_prev = load_sales(previous_file, "过去7天")
    attr_df = load_attr(attr_file)

    merged_cur = merge_sales_attr(sales_cur, attr_df)
    merged_prev = merge_sales_attr(sales_prev, attr_df)

    merged_cur = apply_filters(merged_cur, selected_status, mapped_only)
    merged_prev = apply_filters(merged_prev, selected_status, mapped_only)

    attribute_cols = choose_attribute_columns(attr_df)
    if len(attribute_cols) < 2:
        st.error("属性对照表里可用于分析的属性列不足 2 个，请检查 Season / Color / Style / Holiday / Design Elements 等列。")
        st.stop()

    selected_attrs = st.multiselect(
        "选择要分析的属性列",
        options=attribute_cols,
        default=attribute_cols,
    )

    if len(selected_attrs) == 0:
        st.warning("请至少选择 1 个属性列。")
        st.stop()

    show_kpi_row(merged_cur, merged_prev)

    with st.expander("映射质量检查", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("近7天未映射 SKU")
            unmapped_cur = (
                merged_cur.loc[~merged_cur["Mapped"], ["Seller SKU", "Base SKU"]]
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(unmapped_cur, use_container_width=True, hide_index=True)
        with col2:
            st.subheader("过去7天未映射 SKU")
            unmapped_prev = (
                merged_prev.loc[~merged_prev["Mapped"], ["Seller SKU", "Base SKU"]]
                .value_counts()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            st.dataframe(unmapped_prev, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.header("单属性销售分析")

    focus_attr = st.selectbox("选择一个属性看周对比", options=selected_attrs)
    comp_df = compare_periods(merged_cur, merged_prev, focus_attr)

    display_cols = [
        focus_attr,
        "sales_cur", "sales_prev", "sales_diff", "sales_pct",
        "units_cur", "units_prev", "units_diff", "units_pct",
        "order_lines_cur", "order_lines_prev", "order_lines_diff", "order_lines_pct",
        "refund_cur", "refund_prev", "refund_diff", "refund_pct",
        "sku_count_cur", "sku_count_prev", "sku_count_diff", "sku_count_pct",
    ]

    st.dataframe(comp_df[display_cols], use_container_width=True, hide_index=True)

    chart_metric = st.radio(
        "单属性柱状图指标",
        options=["sales", "units", "order_lines", "refund"],
        horizontal=True,
        index=0,
    )

    plot_df = comp_df[[focus_attr, f"{chart_metric}_cur", f"{chart_metric}_prev"]].copy()
    plot_df = plot_df.melt(id_vars=[focus_attr], var_name="period", value_name="value")
    plot_df["period"] = plot_df["period"].map({
        f"{chart_metric}_cur": "近7天",
        f"{chart_metric}_prev": "过去7天",
    })

    fig_bar = px.bar(
        plot_df,
        x=focus_attr,
        y="value",
        color="period",
        barmode="group",
        title=f"{focus_attr} - 近7天 vs 过去7天",
    )
    fig_bar.update_layout(xaxis_title=focus_attr, yaxis_title=chart_metric)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.header("两两属性 Heat Map")

    if len(selected_attrs) < 2:
        st.warning("至少需要 2 个属性列才能生成 Heat Map。")
        st.stop()

    pair_options = list(combinations(selected_attrs, 2))
    st.write(f"当前可生成 {len(pair_options)} 组属性组合。")

    # 依据近7天指标总量，为用户优先展示更有价值的组合
    pair_scores = []
    for attr_x, attr_y in pair_options:
        src = build_heatmap_source(merged_cur, attr_x, attr_y, metric_choice)
        score = src["value"].sum() if len(src) else 0
        pair_scores.append(((attr_x, attr_y), score))

    pair_scores = sorted(pair_scores, key=lambda x: x[1], reverse=True)
    display_pairs = [p for p, _ in pair_scores[:max_pair_count]]

    for idx, (attr_x, attr_y) in enumerate(display_pairs, start=1):
        st.subheader(f"{idx}. {attr_x} × {attr_y}")
        c1, c2 = st.columns(2)

        cur_src = build_heatmap_source(merged_cur, attr_x, attr_y, metric_choice)
        prev_src = build_heatmap_source(merged_prev, attr_x, attr_y, metric_choice)

        with c1:
            if len(cur_src) == 0:
                st.info("近7天该组合没有数据")
            else:
                fig_cur = px.density_heatmap(
                    cur_src,
                    x=attr_x,
                    y=attr_y,
                    z="value",
                    histfunc="sum",
                    text_auto=True,
                    title=f"近7天 - {metric_choice}",
                )
                fig_cur.update_layout(xaxis_title=attr_x, yaxis_title=attr_y)
                st.plotly_chart(fig_cur, use_container_width=True)

        with c2:
            if len(prev_src) == 0:
                st.info("过去7天该组合没有数据")
            else:
                fig_prev = px.density_heatmap(
                    prev_src,
                    x=attr_x,
                    y=attr_y,
                    z="value",
                    histfunc="sum",
                    text_auto=True,
                    title=f"过去7天 - {metric_choice}",
                )
                fig_prev.update_layout(xaxis_title=attr_x, yaxis_title=attr_y)
                st.plotly_chart(fig_prev, use_container_width=True)

        # 差异热力图
        all_src = cur_src.merge(
            prev_src,
            on=[attr_y, attr_x],
            how="outer",
            suffixes=("_cur", "_prev"),
        ).fillna(0)
        all_src["value_diff"] = all_src["value_cur"] - all_src["value_prev"]

        st.markdown("**差异 Heat Map（近7天 - 过去7天）**")
        if len(all_src) == 0:
            st.info("该组合没有可比较数据")
        else:
            fig_diff = px.density_heatmap(
                all_src,
                x=attr_x,
                y=attr_y,
                z="value_diff",
                histfunc="sum",
                text_auto=True,
                title=f"差异 - {metric_choice}",
            )
            fig_diff.update_layout(xaxis_title=attr_x, yaxis_title=attr_y)
            st.plotly_chart(fig_diff, use_container_width=True)

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
