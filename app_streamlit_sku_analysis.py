def render_heatmap(src_df, attr_x, attr_y, value_col, title, diff=False):
    if len(src_df) == 0:
        st.info("该组合没有数据。")
        return

    matrix = src_df.pivot(index=attr_y, columns=attr_x, values=value_col).fillna(0)
    matrix = sort_matrix(matrix)

    st.markdown(f"**{title}**")

    if px is not None:
        if diff:
            max_abs = float(np.abs(matrix.values).max()) if matrix.size else 0.0
            if max_abs == 0:
                max_abs = 1.0

            fig = px.imshow(
                matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdYlGn",
                range_color=[-max_abs, max_abs],
            )
        else:
            fig = px.imshow(
                matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="YlOrRd",
            )

        fig.update_traces(
            textfont_size=12,
            hovertemplate=f"{attr_y}: %{{y}}<br>{attr_x}: %{{x}}<br>值: %{{z:,.2f}}<extra></extra>"
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
        if diff:
            styled = matrix.style.format("{:.2f}").background_gradient(cmap="RdYlGn", axis=None)
        else:
            styled = matrix.style.format("{:.2f}").background_gradient(cmap="YlOrRd", axis=None)

        st.dataframe(styled, use_container_width=True)
        st.caption("当前环境未安装 plotly，已自动切换为表格热力图。")
