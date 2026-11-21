
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── 한글 폰트 CSS ──────────────────────────────────────────────────────────
st.markdown("""\
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>\
""", unsafe_allow_html=True)

# ── CSV 읽기 (인코딩 이중 시도) ─────────────────────────────────────────────
@st.cache_data
def read_csv_file(file):
    if file is None:
        return pd.DataFrame()
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

# ── 열 이름 표준화 ──────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df

def safe_school_name(filename):
    if filename is None:
        return "우리 학교"
    base = os.path.basename(filename)
    name = base.split("_")[0].split(".")[0]
    return name if name else "우리 학교"

def apply_kr_font(fig):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
    )
    return fig

def scatter_trend(df, x_col, y_col, title):
    d = df[[x_col, y_col]].dropna()
    if d.empty or len(d) < 2:
        st.info(f"⚠️ '{x_col}'·'{y_col}' 데이터가 부족해 차트를 그릴 수 없어요.")
        return None, None
    x = d[x_col].values
    y = d[y_col].values
    r = np.corrcoef(x, y)[0, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=8, color="#2563eb"),
        hovertemplate=f"{x_col}: %{{x:.1f}}<br>{y_col}: %{{y:.2f}}<extra></extra>",
    ))
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        fx = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(
            x=fx, y=m * fx + b, mode="lines",
            line=dict(color="#dc2626", width=2),
            hoverinfo="skip",
        ))
    fig.update_layout(
        title=f"{title}  (상관계수 r = {r:.3f})",
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest",
    )
    apply_kr_font(fig)
    return fig, r

# ─────────────────────────────────────────────────────────────────────────────
#  사이드바
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

env_file    = st.sidebar.file_uploader("📁 환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("📁 생육 결과 CSV 업로드", type="csv")

with st.sidebar:
    st.markdown("---")
    st.caption("📋 생육 측정 항목 선택")

MEASURE_OPTS = ["잎 수", "지상부 길이", "지하부 길이",
                "지상부 질량", "지하부 질량", "생중량"]

# CSV 한 번만 읽기
with st.spinner("CSV를 읽는 중이에요…"):
    env_raw    = read_csv_file(env_file)
    growth_raw = read_csv_file(growth_file)
    env_df     = std_cols(env_raw)    if not env_raw.empty    else pd.DataFrame()
    growth_df  = std_cols(growth_raw) if not growth_raw.empty else pd.DataFrame()

SCHOOL = safe_school_name(
    env_file.name if env_file
    else (growth_file.name if growth_file else None)
)

selected_measures = []
if not growth_df.empty:
    available = [m for m in MEASURE_OPTS if m in growth_df.columns]
    if available:
        selected_measures = st.sidebar.multiselect(
            "측정 항목", available, default=available
        )

tab1, tab2, tab3 = st.tabs(
    ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"]
)

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — 실험 개요
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        st.header(f"🔬 {SCHOOL} 극지식물 실험 개요")

        with st.expander("📌 연구 배경 및 목적"):
            st.markdown("""
**연구 배경**
극지식물은 춥고 건조한 가혹한 환경에서 자라는 식물로, 같은 조건에서도
개체마다 생육 차이가 크게 나타납니다. 이 실험은 우리 학교 극지식물 개체들을
관찰하여 **개체별 생육 차이**가 얼마나 나는지, 그리고 **온도·습도 같은
환경 요인**이 생육에 어떤 영향을 주는지 알아보기 위한 것입니다.

**분석 목적**
1. 개체별 **생중량** 분포를 확인하고 평균·표준편차·변동계수(CV)로 차이를
   수치화합니다.
   - 평균: 모든 개체의 값을 더해 개체로 나눈 대표값
   - 표준편차: 개체가 평균에서 얼마나 흩어져 있는지 나타내는 값
     (클수록 개체 간 차이가 큼)
   - 변동계수(CV): 표준편차를 평균으로 나눈 값 → 평균 크기가 달라도
     차이 크기를 비교할 수 있음
2. **잎 수·지상부 길이**와 생중량 사이의 관계를 상관계수(r)와 산점도로
   살펴봅니다.
   - 상관계수(r): -1 ~ 1 사이 값, 1에 가까울수록 함께 증가,
     -1에 가까울수록 반대로 움직임
3. 생중량 **상위 25%** 와 **하위 25%** 개체를 비교해 어떤 차이가 있는지
   확인합니다.
            """)

        col_summary, col_cards = st.columns([1, 1])

        with col_summary:
            st.subheader("📋 데이터 요약")
            summary_rows = []
            n_growth = len(growth_df) if not growth_df.empty else 0
            summary_rows.append(["개체 수", f"{n_growth}개체" if n_growth else "업로드 필요"])
            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    summary_rows.append([
                        "측정 기간 (환경)",
                        f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}",
                    ])
            if not growth_df.empty and "time" in growth_df.columns:
                t2 = growth_df["time"].dropna()
                if len(t2) > 0:
                    summary_rows.append([
                        "측정 기간 (생육)",
                        f"{t2.min().strftime('%Y-%m-%d')} ~ {t2.max().strftime('%Y-%m-%d')}",
                    ])
            if not env_df.empty:
                env_cols = [c for c in ["temperature","humidity","ph","ec","co2"]
                            if c in env_df.columns]
                env_labels = {
                    "temperature": "온도(℃)", "humidity": "습도(%)",
                    "ph": "pH", "ec": "EC(ms/cm)", "co2": "CO2(ppm)",
                }
                summary_rows.append([
                    "환경 측정 항목",
                    ", ".join(env_labels.get(c, c) for c in env_cols) or "없음",
                ])
            if not growth_df.empty:
                gw_cols = [c for c in ["잎 수","지상부 길이","지하부 길이",
                                        "지상부 질량","지하부 질량","생중량"]
                           if c in growth_df.columns]
                gw_labels = {
                    "잎 수": "잎 수(장)", "지상부 길이": "지상부 길이(mm)",
                    "지하부 길이": "지하부 길이(mm)",
                    "지상부 질량": "지상부 질량(g)",
                    "지하부 질량": "지하부 질량(g)",
                    "생중량": "생중량(g)",
                }
                summary_rows.append([
                    "생육 측정 항목",
                    ", ".join(gw_labels.get(c, c) for c in gw_cols) or "없음",
                ])
            summary_df = pd.DataFrame(summary_rows, columns=["항목", "내용"])
            st.dataframe(summary_df, width="stretch", hide_index=True)

        with col_cards:
            st.subheader("📊 주요 지표")
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            c1.metric("총 개체수",
                      f"{n_growth}개체" if n_growth else "—")

            if not env_df.empty and "temperature" in env_df.columns:
                tv = env_df["temperature"].dropna()
                if len(tv) >= 2:
                    c2.metric("평균 온도", f"{tv.mean():.1f} ℃")
                else:
                    c2.metric("평균 온도", "데이터 부족")
            else:
                c2.metric("평균 온도", "데이터 없음")

            if not env_df.empty and "humidity" in env_df.columns:
                hv = env_df["humidity"].dropna()
                if len(hv) >= 2:
                    c3.metric("평균 습도", f"{hv.mean():.1f} %")
                else:
                    c3.metric("평균 습도", "데이터 부족")
            else:
                c3.metric("평균 습도", "데이터 없음")

            if not growth_df.empty and "생중량" in growth_df.columns:
                gv = growth_df["생중량"].dropna()
                if len(gv) >= 2:
                    c4.metric("평균 생중량", f"{gv.mean():.2f} g")
                else:
                    c4.metric("평균 생중량", "데이터 부족")
            else:
                c4.metric("평균 생중량", "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — 환경 데이터
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        has_env = not env_df.empty

        if not has_env:
            st.info("📁 사이드바에서 환경 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            st.header(f"🌡️ {SCHOOL} 환경 데이터")

            env_num_cols = [c for c in ["temperature","humidity","ph","ec","co2"]
                            if c in env_df.columns]
            if env_num_cols:
                st.subheader("📋 환경 요약 통계")
                env_summary = env_df[env_num_cols].dropna().agg(["mean","min","max"]).T
                env_summary.columns = ["평균", "최소", "최대"]
                label_map = {
                    "temperature": "온도(℃)", "humidity": "습도(%)",
                    "ph": "pH", "ec": "EC(ms/cm)", "co2": "CO2(ppm)",
                }
                env_summary.index = [label_map.get(i, i) for i in env_summary.index]
                env_summary = env_summary.round(2)
                st.dataframe(env_summary, width="stretch", hide_index=False)
            else:
                st.info("⚠️ 환경의 숫자 열이 없어 요약 통계를 표시할 수 없어요.")

            if "time" in env_df.columns:
                env_df_day = env_df.copy()
                env_df_day["day"] = env_df_day["time"].dt.floor("D")
                st.subheader("📈 날짜별 환경 변화 (일별 평균)")

                day_cols = [c for c in ["temperature","humidity","ph","ec"]
                            if c in env_df.columns]
                if day_cols:
                    n_graphs = len(day_cols)
                    chart_cols = st.columns(min(n_graphs, 3))
                    day_agg = env_df_day.groupby("day")[day_cols].mean().dropna()

                    for idx, col in enumerate(day_cols):
                        if col not in day_agg.columns or day_agg.empty:
                            continue
                        d = day_agg[col].dropna()
                        if len(d) < 2:
                            continue
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=d.index, y=d.values,
                            mode="lines+markers",
                            line=dict(color="#2563eb", width=2),
                            marker=dict(size=5),
                            hovertemplate=f"{col}: %{{y:.2f}}<extra></extra>",
                        ))
                        fig.update_layout(
                            title=f"일별 평균 {label_map.get(col, col)}",
                            xaxis_title="날짜",
                            yaxis_title=label_map.get(col, col),
                            height=220,
                        )
                        apply_kr_font(fig)
                        chart_cols[idx].plotly_chart(fig, width="stretch")
                else:
                    st.info("⚠️ 온도·습도·pH·EC 열이 없어 시계열을 그릴 수 없어요.")

            with st.expander("📂 환경 데이터 원본 보기 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch", hide_index=True)
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 환경 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{SCHOOL}_환경데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("환경 데이터가 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — 생육 결과
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        has_growth = not growth_df.empty

        if not has_growth:
            st.info("📁 사이드바에서 생육 결과 CSV를 업로드하면 이 탭의 내용이 표시돼요.")
        else:
            st.header(f"📊 {SCHOOL} 생육 결과 분석")

            if "생중량" not in growth_df.columns \
               and {"지상부 질량", "지하부 질량"} <= set(growth_df.columns):
                growth_df["생중량"] = \
                    growth_df["지상부 질량"] + growth_df["지하부 질량"]

            gw_col = "생중량"
            gw_vals = growth_df[gw_col].dropna() \
                if gw_col in growth_df.columns else pd.Series(dtype=float)

            if len(gw_vals) >= 2:
                avg_gw  = gw_vals.mean()
                max_idx = gw_vals.idxmax()
                min_idx = gw_vals.idxmin()
                if "개체번호" in growth_df.columns:
                    max_id = growth_df.loc[max_idx, "개체번호"]
                    min_id = growth_df.loc[min_idx, "개체번호"]
                else:
                    max_id = f"#{max_idx}"
                    min_id = f"#{min_idx}"
                max_val = gw_vals.max()
                min_val = gw_vals.min()
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("📈 평균 생중량", f"{avg_gw:.2f} g")
                cc2.metric("🏆 최대 개체",
                           f"#{max_id}  ({max_val:.2f} g)")
                cc3.metric("📉 최소 개체",
                           f"#{min_id}  ({min_val:.2f} g)")
            else:
                st.info("⚠️ 생중량 데이터가 부족해 핵심 결과 카드를 표시할 수 없어요.")

            with st.expander("📐 통계 용어 간단 설명"):
                st.markdown("""
- **평균(Mean)**: 모든 값을 더해 개수로 나눈 대표값
- **표준편차(SD)**: 값이 평균에서 얼마나 퍼져 있는지 나타냄
  (클수록 개체 간 차이가 큼)
- **변동계수(CV, Coefficient of Variation)**:
  표준편차 ÷ 평균 × 100(%). 평균이 다른 그룹끼리도 변동 크기를 비교 가능
                """)

            st.subheader("🌿 개체별 생육 비교 (막대 그래프)")
            bio_cols = ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
            available_bio = [c for c in bio_cols if c in growth_df.columns]

            if available_bio:
                n_bio = len(available_bio)
                bio_chart_cols = st.columns(min(n_bio, 4))
                for idx, col in enumerate(available_bio):
                    cols_needed = [c for c in ["개체번호", col]
                                   if c in growth_df.columns]
                    d = growth_df[cols_needed].dropna()
                    if d.empty or len(d) < 2:
                        bio_chart_cols[idx].info(
                            f"⚠️ '{col}' 데이터가 부족해요."
                        )
                        continue
                    x_lab = "개체번호" if "개체번호" in d.columns else d.columns[0]
                    fig = px.bar(
                        d, x=x_lab, y=col,
                        labels={x_lab: "개체", col: col},
                        title=f"개체별 {col}",
                        hover_data=[col],
                    )
                    fig.update_layout(showlegend=False, height=260)
                    apply_kr_font(fig)
                    bio_chart_cols[idx].plotly_chart(fig, width="stretch")
            else:
                st.info("⚠️ 생육 데이터의 숫자 열이 없어 막대 그래프를 그릴 수 없어요.")

            st.subheader("📦 생중량 분포")
            if gw_col in growth_df.columns:
                gw_clean = growth_df[[gw_col]].dropna()
                if len(gw_clean) >= 2:
                    gw_arr  = gw_clean[gw_col].values
                    gw_mean = gw_arr.mean()
                    gw_sd   = gw_arr.std(ddof=1)
                    gw_cv   = (gw_sd / gw_mean * 100) if gw_mean > 0 else float("nan")

                    st.markdown(f"""
**생중량 기술통계**
- 평균: **{gw_mean:.2f} g**
- 표준편차: **{gw_sd:.2f} g**
- 변동계수(CV): **{gw_cv:.1f}%**
  → CV가 20% 미만이면 개체 간 차이가 비교적 작고,
    20% 이상이면 차이가 큰 편이에요.
                    """)

                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=gw_arr,
                        name="생중량(g)",
                        fillcolor="#93c5fd",
                        line=dict(color="#1d4ed8"),
                        boxpoints="outliers",
                        marker=dict(size=6),
                        hovertemplate="생중량: %{y:.2f} g<extra></extra>",
                    ))
                    fig_box.update_layout(
                        title="개체별 생중량 분포 (박스플롯)",
                        yaxis_title="생중량(g)",
                        height=340,
                    )
                    apply_kr_font(fig_box)
                    st.plotly_chart(fig_box, width="stretch")
                else:
                    st.info("⚠️ 생중량 데이터가 2개 미만이라 분포를 그릴 수 없어요.")
            else:
                st.info("⚠️ 생중량 열이 없어 분포 그래프를 그릴 수 없어요.")

            if gw_col in growth_df.columns and len(gw_vals) >= 4:
                q75 = gw_vals.quantile(0.75)
                q25 = gw_vals.quantile(0.25)
                top_df    = growth_df[growth_df[gw_col] >= q75].dropna(subset=[gw_col])
                bottom_df = growth_df[growth_df[gw_col] <= q25].dropna(subset=[gw_col])

                if len(top_df) > 0 and len(bottom_df) > 0:
                    st.subheader("🏅 생중량 상위 25% vs 하위 25% 비교")
                    comp_cols = [c for c in available_bio if c in growth_df.columns]
                    if comp_cols:
                        comp_data = []
                        for grp_name, grp_df in [
                            ("상위 25%", top_df), ("하위 25%", bottom_df)
                        ]:
                            row = [grp_name]
                            for cc in comp_cols:
                                v = grp_df[cc].dropna()
                                row.append(f"{v.mean():.2f}" if len(v) > 0 else "—")
                            comp_data.append(row)
                        comp_df = pd.DataFrame(
                            comp_data,
                            columns=["그룹"] + comp_cols,
                        )
                        st.dataframe(comp_df, width="stretch", hide_index=True)

            st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")
            st.caption("※ 추세선은 np.polyfit으로 직접 계산했어요 (statsmodels 불필요).")

            scatter_pairs = [
                ("잎 수",       gw_col, "잎 수 vs 생중량"),
                ("지상부 길이", gw_col, "지상부 길이 vs 생중량"),
            ]
            for x_col, y_col, ttl in scatter_pairs:
                if x_col in growth_df.columns and y_col in growth_df.columns:
                    fig_sc, r_val = scatter_trend(growth_df, x_col, y_col, ttl)
                    if fig_sc is not None:
                        st.plotly_chart(fig_sc, width="stretch")
                    else:
                        st.info(f"⚠️ '{x_col}' 또는 '{y_col}' 데이터가 부족해요.")
                else:
                    missing = [c for c in [x_col, y_col]
                               if c not in growth_df.columns]
                    st.info(
                        f"⚠️ {', '.join(missing)} 열이 없어 상관관계 분석을 건너뛸게요."
                    )

            with st.expander("📂 생육 데이터 원본 보기 + CSV 다운로드"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch", hide_index=True)
                    csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        label="⬇️ 생육 데이터 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"{SCHOOL}_생육결과데이터.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("생육 데이터가 없어요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
