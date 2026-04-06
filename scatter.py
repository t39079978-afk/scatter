import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


def load_data():
    """
    CSV 파일 업로드 함수
    """
    uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드하세요", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None


def display_preview(df):
    """
    데이터 미리보기 함수
    """
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())


def display_group_statistics(df, group_col, target_col):
    """
    그룹별 통계 요약 (평균, 중앙값) 출력
    """
    st.subheader(f"그룹별 통계 요약 ({group_col} 기준)")

    # 통계량 계산
    stats_df = df.groupby(group_col)[target_col].agg(['mean', 'median', 'count']).reset_index()
    stats_df.columns = [group_col, '평균값', '중앙값', '데이터 개수']
    stats_df = stats_df.round(2)

    st.table(stats_df)

    # 인사이트 자동 추출
    max_group = stats_df.loc[stats_df['평균값'].idxmax(), group_col]
    st.info(f"분석 보고서 가이드: {group_col} 중 평균값이 가장 높은 그룹은 **{max_group}**입니다.")


def run_analysis(df):
    """
    Plotly를 이용한 시각화 (배포 시 한글 깨짐 방지 버전)
    """
    st.divider()
    st.subheader("변수 선택 및 분석 설정")

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("분석을 위해 수치형 변수가 최소 2개 필요합니다.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X축 변수 (원인)", num_cols)
    with col2:
        y_col = st.selectbox("Y축 변수 (결과/값)", num_cols)
    with col3:
        group_col = st.selectbox("그룹화 기준 (범주)", all_cols)

    if st.button("분석 보고서 생성"):
        # 1. Plotly 산점도 및 추세선
        # trendline="ols" 옵션은 선형 회귀 분석을 자동으로 수행합니다.
        fig = px.scatter(
            df, x=x_col, y=y_col, color=group_col,
            trendline="ols",
            title=f"[{x_col}]과(와) [{y_col}]의 관계 분석",
            labels={x_col: x_col, y_col: y_col},
            template="plotly_white"
        )

        # 차트 출력 (use_container_width=True로 화면에 꽉 차게)
        st.plotly_chart(fig, use_container_width=True)

        # 2. 그룹별 통계 데이터
        st.divider()
        display_group_statistics(df, group_col, y_col)

        # 3. 전체 상관관계 요약
        st.divider()
        corr_val = df[x_col].corr(df[y_col])
        st.write(f"**전체 데이터 상관계수:** {corr_val:.4f}")


def main():
    """
    메인 실행 흐름
    """
    st.set_page_config(page_title="변수 관계 분석기", layout="wide")
    st.title("변수 관계 시각화 및 보고서 분석기")

    df = load_data()

    if df is not None:
        display_preview(df)
        run_analysis(df)
    else:
        st.info("CSV 파일을 업로드하면 분석 도구가 활성화됩니다.")


if __name__ == "__main__":
    main()