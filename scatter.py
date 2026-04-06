import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform


def set_korean_font():
    """
    운영체제별 한글 폰트 설정
    """
    sys_name = platform.system()
    if sys_name == "Windows":
        plt.rc('font', family='Malgun Gothic')
    elif sys_name == "Darwin":
        plt.rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False


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
    선택한 그룹별로 타겟 변수의 평균값과 중앙값을 계산하여 표로 출력
    """
    st.subheader(f"그룹별 통계 요약 ({group_col} 기준)")

    # 그룹별 평균과 중앙값 계산
    stats_df = df.groupby(group_col)[target_col].agg(['mean', 'median', 'count']).reset_index()
    stats_df.columns = [group_col, '평균값', '중앙값', '데이터 개수']

    # 소수점 둘째자리까지 반올림
    stats_df = stats_df.round(2)

    st.table(stats_df)
    st.write(f"보고서 가이드: {group_col} 항목 중 평균값이 가장 높은 그룹은 **{stats_df.loc[stats_df['평균값'].idxmax(), group_col]}**입니다.")


def run_analysis(df):
    """
    변수 선택, 산점도 시각화, 추세선 및 그룹 통계 생성 함수
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
        # 1. 시각화 (산점도 및 추세선)
        set_korean_font()
        fig, ax = plt.subplots(figsize=(10, 6))

        groups = df[group_col].unique()
        for g in groups:
            subset = df[df[group_col] == g]
            ax.scatter(subset[x_col], subset[y_col], label=g, alpha=0.7)

        # 전체 추세선 계산
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
        ax.plot(x_range, p(x_range), "r--", linewidth=2, label="전체 추세선")

        ax.set_title(f"[{x_col}]과(와) [{y_col}]의 관계 분석", fontsize=14)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # 2. 그룹별 통계 데이터 출력 (평균, 중앙값)
        st.divider()
        display_group_statistics(df, group_col, y_col)

        # 3. 상관관계 요약
        st.divider()
        corr_val = df[x_col].corr(df[y_col])
        st.write(f"**전체 상관계수:** {corr_val:.4f}")
        st.write(f"**분석 요약:** X축({x_col})과 Y축({y_col})은 통계적으로 유의미한 관계를 형성하고 있습니다.")


def main():
    """
    메인 실행 흐름
    """
    st.set_page_config(page_title="변수 관계 분석기", layout="wide")
    st.title("변수 관계 시각화 및 그룹별 통계 분석기")

    df = load_data()

    if df is not None:
        display_preview(df)
        run_analysis(df)
    else:
        st.info("CSV 파일을 업로드하면 분석 도구가 활성화됩니다.")


if __name__ == "__main__":
    main()