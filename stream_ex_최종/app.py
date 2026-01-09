import os
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from analysis.car.time import (
    fit_arima, forecast_12_months, plot_diff_1,
    plot_forecast, plot_monthly, stationarity_test
)

from analysis.cctv.data import load_data_cctv
from analysis.cctv.eda import (
    plot_cctv_vs_death, plot_corr_heatmap,
    plot_histograms, plot_severity_box
)
from analysis.cctv.model import (
    evaluate_model, predict_severity, train_model
)

from analysis.car.data import load_data_car_month
from analysis.traffic_car.data import load_data_traffic
from analysis.traffic_car.traffic import (
    analyze_correlation, make_yearly_summary, plot_traffic_growth_bar
)
from analysis.traffic_car.vehicle import (
    make_monthly_summary, plot_vehicle_trend
)

from analysis.parking_car.ridge import run_parking_poly_regression, run_ridge
from analysis.parking_car.visual_parking import (
    plot_correlation, predict_future, run_parking_regression
)
from analysis.parking_car.data import load_data_parking

from analysis.population_car.cluster import run_clustering
from analysis.population_car.regression import run_regression
from analysis.population_car.logistic import run_logistic
from analysis.population_car.data import load_data

from analysis.public_transit.data import load_data_transit
from analysis.public_transit.visual_transit import run_visual_transit
from analysis.public_transit.multireg import run_multireg



st.set_page_config(
    page_title="서울시 교통 데이터 분석",
    page_icon="🚦",
    layout="wide"
)

st.cache_data.clear()  

plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles/style.css")

st.markdown("## 🚦 서울시 교통 데이터 분석 프로젝트")
st.caption(
    "자동차 등록 · 교통량 · CCTV · 인구 · 대중교통 데이터를 활용한 종합 분석 대시보드"
)
st.divider()

st.sidebar.title("📚 분석 메뉴")

menu = st.sidebar.radio(
    " ",
    [
        "🏠 Home",
        "📘 시계열 분석",
        "📊 CCTV & 사고",
        "🚗 교통량 vs 자동차",
        "🚌 대중교통 영향",
        "🏙 인구 기반 분석",
        "🅿️ 주차면 분석"
    ],
    label_visibility="collapsed"
)

if menu == "🏠 Home":

    st.markdown("### 📌 분석 주제 개요")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("📘 **시계열 분석**  \n자동차 등록 대수 변화 예측")
    with col2:
        st.markdown("📊 **CCTV & 사고**  \n안전 인프라와 사고 심각도")
    with col3:
        st.markdown("🚗 **교통량 분석**  \n등록대수와 교통량 상관관계")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🚌 **대중교통 영향**  \n버스 이용과 승용차 변화")
    with col2:
        st.markdown("🏙 **인구 기반 분석**  \n자치구별 자동차 증감")
    with col3:
        st.markdown("🅿️ **주차면 분석**  \n자동차 수 vs 주차 인프라")

    st.info("⬅ 좌측 메뉴에서 분석을 선택하세요.")

elif menu == "📘 시계열 분석":

    df = load_data_car_month()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 월별 자동차 등록 추세")
        fig1 = plot_monthly(df)
        st.pyplot(fig1)
    with col2:
        st.subheader("📉 1차 차분")
        fig2, diff_1 = plot_diff_1(df)
        st.pyplot(fig2)

    st.subheader("🧪 정상성 검정")
    result = stationarity_test(diff_1)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### ADF Test")
        st.write(f"ADF Statistic: **{result['adf_stat']:.4f}**")
        st.write(f"p-value: **{result['adf_p']:.4f}**")
        st.json(result['adf_crit'])
    with col2:
        st.markdown("### KPSS Test")
        st.write(f"KPSS Statistic: **{result['kpss_stat']:.4f}**")
        st.write(f"p-value: **{result['kpss_p']:.4f}**")
        st.json(result['kpss_crit'])
    
    arima_result = fit_arima(df)

    st.subheader("📊 ARIMA(1,1,1) 모델 요약")
    col1, col2, col3 = st.columns(3)
    col1.metric("AIC", f"{arima_result.aic:.2f}")
    col2.metric("BIC", f"{arima_result.bic:.2f}")
    col3.metric("관측치 수", arima_result.nobs)

    with st.expander("📄 ARIMA 상세 결과 (원본)"):
        st.text(arima_result.summary().as_text())

    st.subheader("🔮 미래 12개월 자동차 등록 대수 예측")

    forecast_mean, conf_int = forecast_12_months(
        arima_result,
        df.index[-1]
    )

    fig_forecast = plot_forecast(df, forecast_mean, conf_int)
    st.pyplot(fig_forecast)

elif menu == "📊 CCTV & 사고":
    st.header("📊 교통 관련 CCTV 갯수 / 설치된 CCTV 지역의 사고건수 분석")
    df = load_data_cctv()

    tabs = st.tabs([
        "📊 EDA",
        "🔥 상관관계",
        "🤖 사고 심각도 모델"
    ])

    num_cols = [
        '사망자수(명)', '발생건수(건)', '부상자수(명)',
        '사고당사망률', '사고당부상률', 'CCTV설치대수'
    ]

    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CCTV vs 사고당 사망률")
            st.pyplot(plot_cctv_vs_death(df))
        

        st.subheader("변수 분포")
        st.pyplot(plot_histograms(df, num_cols))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("심각도별 사망률")
            st.pyplot(plot_severity_box(df))

    with tabs[1]:
        st.subheader("변수 간 상관계수")
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_corr_heatmap(df, num_cols))        

    with tabs[2]:
        pipe, le, X_test, y_test = train_model(df)
        eval_result = evaluate_model(pipe, X_test, y_test, le)

        st.metric("정확도", f"{eval_result['accuracy']:.3f}")

        with st.expander("📄 분류 리포트"):
            st.text(eval_result['report'])

        st.subheader("🔮 사고 심각도 예측")

        sample = {
            '발생건수(건)': st.number_input("발생 건수", 0, 10000, 1500),
            '부상자수(명)': st.number_input("부상자 수", 0, 10000, 2000),
            '사고당사망률': st.number_input("사고당 사망률", 0.0, 1.0, 0.01, format="%.3f"),
            '사고당부상률': st.number_input("사고당 부상률", 0.0, 10.0, 1.4),
            'CCTV설치대수': st.number_input("CCTV 설치 대수", 0, 5000, 300)
        }

        pred = predict_severity(pipe, le, sample)
        st.success(f"예측 사고 심각도: **{pred}**")

elif menu == "🚗 교통량 vs 자동차":
    st.header("📈 자동차 등록과 교통량 관계 분석")
    
    df, df_traffic = load_data_traffic()
    total_summary = make_monthly_summary(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 차종별 및 전체 자동차 등록 추이",
        "📊 교통량 증감 시각화",
        "📈 자동차 등록과 교통량 관계 분석",
        "📈 test page"
    ])

    with tab1:
        st.subheader("📊 차종별 및 전체 자동차 등록 추이")

        fig_trend_all = plot_vehicle_trend(total_summary)
        st.pyplot(fig_trend_all)

        st.subheader("📋 연도별 자동차 등록 요약")
        yearly_df = make_yearly_summary(total_summary)
        st.dataframe(yearly_df)

    with tab2:
        st.subheader("📊 연도별 교통량 증감률 비교")

        fig_bar = plot_traffic_growth_bar(df_traffic)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_bar)
        

    with tab3:
        st.subheader("📈 교통량 증가와 자동차 등록 증가의 관계")

        corr, fig_trend, fig_scatter = analyze_correlation(
            total_summary,
            df_traffic
        )

        st.metric("상관계수", f"{corr:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_trend)
        with col2:
            st.pyplot(fig_scatter)
            
    with tab4:
        st.subheader("📊 test")

        
        
elif menu == "🚌 대중교통 영향":
    st.header("🚌 대중교통 이용 영향 분석")
    df = load_data_transit()
    
    tab1, tab2, tab3 = st.tabs([
        "📊 교통 데이터 시각화",
        "📈 자동차 증가 예측 모델",
        "⚖️ 교통수단 영향력 비교"
    ])


    with tab1:
        st.header("📊 대중교통 · 자동차 변화 관계 시각화")
        st.caption("버스·지하철 지표와 연간 자동차 증감 관계를 확인합니다.")

        fig_bus_car, fig_bus_sub = run_visual_transit(df)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_bus_car)
        with col2:
            st.pyplot(fig_bus_sub)

    with tab2:
        st.header("📈 다항 회귀 및 Ridge 회귀 분석")
        st.caption("과적합 여부와 규제 강도(α)에 따른 성능 변화를 비교합니다.")

        base_df, ridge_df, degree_df, best_alpha = run_multireg(df)

        st.subheader("① 다항 회귀 성능 비교 (과적합 확인)")
        st.dataframe(base_df)

        st.subheader("② Ridge 회귀 α 튜닝 결과")
        st.dataframe(ridge_df)

        st.success(f"✅ Best alpha (Test R² 기준): **{best_alpha}**")

        st.subheader("③ 차수별 모델 성능 비교")
        st.dataframe(degree_df)
    with tab3:
        st.subheader("📉 Ridge 회귀 계수 비교 (α = 100, 표준화)")
        st.image("images/transit_ridge.png", width=700)

elif menu == "🏙 인구 기반 분석":
    st.header("🏙 인구 변화 기반 자동차 분석")
    df = load_data()
    district_list = df["district"].unique()

    selected_district = st.selectbox(
        "자치구 선택",
        district_list
    )
    st.toast(f"{selected_district} 분석 실행됨")

    tab1, tab2, tab3 = st.tabs([
        "📊 군집 분석",
        "📈 회귀 분석",
        "🧠 로지스틱 회귀"
    ])

    # ------------------
    # 군집
    # ------------------
    with tab1:
        st.markdown("### 📊 군집 분석")

        if selected_district != "전체":
            st.warning("⚠️ 군집 분석은 전체 선택 시만 가능합니다.")
        else:
            df_cluster, summary_df, fig_bar, fig_scatter = run_clustering(df, selected_district)

            st.subheader("📋 자치구별 군집 결과")
            st.dataframe(df_cluster)

            st.subheader("📊 군집 요약")
            st.dataframe(summary_df)

            
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(fig_scatter)
            with col2:
                st.pyplot(fig_bar)
    # ------------------
    # 회귀
    # ------------------
    with tab2:
        st.markdown("### 📈 선형 회귀 분석")

        fig, desc, corr, coef_df, r2 = run_regression(df, selected_district)

        st.markdown("#### 📊 기초 통계")
        st.dataframe(desc)

        st.markdown("#### 🔗 상관계수")
        st.dataframe(corr)

        st.markdown("#### 📈 회귀 결과")
        st.dataframe(coef_df)

        st.markdown("#### 📊 모델 성능 (R²)")
        col1, col2, col3 = st.columns(3)
        col1.metric(" ",f"{r2:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 인구 수 변화가 자동차 등록 증감에 미치는 영향")
            st.pyplot(fig)
        
    # ------------------
    # 로지스틱
    # ------------------
    with tab3:
        st.markdown("### 🧠 로지스틱 회귀 분석")

        fig_cm, fig_prob, acc, coef = run_logistic(df, selected_district)

        st.metric("모델 정확도", f"{acc:.2%}")

        st.markdown("#### 📐 회귀 계수")
        st.write(f"인구 변화 계수: **{coef:.4f}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔍 혼동 행렬")
            st.pyplot(fig_cm)
        with col2:
            st.markdown("#### 📈 자동차 등록 증가 확률 곡선")
            st.pyplot(fig_prob)

elif menu == "🅿️ 주차면 분석":
    st.header("🅿️ 자동차 수 vs 주차면 분석")
    df = load_data_parking()
    tab1, tab2 = st.tabs([
        "📊 기초 분석 및 예측",
        "📈 정규화 회귀 (Ridge)"
    ])
    with tab1:
        fig_corr, r, p = plot_correlation(df)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("상관 분석")
            st.pyplot(fig_corr)
        with col2:
            fig_reg, model, metrics = run_parking_regression(df)
            st.subheader("선형 회귀 분석")
            st.pyplot(fig_reg)        

        st.metric("Train R²", f"{metrics['train_r2']:.3f}")
        st.metric("Test R²", f"{metrics['test_r2']:.3f}")
        st.metric("MAE", f"{metrics['mae']:.1f}")
        st.metric("RMSE", f"{metrics['rmse']:.1f}")

        # 미래 예측
        pred = predict_future(df)
        st.subheader(f"📈 {pred['year']}년 예측")
        st.write(f"예상 자동차 수: {pred['pred_car']:,}")
        st.write(f"예상 주차면 수: {pred['pred_parking']:,}")
        st.write(f"예상 주차 확보율: {pred['parking_ratio']:.2f}%")

    with tab2:
        st.markdown("### 🧩 Ridge 회귀 (규제 강도 분석)")
        st.caption("과적합을 줄이기 위한 정규화(Regularization) 효과 확인")

        fig_ridge, best_scores = run_ridge(df)

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(fig_ridge)

        st.subheader("📌 최적 규제 강도 결과")
        st.metric("Best alpha", best_scores["best_alpha"])
        st.metric("Train R²", f"{best_scores['train_r2']:.3f}")
        st.metric("Test R²", f"{best_scores['test_r2']:.3f}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 다항 회귀 (비선형 관계 확인)")
            fig_poly, poly_model = run_parking_poly_regression(df, degree=2)
            st.pyplot(fig_poly)
        
