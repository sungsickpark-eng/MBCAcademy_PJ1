import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

from analysis.traffic_car.data import load_data_traffic
from analysis.traffic_car.traffic import analyze_correlation, make_yearly_summary, plot_traffic_growth_bar
from analysis.traffic_car.vehicle import make_monthly_summary, plot_vehicle_trend
os.environ["OMP_NUM_THREADS"] = "1"

from analysis.parking_car.ridge import run_ridge
from analysis.parking_car.visual_parking import plot_correlation, predict_future, run_parking_regression
from analysis.parking_car.data import load_data_parking
from analysis.population_car.cluster import run_clustering
from analysis.population_car.regression import run_regression
from analysis.population_car.logistic import run_logistic
from analysis.population_car.data import load_data
from analysis.public_transit.data import load_data_transit
from analysis.public_transit.visual_transit import run_visual_transit
from analysis.public_transit.multireg import run_multireg

# 한글 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)

# 마이너스 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

st.cache_data.clear()  # 개발 중만


# CSS 로드 함수
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 1. 페이지 설정 (가장 먼저)
st.set_page_config(
    page_title="Streamlit 프로젝트",
    page_icon="🚀",
    layout="wide"
)

# CSS 적용
load_css("styles/style.css")

# 2. 사이드바 네비게이션
st.sidebar.title("📚 메뉴")

PAGES = {
    "🏠 Home": {
        "title": "🚀 교통 데이터 분석 프로젝트",
        "description": """
        본 프로젝트는 **서울시 교통 및 자동차 관련 데이터**를 기반으로 
        자동차 등록 현황과 증감 요인을 분석하고 향후 변화 가능성을 탐색하는 것을 목표로 합니다.
        
        ### 📌 주요 분석 내용
        - 교통 관련 CCTV와 사고 발생 건수 분석
        - 자동차 등록 대수와 교통량 증감의 상관관계
        - 대중교통 이용 변화와 승용차 등록 영향
        - 생활인구 변화에 따른 자동차 등록 추세
        - 렌트카 이용 패턴 분석
        """
    },

    "📘 서울시 자동차 등록 현황 및 증감 예측": {
        "title": "📘 서울시 자동차 등록 현황 및 증감 예측",
        "description": "전체 분석 흐름과 각 소주제의 연결 관계를 설명합니다."
    },

    "① 교통 관련 CCTV 갯수 / 설치된 CCTV 지역의 사고건수 분석": {
        "title": "① 교통 관련 CCTV 갯수 및 사고건수 분석",
        "author": "공통"
    },

    "② 자동차 등록 대수와 교통량 증감의 상관 (박성식)": {
        "title": "② 자동차 등록 대수와 교통량 증감의 상관관계",
        "author": "박성식"
    },

    "③ 대중교통 이용량 변화가 승용차 등록 대수에 미치는 영향 (이화섭)": {
        "title": "③ 버스 이용량 변화와 승용차 등록 대수",
        "author": "이화섭"
    },

    "④ 서울시 인구 수 변화 기반 자동차 등록 증가 여부 (이동훈)": {
        "title": "④ 서울시 자치구 인구 변화 기반 자동차 증감 분석",
        "author": "이동훈"
    },

    "⑤ 서울시 자동차 수와 주차면적의 증가 여부 (서경환)": {
        "title": "⑤ 자동차 등록댓수와 주차면적의 증감비교",
        "author": "서경환"
    }
}

menu = st.sidebar.radio(
    " ",
    list(PAGES.keys()),
    label_visibility="collapsed"
)

# 3. 메인 화면 렌더링
page = PAGES[menu]

st.title(page["title"])
st.divider()

if "author" in page:
    st.caption(f"👤 담당자: {page['author']}")

if "description" in page:
    st.markdown(page["description"])

# 4. 소주제별 콘텐츠 영역 (예시)
if menu.startswith("①"):
    st.write("📊 교통 관련 CCTV 갯수 / 설치된 CCTV 지역의 사고건수 분석")

elif menu.startswith("②"):
    st.header("📈 자동차 등록대수와 교통량 증감의 상관 및 예측 분석")

    # ===============================
    # 데이터 로딩
    # ===============================
    df, df_traffic = load_data_traffic()
    total_summary = make_monthly_summary(df)

    # ===============================
    # TAB 구성
    # ===============================
    tab1, tab2, tab3 = st.tabs([
        "📊 차종별 및 전체 자동차 등록 추이",
        "📊 교통량 증감 시각화",
        "📈 교통량 vs 등록대수 관계 분석"
    ])

    # ===============================
    # TAB 1
    # ===============================
    with tab1:
        st.subheader("📊 차종별 및 전체 자동차 등록 추이")

        fig_trend_all = plot_vehicle_trend(total_summary)
        st.pyplot(fig_trend_all, use_container_width=True)

        st.subheader("📋 연도별 자동차 등록 요약")
        yearly_df = make_yearly_summary(total_summary)
        st.dataframe(yearly_df, use_container_width=True)

    # ===============================
    # TAB 2
    # ===============================
    with tab2:
        st.subheader("📊 연도별 교통량 증감률 비교")

        fig_bar = plot_traffic_growth_bar(df_traffic)
        st.pyplot(fig_bar, use_container_width=False)

    # ===============================
    # TAB 3
    # ===============================
    with tab3:
        st.subheader("📈 교통량 증가와 자동차 등록 증가의 관계")

        corr, fig_trend, fig_scatter = analyze_correlation(
            total_summary,
            df_traffic
        )

        st.metric("상관계수", f"{corr:.3f}")

        st.pyplot(fig_trend, use_container_width=False)
        st.pyplot(fig_scatter, use_container_width=False)

elif menu.startswith("③"):
    st.header("🚌 자동차 증감에 따른 대중교통 이용횟수의 변화")
    
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
            st.pyplot(fig_bus_car, use_container_width=True)
        with col2:
            st.pyplot(fig_bus_sub, use_container_width=True)

    with tab2:
        st.header("📈 다항 회귀 및 Ridge 회귀 분석")
        st.caption("과적합 여부와 규제 강도(α)에 따른 성능 변화를 비교합니다.")

        base_df, ridge_df, degree_df, best_alpha = run_multireg(df)

        st.subheader("① 다항 회귀 성능 비교 (과적합 확인)")
        st.dataframe(base_df, use_container_width=True)

        st.subheader("② Ridge 회귀 α 튜닝 결과")
        st.dataframe(ridge_df, use_container_width=True)

        st.success(f"✅ Best alpha (Test R² 기준): **{best_alpha}**")

        st.subheader("③ 차수별 모델 성능 비교")
        st.dataframe(degree_df, use_container_width=True)
    with tab3:
        st.subheader("📉 Ridge 회귀 계수 비교 (α = 100, 표준화)")
        st.image("images/transit_ridge.png", width=700)


elif menu.startswith("④"):
    st.header("🏙 서울시 자치구 인구 변화 기반 자동차 증감 분석")

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

            st.pyplot(fig_bar, use_container_width=False)
            
            st.pyplot(fig_scatter, use_container_width=False)

            st.markdown("#### 📝 해석 요약")

            st.info("""
            - 인구 수와 자동차 등록 대수를 기준으로 서울 자치구는 **3개 군집**으로 분류됨  
            - 특정 군집은 인구 수 대비 자동차 등록 비율이 높게 나타남  
            - 교통 정책 수립 시 **군집별 차별화 전략** 필요
            """)
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

        st.markdown("#### 인구 수 변화가 자동차 등록 증감에 미치는 영향")
        st.pyplot(fig, use_container_width=False)
        
    # ------------------
    # 로지스틱
    # ------------------
    with tab3:
        st.markdown("### 🧠 로지스틱 회귀 분석")

        fig_cm, fig_prob, acc, coef = run_logistic(df, selected_district)

        st.metric("모델 정확도", f"{acc:.2%}")

        st.markdown("#### 📐 회귀 계수")
        st.write(f"인구 변화 계수: **{coef:.4f}**")

        st.markdown("#### 🔍 혼동 행렬")
        st.pyplot(fig_cm, use_container_width=False)

        st.markdown("#### 📈 자동차 등록 증가 확률 곡선")
        st.pyplot(fig_prob, use_container_width=False)


elif menu.startswith("⑤"):
    df = load_data_parking()
    
    st.header("🚗 자동차 수 vs 주차면 수 분석")

    tab1, tab2 = st.tabs([
        "📊 기초 분석 및 예측",
        "📈 정규화 회귀 (Ridge)"
    ])
    with tab1:
        # 상관분석
        fig_corr, r, p = plot_correlation(df)
        st.subheader("상관 분석")
        st.pyplot(fig_corr, use_container_width=False)
        st.write(f"Pearson r = {r:.3f}, p-value = {p:.4f}")

        # 회귀 분석
        fig_reg, model, metrics = run_parking_regression(df)
        st.subheader("선형 회귀 분석")
        st.pyplot(fig_reg, use_container_width=False)

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

        st.pyplot(fig_ridge, use_container_width=False)

        st.subheader("📌 최적 규제 강도 결과")
        st.metric("Best alpha", best_scores["best_alpha"])
        st.metric("Train R²", f"{best_scores['train_r2']:.3f}")
        st.metric("Test R²", f"{best_scores['test_r2']:.3f}")
        

    