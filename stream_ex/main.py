from analysis.population_car.cluster import run_clustering
from analysis.population_car.regression import run_regression
from analysis.population_car.logistic import run_logistic
from analysis.population_car.data import load_data
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# 한글 폰트 설정
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False


if os.path.exists(font_path):
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc("font", family=font_name)

# 마이너스 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False


# -------------------------
# CSS 로드 함수
# -------------------------
def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -------------------------
# 1. 페이지 설정 (가장 먼저!)
# -------------------------
st.set_page_config(
    page_title="Streamlit 프로젝트",
    page_icon="🚀",
    layout="wide"
)

# CSS 적용
load_css("styles/style.css")

# -------------------------
# 2. 사이드바 네비게이션
# -------------------------
st.sidebar.title("📚 메뉴")

PAGES = {
    "🏠 Home": {
        "title": "🚀 교통 데이터 분석 프로젝트",
        "description": """
        본 프로젝트는 **서울시 교통 및 자동차 관련 데이터**를 기반으로  
        자동차 등록 현황과 증감 요인을 분석하고  
        향후 변화 가능성을 탐색하는 것을 목표로 합니다.
        
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

    "④ 서울시 생활인구 변화 기반 자동차 등록 증가 여부 (이동훈)": {
        "title": "④ 서울시 자치구 인구 변화 기반 자동차 증감 분석",
        "author": "이동훈"
    },

    "⑤ 렌트카 이용 (서경환)": {
        "title": "⑤ 서울시 자치구 자동차 등록 현황별 택시 이용 횟수",
        "author": "서경환"
    }
}

menu = st.sidebar.radio(
    " ",
    list(PAGES.keys()),
    label_visibility="collapsed"
)

# -------------------------
# 3. 메인 화면 렌더링
# -------------------------
page = PAGES[menu]

st.title(page["title"])
st.divider()

if "author" in page:
    st.caption(f"👤 담당자: {page['author']}")

if "description" in page:
    st.markdown(page["description"])

# -------------------------
# 4. 소주제별 콘텐츠 영역 (예시)
# -------------------------
if menu.startswith("①"):
    st.write("📊 CCTV 수와 사고건수 데이터를 불러와 분석합니다.")

elif menu.startswith("②"):
    st.write("📈 자동차 등록 대수와 교통량 증감 간의 상관관계를 분석합니다.")

elif menu.startswith("③"):
    st.write("🚌 버스 이용량 변화가 승용차 등록 대수에 미치는 영향을 분석합니다.")

elif menu.startswith("④"):
    st.header("🏙 서울시 생활인구 변화 기반 자동차 등록 분석")
    st.caption("담당자: 이동훈")

    df = load_data()

    tab1, tab2, tab3 = st.tabs([
        "📊 군집 분석",
        "📈 회귀 분석",
        "🧠 로지스틱 회귀"
    ])

    # ------------------
    # 군집
    # ------------------
    with tab1:
        st.markdown("### 📊 군집 분석 결과")

        df_cluster, summary_df, fig_bar, fig_scatter = run_clustering(df)

        st.subheader("📋 자치구별 군집 결과")
        st.dataframe(df_cluster)

        st.subheader("📊 군집 요약")
        st.dataframe(summary_df)

        st.pyplot(fig_bar, use_container_width=False)
        
        st.pyplot(fig_scatter, use_container_width=False)


        # ------------------
        # 4️⃣ 해석 요약
        # ------------------
        st.markdown("#### 📝 해석 요약")

        st.info("""
        - 생활인구와 자동차 등록 대수를 기준으로 서울 자치구는 **3개 군집**으로 분류됨  
        - 특정 군집은 생활인구 대비 자동차 등록 비율이 높게 나타남  
        - 교통 정책 수립 시 **군집별 차별화 전략** 필요
        """)

    # ------------------
    # 회귀
    # ------------------
    with tab2:
        st.markdown("### 📈 선형 회귀 분석")

        fig, desc, corr, coef_df, std_coef_df = run_regression(df)

        st.markdown("#### 📊 기초 통계")
        st.dataframe(desc)

        st.markdown("#### 🔗 상관계수")
        st.dataframe(corr)

        st.markdown("#### 📈 회귀 결과")
        st.dataframe(coef_df)

        st.markdown("#### 📐 표준화 회귀계수")
        st.dataframe(std_coef_df)

        st.markdown("#### 생활인구 변화가 자동차 등록 증감에 미치는 영향")
        st.pyplot(fig, use_container_width=False)

    # ------------------
    # 로지스틱
    # ------------------
    with tab3:
        st.markdown("### 🧠 로지스틱 회귀 분석")

        with st.expander("📌 분석 설명", expanded=False):
            st.write("""
            생활인구 변화량을 기준으로  
            자동차 등록이 **증가할 확률**을 예측합니다.
            """)

        fig_cm, fig_prob, acc, coef = run_logistic(df)

        st.metric("모델 정확도", f"{acc:.2%}")

        st.markdown("#### 📐 회귀 계수")
        st.write(f"생활인구 변화 계수: **{coef:.4f}**")

        st.markdown("#### 🔍 혼동 행렬")
        st.pyplot(fig_cm, use_container_width=False)

        st.markdown("#### 📈 자동차 등록 증가 확률 곡선")
        st.pyplot(fig_prob, use_container_width=False)


elif menu.startswith("⑤"):
    st.write("🚗 렌트카 이용 패턴 및 추세를 분석합니다.")
