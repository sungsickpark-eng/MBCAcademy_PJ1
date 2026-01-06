# analysis/population_car/cluster.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.font_manager as fm


# ------------------
# 한글 폰트 설정
# ------------------
def set_korean_font():
    try:
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc("font", family=font_name)
        plt.rcParams["axes.unicode_minus"] = False
    except:
        pass


@st.cache_data
def run_clustering(df, n_clusters=3):
    set_korean_font()

    # ------------------
    # 자치구 단위 집계
    # ------------------
    df_cluster = (
        df.groupby("district_id")
          .agg({
              "population": "mean",
              "car_count": "mean"
          })
          .reset_index()
    )

    # ------------------
    # 표준화 + KMeans
    # ------------------
    X = df_cluster[["population", "car_count"]]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    df_cluster["cluster"] = kmeans.fit_predict(X_scaled)

    # ------------------
    # 군집 요약
    # ------------------
    summary_df = (
        df_cluster
        .groupby("cluster")[["population", "car_count"]]
        .mean()
        .round(1)
    )

    # ------------------
    # Figure 1: 군집별 평균
    # ------------------
    fig_bar, ax_bar = plt.subplots(figsize=(5, 3.5))

    summary_df.plot(
        kind="bar",
        ax=ax_bar
    )

    ax_bar.set_title("군집별 평균 생활인구 및 자동차 등록 대수")
    ax_bar.set_xlabel("군집")
    ax_bar.set_ylabel("평균 값")
    ax_bar.legend(["생활인구", "자동차 등록 대수"])

    # ------------------
    # Figure 2: 군집 분포
    # ------------------
    fig_scatter, ax_scatter = plt.subplots(figsize=(5, 3.5))

    ax_scatter.scatter(
        df_cluster["population"],
        df_cluster["car_count"],
        c=df_cluster["cluster"],
        cmap="tab10",
        s=60,
        alpha=0.8
    )

    ax_scatter.set_xlabel("평균 생활인구")
    ax_scatter.set_ylabel("평균 자동차 등록 대수")
    ax_scatter.set_title("자치구별 군집 분포")

    return df_cluster, summary_df, fig_bar, fig_scatter
