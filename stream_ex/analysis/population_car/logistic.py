# analysis/population_car/logistic.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import seaborn as sns
import matplotlib.font_manager as fm


# ------------------
# 한글 폰트 설정 (Streamlit 대응)
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
def run_logistic(df):
    set_korean_font()

    df = df.copy()
    df["car_increase"] = (df["car_diff"] > 0).astype(int)

    X = df[["population_diff"]]
    y = df["car_increase"]

    # ------------------
    # Train / Test
    # ------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------
    # 모델 학습
    # ------------------
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # ------------------
    # 성능 평가
    # ------------------
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    coef = model.coef_[0][0]

    # ------------------
    # Confusion Matrix (Heatmap)
    # ------------------
    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax_cm
    )

    ax_cm.set_xlabel("예측값")
    ax_cm.set_ylabel("실제값")
    ax_cm.set_title("자동차 등록 증가 여부 예측 (혼동 행렬)")

    # ------------------
    # 확률 곡선
    # ------------------
    pop_range = np.linspace(
        df["population_diff"].min(),
        df["population_diff"].max(),
        100
    ).reshape(-1, 1)

    prob = model.predict_proba(pop_range)[:, 1]

    fig_prob, ax_prob = plt.subplots(figsize=(4.5, 3.5))

    ax_prob.plot(pop_range, prob)
    ax_prob.set_xlabel("생활인구 변화량")
    ax_prob.set_ylabel("자동차 등록 증가 확률")
    ax_prob.set_title("생활인구 변화에 따른 자동차 등록 증가 확률")

    return fig_cm, fig_prob, accuracy, coef
