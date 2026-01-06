# analysis/population_car/regression.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import streamlit as st


@st.cache_data
def run_regression(df):
    # ------------------
    # 변수 선택
    # ------------------
    X = df[["population_diff"]]
    y = df["car_diff"]

    # ------------------
    # 기초 통계
    # ------------------
    desc = df[["population_diff", "car_diff"]].describe()
    corr = df[["population_diff", "car_diff"]].corr()

    # ------------------
    # 선형 회귀
    # ------------------
    model = LinearRegression()
    model.fit(X, y)

    coef = model.coef_[0]
    intercept = model.intercept_

    coef_df = pd.DataFrame({
        "Coefficient": [coef],
        "Intercept": [intercept]
    }, index=["population_diff"])

    # ------------------
    # 표준화 회귀
    # ------------------
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    model.fit(X_scaled, y_scaled)
    std_coef = model.coef_[0][0]

    std_coef_df = pd.DataFrame({
        "Standardized Coefficient": [std_coef]
    }, index=["population_diff"])

    # ------------------
    # 시각화
    # ------------------
    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.scatter(X, y, alpha=0.4)
    ax.plot(X, model.predict(X), color="red")

    ax.set_xlabel("population_diff")
    ax.set_ylabel("car_diff")

    return fig, desc, corr, coef_df, std_coef_df
