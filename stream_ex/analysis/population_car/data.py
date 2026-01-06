# analysis/population_car/data.py

import pandas as pd
from sqlalchemy import create_engine
import streamlit as st


@st.cache_data
def load_data():
    engine = create_engine(
        "mysql+pymysql://root:12341234@localhost:3306/miniproject"
    )

    df = pd.read_sql(
        "SELECT * FROM seoul_analysis_view",
        engine
    )

    return df
