import pandas as pd
from sqlalchemy import create_engine, text

# -------------------------
# 1. DB 연결
# -------------------------
engine = create_engine(
    "mysql+pymysql://root:12341234@localhost:3306/miniproject2",
    echo=True
)

# -------------------------
# 2. CSV → 테이블 적재
# -------------------------
def load_csv_to_db(csv_path, table_name):
    df = pd.read_csv(csv_path)
    df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False
    )
    print(f"✅ {table_name} 테이블 생성 완료")

# -------------------------
# 3. 메인 실행
# -------------------------
if __name__ == "__main__":
    print("🚀 DB 초기화 시작")

    load_csv_to_db("./data/district.csv", "district")
    load_csv_to_db("./data/population.csv", "population")
    load_csv_to_db("./data/car.csv", "car")
    load_csv_to_db("./data/public_transit.csv", "public_transit")
    load_csv_to_db("./data/parking_car.csv", "parking_car")
    load_csv_to_db("./data/vehicle.csv", "vehicle")
    load_csv_to_db("./data/traffic.csv", "traffic")

    # -------------------------
    # 4. View 생성
    # -------------------------
    with engine.connect() as conn:
        conn.execute(text("DROP VIEW IF EXISTS ml_base_view"))

        conn.execute(text("""
        CREATE VIEW ml_base_view AS
        SELECT
            p.district,
            p.datetime,
            p.population,
            p.population_diff,
            c.car_count,
            c.car_diff
        FROM population p
        JOIN car c
        ON p.district = c.district
        AND p.datetime = c.datetime;

        """))

        conn.commit()

    print("🎉 DB 초기화 및 View 생성 완료")
