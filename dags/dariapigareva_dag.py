import os
import datetime
import pendulum
import logging
import glob
import pandas as pd
import cianparser
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator

# Конфигурация путей
file_dir = "/home/admins/airflow/dags"
log_path = os.path.join(file_dir, "..", "logs", "dariapigareva_processing.log")
raw_data_path = os.path.join(file_dir, "..", "data", "raw")
model_path = os.path.join(file_dir, "..", "models", "dariapigareva_model.pkl")

os.makedirs(os.path.dirname(log_path), exist_ok=True)
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(os.path.dirname(model_path), exist_ok=True)

logging.basicConfig(filename=log_path, level=logging.INFO)
logger = logging.getLogger(__name__)

@dag(
    dag_id="cian_pipeline_dariapigareva",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2025, 6, 2, tz="UTC"),
    catchup=False,
    dagrun_timeout=datetime.timedelta(minutes=120),
)
def CianDataPipeline():

    # Создание таблицы с вашим именем
    create_table = SQLExecuteQueryOperator(
        task_id="create_dariapigareva_table",
        conn_id="tutorial_pg_conn",
        sql=f"""
        CREATE TABLE IF NOT EXISTS dariapigareva_flats (
            url_id TEXT PRIMARY KEY,
            total_meters NUMERIC,
            price NUMERIC,
            floor INTEGER,
            floors_count INTEGER,
            first_floor BOOLEAN,
            last_floor BOOLEAN,
            rooms_1 BOOLEAN,
            rooms_2 BOOLEAN,
            rooms_3 BOOLEAN
        );
        """,
    )

    @task()
    def parse_cian_data(room_numbers=3, pages=5):
        """Парсинг данных с CIAN для 1-3 комнатных квартир"""
        logger.info("Starting CIAN parsing...")
        parser = cianparser.CianParser(location="Москва")
        
        for n_rooms in range(1, room_numbers + 1):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            csv_path = os.path.join(raw_data_path, f"{n_rooms}_{timestamp}.csv")
            
            try:
                data = parser.get_flats(
                    deal_type="sale",
                    rooms=(n_rooms,),
                    additional_settings={
                        "start_page": 1,
                        "end_page": pages,
                        "object_type": "secondary",
                    },
                )
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Saved {len(df)} records to {csv_path}")
                else:
                    logger.warning(f"No data found for {n_rooms} rooms")
                    
            except Exception as e:
                logger.error(f"Parsing failed for {n_rooms} rooms: {str(e)}")
                raise

    @task()
    def process_data():
        """Обработка и объединение данных"""
        logger.info("Processing raw data...")
        csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
        
        if not csv_files:
            logger.error("No CSV files found for processing")
            raise FileNotFoundError("No raw data files")
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {str(e)}")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Feature Engineering
        combined_df = combined_df[[
            'total_meters', 'price', 'floor', 'floors_count', 'rooms_count'
        ]].dropna()
        
        combined_df['first_floor'] = (combined_df['floor'] == 1).astype(int)
        combined_df['last_floor'] = (combined_df['floor'] == combined_df['floors_count']).astype(int)
        
        # One-Hot Encoding для комнат
        for n in [1, 2, 3]:
            combined_df[f'rooms_{n}'] = (combined_df['rooms_count'] == n).astype(int)
        
        return combined_df.to_dict(orient='records')

    @task()
    def load_to_postgres(records):
        """Загрузка данных в PostgreSQL"""
        logger.info("Loading data to PostgreSQL...")
        postgres_hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
        conn = postgres_hook.get_conn()
        
        insert_sql = """
        INSERT INTO dariapigareva_flats VALUES (
            %(url_id)s,
            %(total_meters)s,
            %(price)s,
            %(floor)s,
            %(floors_count)s,
            %(first_floor)s,
            %(last_floor)s,
            %(rooms_1)s,
            %(rooms_2)s,
            %(rooms_3)s
        ) ON CONFLICT (url_id) DO NOTHING
        """
        
        with conn.cursor() as cur:
            for record in records:
                try:
                    # Генерация уникального ID
                    record['url_id'] = f"{record['floor']}-{record['total_meters']}-{record['price']}"
                    cur.execute(insert_sql, record)
                except Exception as e:
                    logger.error(f"Error inserting record: {str(e)}")
            conn.commit()

    @task()
    def train_ml_model():
        """Обучение модели машинного обучения"""
        logger.info("Training ML model...")
        
        # Загрузка данных из БД
        postgres_hook = PostgresHook(postgres_conn_id="tutorial_pg_conn")
        df = postgres_hook.get_pandas_df(
            sql="SELECT total_meters, floors_count, first_floor, last_floor, rooms_1, rooms_2, rooms_3, price FROM dariapigareva_flats"
        )
        
        if df.empty:
            raise ValueError("No data available for training")
        
        X = df.drop('price', axis=1)
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X_train, y_train)
        
        # Сохранение модели
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

    # Определение порядка задач
    create_table >> parse_cian_data() >> process_data() >> load_to_postgres() >> train_ml_model()

dag = CianDataPipeline()
