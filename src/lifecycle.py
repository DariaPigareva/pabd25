"""
This is full life cycle for ml model.
DecisionTreeRegressor with 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import datetime
import glob
import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



TEST_SIZE = 0.2
N_ROOMS = 1
MODEL_NAME = "decision_tree_reg_1.pkl"

# Настройка логирования в консоль + файл
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)

def parse_cian(n_rooms=1):
    """Парсинг данных с CIAN"""
    try:
        import cianparser
        moscow_parser = cianparser.CianParser(location="Москва")
        t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        os.makedirs("data/raw", exist_ok=True)
        csv_path = f"data/raw/{n_rooms}_{t}.csv"
        
        logging.info("Запуск парсинга новых данных...")
        
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 5,
                "object_type": "secondary",
                "delay": 2.0,  # Увеличьте задержку
            },
        )
        
        if not data:
            logging.error("Парсер вернул пустой результат!")
            return
            
        df = pd.DataFrame(data)
        if df.empty:
            logging.error("Нет данных для сохранения!")
            return
            
        df.to_csv(csv_path, index=False)
        logging.info(f"Сохранено {len(df)} записей в {csv_path}")
        
    except Exception as e:
        logging.error(f"Ошибка: {str(e)}", exc_info=True)  # Добавлен traceback
        raise


def preprocess_data(test_size):
    """Обработка данных"""
    try:
        raw_data_path = "data/raw"
        file_list = glob.glob(os.path.join(raw_data_path, "*.csv"))
        
        if not file_list:
            raise FileNotFoundError("Нет CSV-файлов для обработки в data/raw")
        
        logging.info(f"Обработка {len(file_list)} файлов...")
        df = pd.concat([pd.read_csv(f) for f in file_list], ignore_index=True)

        # Обработка данных
        df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
        df = df[
            ["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]
        ].set_index("url_id")
        
        # Фильтрация
        df = df[
            (df["price"] < 100_000_000) & 
            (df["total_meters"] < 100)
        ].drop_duplicates()
        
        # Feature engineering
        df["rooms_1"] = (df["rooms_count"] == 1).astype(int)
        df["rooms_2"] = (df["rooms_count"] == 2).astype(int)
        df["rooms_3"] = (df["rooms_count"] == 3).astype(int)
        df["first_floor"] = (df["floor"] == 1).astype(int)
        df["last_floor"] = (df["floor"] == df["floors_count"]).astype(int)
        df = df.drop(columns=["floor", "rooms_count"])
        
        # Разделение данных
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # Сохранение
        os.makedirs("data/processed", exist_ok=True)
        train_df.to_csv("data/processed/train.csv")
        test_df.to_csv("data/processed/test.csv")
        logging.info("Данные успешно обработаны и сохранены")
        
    except Exception as e:
        logging.error(f"Ошибка обработки данных: {str(e)}")
        raise

def train_model(model_path):
    """Обучение модели"""
    try:
        train_df = pd.read_csv("data/processed/train.csv")
        X = train_df.drop(columns=["price"])
        y = train_df["price"]
        
        model = DecisionTreeRegressor(max_depth=5)
        model.fit(X, y)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model, model_path)
        logging.info(f"Модель сохранена в {model_path}")
        
    except Exception as e:
        logging.error(f"Ошибка обучения: {str(e)}")
        raise

def test_model(model_path):
    """Тестирование модели"""
    try:
        test_df = pd.read_csv("data/processed/test.csv")
        X_test = test_df.drop(columns=["price"])
        y_test = test_df["price"]
        
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        # Метрики
        mse = mean_squared_error(y_test, y_pred)
        logging.info("\nРезультаты тестирования:")
        logging.info(f"MSE: {mse:.2f}")
        logging.info(f"RMSE: {np.sqrt(mse):.2f}")
        logging.info(f"MAE: {np.mean(np.abs(y_test - y_pred)):.2f}")
        logging.info(f"R2: {model.score(X_test, y_test):.2f}")
        
    except Exception as e:
        logging.error(f"Ошибка тестирования: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--parse", help="Спарсить новые данные", action="store_true")
    parser.add_argument("-s", "--split", type=float, default=TEST_SIZE, help="Размер тестовой выборки")
    parser.add_argument("-m", "--model", default=MODEL_NAME, help="Имя модели")
    args = parser.parse_args()
    
    try:
        parse_cian(N_ROOMS)  # ВСЕГДА парсить новые данные!
        preprocess_data(args.split)
        train_model(os.path.join("models", args.model))
        test_model(os.path.join("models", args.model))
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}")
        exit(1)
