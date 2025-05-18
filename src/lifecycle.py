"""
Модуль для полного цикла ML-модели линейной регрессии.

Особенности:
- Парсинг данных с CIAN
- Предобработка данных с переименованием столбцов
- Обучение модели с настройкой через аргументы
- Тестирование на новых данных с сортировкой по url_id
"""

import argparse
import datetime
import glob
import os
import numpy as np
import logging
import cianparser
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DEFAULT_TEST_SIZE = 0.2
DEFAULT_N_ROOMS = 1
DEFAULT_MODEL_NAME = "linear_regression_model.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def parse_cian(n_rooms: int = 1) -> None:
    """
    Парсинг данных с CIAN и сохранение в папку data/raw.

    Args:
        n_rooms (int): Количество комнат для парсинга (по умолчанию 1)
    """
    moscow_parser = cianparser.CianParser(location="Москва")
    os.makedirs("data/raw", exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    csv_path = f"data/raw/{n_rooms}_{timestamp}.csv"

    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 2,
            "object_type": "secondary",
        },
    )

    df = pd.DataFrame(data).rename(columns={
        'total_meters': 'area',
        'floors_count': 'total_floors',
        'rooms_count': 'rooms'
    })

    df.to_csv(csv_path, encoding="utf-8", index=False)
    logging.info("Данные успешно спарсены и сохранены в %s", csv_path)


def preprocess_data(test_size: float) -> None:
    """
    Предобработка данных и подготовка к обучению модели.

    Args:
        test_size (float): Доля тестовой выборки (от 0.0 до 1.0)
    """
    raw_files = glob.glob("data/raw/*.csv")
    if not raw_files:
        raise FileNotFoundError("Нет сырых данных для обработки")

    # Загрузка и объединение данных
    df = pd.concat(
        [pd.read_csv(f).rename(columns={
            'total_meters': 'area',
            'floors_count': 'total_floors',
            'rooms_count': 'rooms'
        }) for f in raw_files],
        ignore_index=True
    )

    # Сортировка по url_id для выделения новых данных в тест
    df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
    df.sort_values("url_id", inplace=True)

    # Фильтрация и признаки
    df = df[
        (df["price"] < 100_000_000) &
        (df["area"] < 100)
    ].drop_duplicates().copy()

    df["rooms_1"] = df["rooms"] == 1
    df["rooms_2"] = df["rooms"] == 2
    df["rooms_3"] = df["rooms"] == 3
    df["first_floor"] = df["floor"] == 1
    df["last_floor"] = df["floor"] == df["total_floors"]

    # Разделение данных с сохранением порядка (по url_id)
    test_size_abs = int(len(df) * test_size)
    train_df = df.iloc[:-test_size_abs] if test_size_abs > 0 else df
    test_df = df.iloc[-test_size_abs:] if test_size_abs > 0 else df.iloc[0:0]

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    logging.info("Данные предобработаны и сохранены.")


def train_model(model_path: str) -> None:
    """
    Обучение модели линейной регрессии и сохранение.

    Args:
        model_path (str): Путь для сохранения модели
    """
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[["area", "total_floors", "rooms_1", "rooms_2", "rooms_3", "first_floor", "last_floor"]]
    y = train_df["price"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    logging.info("Модель обучена и сохранена в %s", model_path)
    logging.info(f"Коэффициенты: {model.coef_}")


def test_model(model_path: str) -> None:
    """
    Тестирование модели на новых данных.

    Args:
        model_path (str): Путь к файлу модели
    """
    test_df = pd.read_csv("data/processed/test.csv")
    if test_df.empty:
        logging.warning("Тестовая выборка пуста!")
        return

    X_test = test_df[["area", "total_floors", "rooms_1", "rooms_2", "rooms_3", "first_floor", "last_floor"]]
    y_test = test_df["price"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
    print(f"Test MSE: {mse:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML pipeline for CIAN flats price prediction")
    parser.add_argument("-s", "--split", type=float, default=DEFAULT_TEST_SIZE, help="Доля тестовой выборки (0.0 - 0.5)")
    parser.add_argument("-n", "--n_rooms", type=int, default=DEFAULT_N_ROOMS, help="Количество комнат для парсинга")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_NAME, help="Имя файла для сохранения модели")
    parser.add_argument("-p", "--parse_data", action="store_true", help="Парсить новые данные")
    args = parser.parse_args()

    model_path = os.path.join("models", args.model)
    os.makedirs("models", exist_ok=True)

    if args.parse_data:
        parse_cian(args.n_rooms)

    preprocess_data(args.split)
    train_model(model_path)
    test_model(model_path)
