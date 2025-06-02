"""
This is offline test script for the ml model.
Linear regression with 5 features:
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

import cianparser
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TEST_SIZE = 0.2
N_ROOMS = 1  # just for the parsing step
MODEL_NAME = "decision_tree_reg_1.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def test_model(model_path):
    """Test model with new data"""
    test_df = pd.read_csv("data/processed/test.csv")
    train_df = pd.read_csv("data/processed/train.csv")
    X_test = test_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_test = test_df["price"]
    X_train = train_df[
        [
            "total_meters",
            "floors_count",
            "rooms_1",
            "rooms_2",
            "rooms_3",
            "first_floor",
            "last_floor",
        ]
    ]
    y_train = train_df["price"]
    model = joblib.load(model_path)
    # Предсказание на тестовой выборке
    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    logging.info(f"Test model. MSE: {mse:.2f}")
    logging.info(f"Test model. RMSE: {rmse:.2f}")
    logging.info(f"Test model. MAE: {mae:.2f}")
    logging.info(f"Test model. R2 train: {r2_train:.2f}")
    logging.info(f"Test model. R2 test: {r2_test:.2f}")


if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_rooms", help="Number of rooms to parse", type=int, default=N_ROOMS
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    model_path = os.path.join("models", args.model)

    test_model(model_path)