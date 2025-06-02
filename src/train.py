"""
This is the train ml model script.
Linear regression with 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import os

import logging
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


MODEL_NAME = "decision_tree_reg_1.pkl"

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def train_model(model_path):
    """Train model and save with MODEL_NAME"""
    train_df = pd.read_csv("data/processed/train.csv")
    X = train_df[
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
    y = train_df["price"]
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X.values, y)

    logging.info(f"Train {model} and save to {model_path}")

    joblib.dump(model, model_path)


if __name__ == "__main__":
    """Parse arguments and train model"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    model_path = os.path.join("models", args.model)

    train_model(model_path)