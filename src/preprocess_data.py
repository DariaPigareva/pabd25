"""
This is preprocess data script. 
Save clean data to data/preprocessed/train.csv and test.csv. 
Use 5 features:
- total_meters
- floors_count
- first_floor
- last_floor
- n_rooms (One Hot Encoded)
"""

import argparse
import glob

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TEST_SIZE = 0.2

logging.basicConfig(
    filename="train.log",
    filemode="a",
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def preprocess_data(test_size):
    """
    Filter, sort and remove duplicates
    """
    raw_data_path = "./data/raw"
    file_list = glob.glob(raw_data_path + "/*.csv")
    logging.info(f"Preprocess_data. Use files to train: {file_list}")
    df = pd.read_csv(file_list[0])
    for i in range(1, len(file_list)):
        data = pd.read_csv(file_list[i])
        df_i = pd.DataFrame(data)
        df = pd.concat([df, df_i], axis=0)

    df["url_id"] = df["url"].map(lambda x: x.split("/")[-2])
    df = (
        df[["url_id", "total_meters", "floor", "floors_count", "rooms_count", "price"]]
        .set_index("url_id")
        .sort_index()
    )

    df.drop_duplicates(inplace=True)
    df = df[df["price"] < 100_000_000]
    df = df[df["total_meters"] < 100]

    df["rooms_1"] = df["rooms_count"] == 1
    df["rooms_2"] = df["rooms_count"] == 2
    df["rooms_3"] = df["rooms_count"] == 3
    df["first_floor"] = df["floor"] == 1
    df["last_floor"] = df["floor"] == df["floors_count"]
    df.drop(columns=["floor", "rooms_count"], inplace=True)

    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    logging.info(f"Preprocess_data. train_df: {len(train_df)} samples")
    train_head = "\n" + str(train_df.head())
    logging.info(train_head)
    logging.info(f"Preprocess_data. test_df: {len(test_df)} samples")
    test_head = "\n" + str(test_df.head())
    logging.info(test_head)

    train_df.to_csv("data/processed/train.csv")
    test_df.to_csv("data/processed/test.csv")


if __name__ == "__main__":
    """Parse arguments, preprocess data and save to train.csv and test.csv
    todo: add price and total meters arguments for the data filtration
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test size, from 0 to 0.5",
        default=TEST_SIZE,
    )
    args = parser.parse_args()
    test_size = float(args.split)
    assert 0.0 <= test_size <= 0.5

    preprocess_data(test_size)