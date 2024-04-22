import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import joblib


DATASET_URL = (
    "https://raw.githubusercontent.com/kavlab/mfoml/main/house_prices_train.csv"
)
RANDOM_STATE = 82
TEST_SIZE = 0.2
DATA_DIR = "data"
MODEL_DIR = "model"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_URL, index_col="Id")


def get_num_columns(df: pd.DataFrame) -> list[str]:
    num_columns = []

    # Делим колонки на числовые и категориальные по типу данных,
    #  исходя из предположения, что в колонках с типом object
    #  содержатся категориальные данные.
    for column in df.columns:
        if df[column].dtypes != object:
            num_columns.append(column)

    return num_columns


def main():
    # загружаем набор данных
    df = load_dataset()

    # получаем список числовых принаков
    num_columns = get_num_columns(df)
    target = "SalePrice"
    scaled_columns = [column for column in num_columns if column != target]

    # выполняем стандартизацию, целевую переменную не трогаем
    scaler = StandardScaler()
    transformer = ColumnTransformer(
        [
            ("scale", scaler, scaled_columns),
            ("target", "passthrough", [target]),
        ]
    )
    columns = np.hstack([scaled_columns, [target]])
    df_scaled = pd.DataFrame(transformer.fit_transform(df), columns=columns)
    df_scaled.fillna(0, inplace=True)

    # делим набор на тренировочную и тестовую выборки
    train, test = train_test_split(
        df_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # сохраняем результат в файлы
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    train.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test.to_csv(f"{DATA_DIR}/test.csv", index=False)

    # сохраняем ColumnTransformer в файл
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(transformer, f"{MODEL_DIR}/transformer.joblib")


if __name__ == "__main__":
    main()
