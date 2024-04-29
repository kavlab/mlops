import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_URL = (
    "https://raw.githubusercontent.com/kavlab/mfoml/main/house_prices_train.csv"
)
RANDOM_STATE = 82
TEST_SIZE = 0.2
DATA_DIR = "data"
MODEL_DIR = "model"


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_URL, index_col="Id")


def get_num_columns() -> list[str]:
    """Функция возвращает числовые колонки
    """
    num_columns = [
        'MSSubClass',
        'LotFrontage',
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'MasVnrArea',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'GarageYrBlt',
        'GarageCars',
        'GarageArea',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'MoSold',
        'YrSold',
        'SalePrice'
    ]
    return num_columns


def main():
    # загружаем набор данных
    df = load_dataset()

    # получаем список числовых признаков
    num_columns = get_num_columns()
    target = "SalePrice"
    scaled_columns = [column for column in num_columns if column != target]

    # выполняем стандартизацию, целевую переменную не трогаем
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='median')
    transformer = ColumnTransformer(
        [
            ("pipeline", make_pipeline(imputer, scaler), scaled_columns),
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
    joblib.dump(transformer.transformers_[0][1], f"{MODEL_DIR}/pipeline.joblib")


if __name__ == "__main__":
    main()
