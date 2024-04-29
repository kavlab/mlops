import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib


RANDOM_STATE = 82
TEST_SIZE = 0.2
DATA_DIR = "data"
MODEL_DIR = "model"


def main():
    # загружаем данные
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    target = train[["SalePrice"]]
    train.drop(["SalePrice"], axis=1, inplace=True)

    # выделяем в тренировочном наборе валидационную выборку
    X_train, X_val, y_train, y_val = train_test_split(
        train, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # создаем и обучаем модель
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.08,
        gamma=0,
        subsample=0.75,
        colsample_bytree=1,
        max_depth=7,
    )
    model.fit(
        train,
        target,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=0,
    )

    # сохраняем модель в файл
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, f"{MODEL_DIR}/xgb_model.joblib")


if __name__ == "__main__":
    main()
