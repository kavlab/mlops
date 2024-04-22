import pandas as pd
from sklearn.metrics import r2_score
import joblib


DATA_DIR = "data"
MODEL_DIR = "model"


def main():
    # загружаем модель
    model = joblib.load(f'{MODEL_DIR}/xgb_model.joblib')

    # загружаем данные
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    y_test = test[["SalePrice"]]
    test.drop(["SalePrice"], axis=1, inplace=True)

    # оцениваем модель
    y_pred = model.predict(test)

    score = r2_score(y_test, y_pred)
    print(f"Коэффициент R2: {score}")


if __name__ == "__main__":
    main()
