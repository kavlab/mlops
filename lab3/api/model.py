import pandas as pd
from sklearn.metrics import r2_score
import joblib
from api_model import InputData


DATA_DIR = "data"
MODEL_DIR = "model"


def r2() -> float:
    """Функция оценивает модель на тестовых данных
    """
    # загружаем модель
    model = joblib.load(f'{MODEL_DIR}/xgb_model.joblib')

    # загружаем данные
    test = pd.read_csv(f'{DATA_DIR}/test.csv')
    y_test = test[["SalePrice"]]
    test.drop(["SalePrice"], axis=1, inplace=True)

    # оцениваем модель
    y_pred = model.predict(test)

    # возвращаем R2
    score = r2_score(y_test, y_pred)
    return float(score)


def predict_price(data: InputData) -> int:
    """Функция определяет цену на основе переданных данных
    """
    # загружаем pipeline и модель
    pipeline = joblib.load(f'{MODEL_DIR}/pipeline.joblib')
    model = joblib.load(f'{MODEL_DIR}/xgb_model.joblib')

    df = pd.DataFrame([data.model_dump()])

    df_transformed = pipeline.transform(df)

    # предсказание цены на новых данных
    y_pred = model.predict(df_transformed)

    return int(y_pred)
