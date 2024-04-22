import pandas as pd
from joblib import load
from sklearn.metrics import r2_score


def test_model(dataset):
    # Загружаем ранее сохраненную модель
    model = load(f'models/model{dataset}.joblib')

    # Загружаем тестовый набор данных
    test = pd.read_csv(f'test/data{dataset}.csv')
    X = test.drop('time', axis=1)
    y = test['time']

    # Получаем предсказание модели на тестовых данных
    y_predict = model.predict(X)

    # Вычисляем значение метрики
    r2 = r2_score(y, y_predict)

    print(f'Dataset {dataset}: R2 = {r2:.3f}')


if __name__ == '__main__':
    for i in range(1, 4):
        test_model(i)
