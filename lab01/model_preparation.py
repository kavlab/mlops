import os

import pandas as pd
from joblib import dump
from sklearn.linear_model import LinearRegression


def train_model(dataset):
    # Создаем модель линейной регрессии
    model = LinearRegression()

    train = pd.read_csv(f'train/data{dataset}.csv')
    X = train.drop('time', axis=1)
    y = train['time']

    # Обучаем модель
    model.fit(X, y)

    if not os.path.exists('models'):
        os.makedirs('models')

    # Сохраняем обученную модель
    dump(model, f'models/model{dataset}.joblib')


if __name__ == '__main__':
    for i in range(1, 4):
        train_model(i)
