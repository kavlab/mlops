import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 82
SAMPLE_SIZE = 1000


# Функция для моделирования
def paint_drying_time(temperature, humidity):
    return 24 - 0.5 * temperature + 0.2 * humidity


# Создаем папки train и test, если их еще не существует
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

np.random.seed(RANDOM_STATE)

# Температура воздуха
temperature = np.random.uniform(low=15, high=35, size=SAMPLE_SIZE)
# Влажность воздуха
humidity = np.random.uniform(low=0, high=100, size=SAMPLE_SIZE)
# Независимый параметр
noise = 50 * np.random.randn(SAMPLE_SIZE) + 200
# Вычисляем время высыхания краски плюс шум
drying_time = paint_drying_time(temperature, humidity) + np.random.randn(SAMPLE_SIZE) + 2

# Набор данных с шумом только в целевой переменной
df = pd.DataFrame(
    data=np.array([temperature, humidity, noise, drying_time]).T,
    columns=['temperature', 'humidity', 'noise', 'time']
)

train, test = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
train.to_csv('train/data1.csv', index=False)
test.to_csv('test/data1.csv', index=False)

# Добавим шум к температуре
temperature = temperature + 2.2 * np.random.randn(SAMPLE_SIZE) + 5
df = pd.DataFrame(
    data=np.array([temperature, humidity, noise, drying_time]).T,
    columns=['temperature', 'humidity', 'noise', 'time']
)

train, test = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
train.to_csv('train/data2.csv', index=False)
test.to_csv('test/data2.csv', index=False)

# Добавим шум еще к влажности
humidity = humidity + 2 * np.random.randn(SAMPLE_SIZE) + 5
# Добавим выбросы
outliers = np.random.randint(low=100, high=SAMPLE_SIZE-100, size=5)
drying_time[outliers] = drying_time[outliers] * 2
drying_time[outliers-10] = drying_time[outliers-10] / 2

df = pd.DataFrame(
    data=np.array([temperature, humidity, noise, drying_time]).T,
    columns=['temperature', 'humidity', 'noise', 'time']
)

train, test = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
train.to_csv('train/data3.csv', index=False)
test.to_csv('test/data3.csv', index=False)
