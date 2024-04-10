import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def speed_of_sound(temperature, altitude):
    return 331.5 * np.sqrt(1 + (temperature / 273.15)) * np.sqrt(altitude / 1.83)


# Создаем папки train и test, если их еще не существует
if not os.path.exists('train'):
    os.makedirs('train')
if not os.path.exists('test'):
    os.makedirs('test')

temperature = np.random.uniform(low=15, high=35, size=1000)
altitude = np.random.uniform(low=0, high=300, size=1000)

# Набор данных без шумов
speed = speed_of_sound(temperature, altitude)
df = pd.DataFrame(
    data=np.array([temperature, altitude, speed]).T,
    columns=['temperature', 'altitude', 'speed']
)

train, test = train_test_split(df, test_size=0.3)
train.to_csv('train/data1.csv', index=False)
test.to_csv('test/data1.csv', index=False)

# Добавим шум к скорости звука
speed = speed + np.random.normal(0, 10)
df = pd.DataFrame(
    data=np.array([temperature, altitude, speed]).T,
    columns=['temperature', 'altitude', 'speed']
)

train, test = train_test_split(df, test_size=0.3)
train.to_csv('train/data2.csv', index=False)
test.to_csv('test/data2.csv', index=False)

# Добавим шум к температуре
temperature = temperature + np.random.normal(0, 2)
df = pd.DataFrame(
    data=np.array([temperature, altitude, speed]).T,
    columns=['temperature', 'altitude', 'speed']
)

train, test = train_test_split(df, test_size=0.3)
train.to_csv('train/data3.csv', index=False)
test.to_csv('test/data3.csv', index=False)
