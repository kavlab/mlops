#!/bin/bash

# Установка зависимостей
pip install -r ./requirements.txt

# генерация данных
python3 ./data_creation.py

# предобработка данных
python3 ./model_preprocessing.py

# обучение модели
python3 ./model_preparation.py

# тестирование модели
python3 ./model_testing.py
