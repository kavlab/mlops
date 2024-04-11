#!/bin/bash

# генерация данных
python ./data_creation.py

# предобработка данных
python ./model_preprocessing.py

# обучение модели
python ./model_preparation.py

# тестирование модели
python ./model_testing.py
