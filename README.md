# Практические задания по MLOps

## Модуль 1

Скрипты первого задания находятся в каталоге [lab01](lab01)

### Этапы работы конвейера

1. Скрипт data_creation.py создает три набора данных.
В качестве целевой переменной выбрано время высыхания краски. А в качестве параметров температура и влажность воздуха. Написана функция моделирующая зависимость целевой переменной от параметров. Добавлен независимый параметр, представляющий из себя случайно сгенерированные данные.

    - В первом наборе данных добавлен небольшой шум к целевой переменной.
    - Во втором в дополнение к первому набору добавляется шум к температуре.
    - В третьем в дополнение ко второму набору добавляется шум к влажности и выбросы в значениях целевой переменной.

    Полученные наборы данных делятся на тренировочную и тестовую части и сохраняются в каталоги train и test.

2. Скрипт model_preprocessing.py выполняет стандартизацию всех параметров кроме целевой переменной для тренировочной и тестовой выборок.

3. Скрипт model_preparation.py обучает модель линейной регрессии (sklearn.linear_model.LinearRegression) на тренировочных наборах. И сохраняет обученные модели с помощью библиотеки joblib.

4. Скрипт model_testing.py загружает модели и проверяет их на тестовых наборах данных. В качестве метрики выбран коэффициент детерминации R2.

Bash скрипт pipeline.sh устанавливает зависимости из requirements.txt и последовательно выполняет все скрипты из пунктов 1-4.

В процессе тестирования моделей были получены следующие значения метрики:

```
Dataset 1: R2 = 0.971
Dataset 2: R2 = 0.948
Dataset 3: R2 = 0.870
```
