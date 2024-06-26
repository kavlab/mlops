# Практические задания по MLOps

## Модуль 1

Скрипты первого задания находятся в каталоге [lab1](lab1)

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

## Модуль 2

Все материалы второго задания находятся в каталоге [lab2](lab2)

### Скрипты Python

Работа конвейера обеспечивается тремя скриптами Python:
- data_preprocessing.py
- model_preparation.py
- model_testing.py

`data_preprocessing.py` выполняет загрузку датасета из моего репозитория на GitHub (набор был взят из [соревнования Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/) и загружен на GitHub для удобства скачивания).

Датасет представляет из себя набор параметров (80 числовых и категориальных признаков) домов с ценой, которая взята в качестве целевой переменной.

После загрузки скрипт производит обработку числовых признаков (стандартизацию), делит на тренировочный и тестовый наборы и сохраняет их в отдельные файлы. ColumnTransformer, который использовался для обработки, тоже сохраняется в файл.

`model_preparation.py` загружает предобработанные данные, создает и обучает модель XGBRegressor из пакета xgboost. Загруженный набор дополнительно делится на тренировочный и валидационный. Обученная модель сохраняется в файл.

`model_testing.py` загружает модель и тестовый набор данных. Далее выполняется предсказание и вычисляется значение метрики. В качестве метрики был выбран коэффициент детерминации R2. Значение R2 выводится в консоль.

### Dockerfile

Установка Jenkins производилась в контейнер Docker. Для этого был написан Dockerfile. 
Собираемый образ основан на последней версии образа jenkins/jenkins:lts. Кроме этого произодится установка python и модулей pip и venv.

### Jenkinsfile

В файле `Jenkinsfile` находится скрипт пайплайна Jenkins.

На этапе "Подготовка" выполняются создание виртуального окружения и установка зависимостей.

На следующих этапах последовательно запускаются скрипты Python, которые загружают и обрабатывают данные, обучают и производят оценку модели.

## Модуль 3

Все материалы третьего задания находятся в каталоге [lab3](lab3)

В подкаталоге `api` расположены файлы сервиса реализующего API для предсказания цены дома.
В этом же каталоге расположен файл `Dockerfile` для построения образа микросервиса с API.

В подкаталоге `ui` расположены файлы web-интерфейса выполненного с помощью Streamlit.
`Dockerfile` из этого каталога создает образ микросервиса с web-интерфейсом.

В корне каталога находится файл `docker-compose.yaml` для оркестрации двух микросервисов.

Для развертывания сервиса достаточно выполнения следующей команды на сервере, где установлен Docker:
```
docker compose up --build
```

## Модуль 4

Все материалы четвертого задания находятся в каталоге [lab4](lab4)

В каталоге `src` находятся скрипты для загрузки и обработки данных:
- `data_loader.py` - загружает датасет и сохраняет файл `house_prices_train.csv` в каталоге `datasets`;
- `fill_na.py` - читает датасет из файла, заменяет пропущенные значения в числовых признаках на средние значения и сохраняет результаты в тот же файл;
- `fill_na_categorical.py` - читает датасет из файла, заменяет пропущенные значения в категориальных признаках на чаще всего встречающиеся значения, после этого сохраняет результаты в тот же файл;
- `one_hot.py` - читает датасет из файла, выполняет One Hot кодирование признака `MSZoning`, сохраняет результаты в тот же файл.

После загрузки датасета для обработки сначала выполнялся скрипт `fill_na.py`. В первой версии этого скрипта пропущенные значения заменялись нулями (коммит a2bf99a).

После этого была произведена замена в категориальных признаках (коммит 9bcceb2).

Скрипт `fill_na.py` был изменен для замены на средние значения.
Для того, чтобы вернуть первоначальное состояние датасета, был использован DVC.
Для этого использовались следующие команды:
```
git checkout cfd510e
dvc pull
```

Возврат к последней версии скрипта `fill_na.py` и запуск скриптов по заполнению пропущенных значений:
```
git checkout lab4
python3 fill_na.py
python3 fill_na_categorical.py
```

Далее коммиты git c6425fd, c3f5527 и коммит и push dvc:
```
git commit -m "..."
dvc push -r gdrive
```

В качестве удаленного хранилища файлов использовался Google Drive. Ссылка на каталог с датасетами: [datasets](https://drive.google.com/drive/folders/1XVegteuM7M4zqKpqlCu-MwUTsKV2eyP_?usp=sharing).

Коммиты по этому заданию:

```
9222816 (HEAD -> lab4, origin/lab4) one hot encoding MSZoning
63300ad add one hot encoding script
c3f5527 fill NA in categorical columns with most frequently values
c6425fd fill NA in number columns with median
cfdcf25 fix fill_na script
9bcceb2 fill NA in categorical columns
2e5a01d add fill NA in categorical features
a2bf99a fill NA in number columns
962c55b add fill na script
cfd510e add gdrive
5a439d7 put dataset under control
867987d add data loader script
93f5ddd add dvc
```

Файл [datasets.dvc](lab4/datasets.dvc) для отслеживания версий находится в каталоге lab4.

## Модуль 5

Все материалы пятого задания находятся в каталоге [lab5](lab5)

Файл [test.ipynb](lab5/test.ipynb) - ноутбук с заданием.

При выполнении кода в ячейках формируются 3 датасета. Первый основной (с "правильными" данными), на нем обучается модель линейной регрессии.
Второй и третий с шумами. Во втором небольшие шумы, в третьем еще добавлены выбросы, которые должны значительно ухудшить качество предсказания модели.

Тестирование производится с помощью ipytest. Для этого было написано две функции test_dataset2 и test_dataset3.
Данные считаются качественными, если значение метрики R2 >= 0.98.
