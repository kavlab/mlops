import os

import requests
import streamlit as st


API_URL = os.getenv('API_URL', 'http://mlops-lab3-api:8080')


def get_features() -> list[str]:
    """
    """
    num_columns = [
        'MSSubClass',
        'LotFrontage',
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'MasVnrArea',
        'BsmtFinSF1',
        'BsmtFinSF2',
        'BsmtUnfSF',
        'TotalBsmtSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'GarageYrBlt',
        'GarageCars',
        'GarageArea',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'MoSold',
        'YrSold',
    ]
    return num_columns


def model_testing():
    response = requests.get(f'{API_URL}/test')
    if response.status_code == 200:
        r2 = response.json()['r2']
        st.write(f'R2: {r2}')
    else:
        st.write('Ошибка при выполнении запроса к API')


def predict_price():
    data = {}
    for feature in get_features:
        data[feature] = st.session_state.get(feature)
    response = requests.post(f'{API_URL}/predict', json=data)
    if response.status_code == 200:
        price = response.json()['price']
        st.write(f'Полученный результат: {price}')
    else:
        st.write('Ошибка при выполнении запроса к API')


def ui():
    st.title('Lab 3')

    if st.button(label='Выполнить оценку модели'):
        model_testing()

    for feature in get_features:
        st.number_input(
            label=feature,
            value=0,
            key=feature,
        )

    if st.button(label='Вычислить цену'):
        predict_price()


if __name__ == '__main__':
    ui()
