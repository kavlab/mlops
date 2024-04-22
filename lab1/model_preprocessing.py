import pandas as pd
from sklearn.preprocessing import StandardScaler


# Стандартизация
def scale(scaler, data_type, dataset):
    filename = f'{data_type}/data{dataset}.csv'

    train = pd.read_csv(filename)
    X = train.drop('time', axis=1)
    y = train['time']

    if data_type == 'train':
        func = scaler.fit_transform
    else:
        func = scaler.transform

    result = pd.DataFrame(
        func(X),
        columns=X.columns
    )
    result['time'] = y

    result.to_csv(filename, index=False)


if __name__ == '__main__':
    for i in range(1, 4):
        scaler = StandardScaler()

        scale(scaler, 'train', i)
        scale(scaler, 'test', i)
