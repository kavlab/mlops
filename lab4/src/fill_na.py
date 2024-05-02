import pandas as pd


DATASET = '../datasets/house_prices_train.csv'


def get_num_columns() -> list[str]:
    """Функция возвращает числовые колонки
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


def main():
    df = pd.read_csv(DATASET)

    num_columns = get_num_columns()
    values = dict([(column, 0) for column in num_columns])
    df = df.fillna(value=values)

    df.to_csv(DATASET, index=False)


if __name__ == "__main__":
    main()
