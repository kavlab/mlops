import pandas as pd
from sklearn.impute import SimpleImputer

DATASET = '../datasets/house_prices_train.csv'


def main():
    df = pd.read_csv(DATASET)

    cat_columns = []
    for column in df.columns:
        if df[column].dtypes == object:
            cat_columns.append(column)

    imputer = SimpleImputer(strategy='most_frequent')
    df.loc[:, cat_columns] = imputer.fit_transform(df[cat_columns])

    df.to_csv(DATASET, index=False)


if __name__ == "__main__":
    main()
