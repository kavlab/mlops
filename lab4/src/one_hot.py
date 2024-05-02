import pandas as pd

DATASET = '../datasets/house_prices_train.csv'


def main():
    df = pd.read_csv(DATASET)

    # OneHot encoding
    df = pd.get_dummies(df, columns=['MSZoning'])

    df.to_csv(DATASET, index=False)


if __name__ == "__main__":
    main()
