import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/kavlab/mfoml/main/house_prices_train.csv"
)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_URL, index_col="Id")


def main():
    # загружаем набор данных
    df = load_dataset()

    df.to_csv('../datasets/house_prices_train.csv', index=False)


if __name__ == "__main__":
    main()
