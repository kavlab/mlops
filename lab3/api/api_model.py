from pydantic import BaseModel


class InputData(BaseModel):
    MSSubClass: int
    LotFrontage: int
    LotArea: int
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: int
    BsmtFinSF1: int
    BsmtFinSF2: int
    BsmtUnfSF: int
    TotalBsmtSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: int
    BsmtHalfBath: int
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    TotRmsAbvGrd: int
    Fireplaces: int
    GarageYrBlt: int
    GarageCars: int
    GarageArea: int
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ScreenPorch: int
    PoolArea: int
    MiscVal: int
    MoSold: int
    YrSold: int
