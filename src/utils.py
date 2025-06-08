import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

def prepare_data():
    train = pd.read_csv("data/realty_data.csv")

    train.loc[train['rooms'].isnull(), 'rooms'] = train['rooms'].median()
    train.loc[train["city"].isnull(), "city"] = train["city"].mode()[0]
    train['rooms'] = train['rooms'].astype(int)

    cols_to_drop = ["product_name", "period", 'postcode', 'address_name', 'lat', 'lon', 'object_type', 'floor', 'settlement', 'district', 'area', 'description', 'source']
    train = train.drop(columns=cols_to_drop, errors='ignore')
    train = pd.get_dummies(train, drop_first=False)
    return train

def train_model(train):
    X, y = train.drop("price", axis=1), train['price']
    lr = LinearRegression()
    lr.fit(X, y)
    with open('rf_fitted.pkl', 'wb') as f:
        pickle.dump(lr, f)

def read_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Файл модели не существует")

    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    return model