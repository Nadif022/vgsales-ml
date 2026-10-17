"""Latih model & simpan sebagai Pipeline (preprocessing + model jadi satu file).
Jalankan:  python train.py "data/Raw Data.csv"
"""
import json, sys
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

TARGET = "NA_Sales"
NUMERIC = ["Year_of_Release", "EU_Sales", "JP_Sales", "Other_Sales"]
CATEGORICAL = ["Platform", "Genre", "Publisher"]
# Global_Sales sengaja DIBUANG: nilainya = NA+EU+JP+Other, jadi bocoran target (data leakage).
# Name juga dibuang: hampir unik per baris, tidak berguna untuk generalisasi.
FEATURES = NUMERIC + CATEGORICAL

path = sys.argv[1] if len(sys.argv) > 1 else "data/Raw Data.csv"
df = pd.read_csv(path)[FEATURES + [TARGET]]
df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median())
for c in CATEGORICAL:
    df[c] = df[c].fillna(df[c].mode()[0]).astype(str)

x_train, x_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.2, random_state=1)

pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL),
], remainder="passthrough")
pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=184))])
pipe.fit(x_train, y_train)

pred = pipe.predict(x_test)
metrics = {"MAE": mean_absolute_error(y_test, pred),
           "MSE": mean_squared_error(y_test, pred),
           "R2": r2_score(y_test, pred)}
print(metrics)

joblib.dump({"pipeline": pipe, "features": FEATURES}, "model/model.joblib")
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
x_train.sample(min(2000, len(x_train)), random_state=1).to_csv("monitoring/reference.csv", index=False)
print("Tersimpan: model/model.joblib, model/metrics.json, monitoring/reference.csv")
