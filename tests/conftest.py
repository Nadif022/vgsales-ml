import os, sys
import joblib, numpy as np, pandas as pd, pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
FEATURES = ["Year_of_Release", "EU_Sales", "JP_Sales", "Other_Sales", "Platform", "Genre", "Publisher"]


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Year_of_Release": rng.integers(1990, 2016, 50),
                       "EU_Sales": rng.random(50), "JP_Sales": rng.random(50),
                       "Other_Sales": rng.random(50),
                       "Platform": rng.choice(["PS4", "X360"], 50),
                       "Genre": rng.choice(["Action", "Sports"], 50),
                       "Publisher": rng.choice(["EA", "Sony"], 50)})
    pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                              ["Platform", "Genre", "Publisher"])], remainder="passthrough")
    pipe = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(n_estimators=5))])
    pipe.fit(df[FEATURES], rng.random(50))
    joblib.dump({"pipeline": pipe, "features": FEATURES}, d / "m.joblib")
    os.environ["MODEL_PATH"] = str(d / "m.joblib")
    os.environ["PRED_LOG"] = str(d / "log.jsonl")
    from app.main import app
    return app.test_client()
