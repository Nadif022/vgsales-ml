import json, os, time
from pathlib import Path

import joblib, pandas as pd
from flask import Flask, jsonify, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
LOG_PATH = Path(os.getenv("PRED_LOG", "logs/predictions.jsonl"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

bundle = joblib.load(MODEL_PATH)
pipeline, FEATURES = bundle["pipeline"], bundle["features"]

app = Flask(__name__)

REQS = Counter("api_requests_total", "Jumlah request", ["endpoint", "status"])
LATENCY = Histogram("api_request_seconds", "Latensi request", ["endpoint"])
PRED = Histogram("prediction_value", "Distribusi nilai prediksi",
                 buckets=[0, .05, .1, .25, .5, 1, 2, 5, 10, 50])


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.post("/predict")
def predict():
    start = time.time()
    status = 200
    try:
        body = request.get_json(force=True)
        records = body["data"] if isinstance(body, dict) and "data" in body else body
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(records)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            status = 400
            return jsonify(error=f"Fitur hilang: {missing}", required=FEATURES), 400
        df = df[FEATURES]
        preds = pipeline.predict(df).tolist()
        with LOG_PATH.open("a") as f:
            for row, p in zip(df.to_dict("records"), preds):
                PRED.observe(max(p, 0))
                f.write(json.dumps({"ts": time.time(), "input": row, "prediction": p}) + "\n")
        return jsonify(prediction=preds)
    except Exception as e:  # input rusak, dsb.
        status = 400
        return jsonify(error=str(e)), 400
    finally:
        REQS.labels("/predict", status).inc()
        LATENCY.labels("/predict").observe(time.time() - start)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
