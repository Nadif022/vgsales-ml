SAMPLE = {"Year_of_Release": 2010, "EU_Sales": 0.5, "JP_Sales": 0.1,
          "Other_Sales": 0.2, "Platform": "PS4", "Genre": "Action", "Publisher": "EA"}


def test_health(client):
    assert client.get("/health").json == {"status": "ok"}


def test_predict_ok(client):
    r = client.post("/predict", json={"data": [SAMPLE]})
    assert r.status_code == 200 and len(r.json["prediction"]) == 1


def test_unknown_category_tidak_crash(client):
    r = client.post("/predict", json={"data": [{**SAMPLE, "Publisher": "BelumPernahAda"}]})
    assert r.status_code == 200


def test_fitur_hilang(client):
    r = client.post("/predict", json={"data": [{"EU_Sales": 1}]})
    assert r.status_code == 400


def test_metrics(client):
    assert b"api_requests_total" in client.get("/metrics").data
