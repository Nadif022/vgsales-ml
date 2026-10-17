# Video Game Sales – NA_Sales Predictor
API prediksi (Flask) + CI/CD (GitHub Actions) + monitoring (Prometheus/Grafana, drift check, uptime).

## Jalankan lokal
    pip install -r requirements.txt
    python train.py "data/Raw Data.csv"
    gunicorn -b 0.0.0.0:8000 app.main:app

    curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"data":[{"Year_of_Release":2010,"EU_Sales":0.5,"JP_Sales":0.1,"Other_Sales":0.2,"Platform":"PS4","Genre":"Action","Publisher":"Electronic Arts"}]}'

## Hasil training (dataset Raw Data.csv, 16.719 baris)
MAE 0.110 | MSE 0.133 | R² 0.772  (tanpa Global_Sales agar tidak ada data leakage)

## Fitur yang wajib dikirim ke /predict
Year_of_Release, EU_Sales, JP_Sales, Other_Sales, Platform, Genre, Publisher
