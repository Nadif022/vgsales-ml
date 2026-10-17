FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app app
COPY model model
ENV PORT=8000
EXPOSE 8000
CMD gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 4 app.main:app
