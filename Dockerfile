FROM python:3.12-slim

# Hugging Face Spaces menjalankan container sebagai user non-root (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user app app
COPY --chown=user model model

EXPOSE 7860
CMD gunicorn -b 0.0.0.0:${PORT} -w 1 --threads 4 app.main:app
