# Backend image: FastAPI wrapping the core recommendation engine.
# Build context must be the repository root so that ./core and ./backend are
# both available:
#   docker build -f backend/Dockerfile -t movie-recommender-backend .
FROM python:3.12-slim

WORKDIR /app

# Install the core engine first (its deps change rarely -> better layer caching).
COPY core/ /app/core/
RUN pip install --no-cache-dir /app/core

# Backend web dependencies.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Backend source.
COPY backend/ /app/backend/

# Dataset is downloaded at startup into a writable, mountable directory.
ENV MOVIELENS_DATA_DIR=/data \
    AUTO_TRAIN_ON_STARTUP=true \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser \
    && mkdir -p /data \
    && chown -R appuser /data
USER appuser

WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
