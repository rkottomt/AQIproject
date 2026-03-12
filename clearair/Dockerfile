FROM python:3.11-slim

WORKDIR /app

# System dependencies for netCDF4, psycopg2, and scientific libs
RUN apt-get update && apt-get install -y \
    libhdf5-dev libnetcdf-dev gcc g++ \
    libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/plots models/saved

EXPOSE 8000 8501

CMD ["python", "main.py", "serve"]
