
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY Membangun_model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode
COPY . .

# Expose port
EXPOSE 8000

# Jalankan aplikasi monitoring saat container start
CMD ["python", "Monitoring_and_Logging/prometheus_exporter.py"]
