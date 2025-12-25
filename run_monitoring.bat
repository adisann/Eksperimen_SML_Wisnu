@echo off
echo Starting Prometheus and Grafana via Docker...
docker-compose up -d

echo.
echo Monitoring Stack Started!
echo 1. Prometheus: http://localhost:9090
echo 2. Grafana:    http://localhost:3000 (Login: admin / admin)
echo.
echo Please open Grafana to set up your dashboard.
pause
