# Sideral ECMWF Backend

Serviço independente para o Render. Baixa os GRIB do ECMWF IFS HRES diretamente pelo `ecmwf-opendata`, aproximadamente 9 km (`0p1`), e mantém cache no disco temporário.

Build: `pip install -r requirements.txt`  
Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`  
Health check: `/health`

Endpoints: `/api/ecmwf/status`, `/api/ecmwf/latest?step=0` e `/api/ecmwf/file/{arquivo}`.
