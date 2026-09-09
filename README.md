# AgniDrishti

AgniDrishti is a Flask-based disaster intelligence command center for detecting NASA thermal hotspots, estimating GeoRisk, classifying thermal sources, explaining escalation, and supporting emergency response decisions.

## Capabilities

- NASA FIRMS hotspot ingestion and GeoRisk scoring
- Escalation and probability analysis
- Thermal event classification
- Explainable AI-style factor summaries
- Temporal and weather-risk analysis
- Population exposure estimation
- Infrastructure threat scoring
- Interactive Folium hotspot map
- Decision-support center and PDF incident reports
- Responsive Bootstrap command-center UI

## Repository Layout

```text
GeoRisk/
|-- app.py                    Flask application and routes
|-- config.py                 Runtime configuration placeholder
|-- requirements.txt          Python dependencies
|-- Dockerfile                Container deployment
|-- Procfile                  PaaS deployment command
|-- .env.example              Environment variable template
|-- alerts/                   Alert generation logic
|-- analytics/                Analytics helpers
|-- anomaly_detection/        Anomaly detection logic
|-- classification/           Event classification logic
|-- database/                 Database models and access
|-- gis/                      Map generation and spatial analysis
|-- ml/                       Risk and GeoRisk engines
|-- notifications/            Notification generation
|-- prediction/               Prediction logic
|-- src/                      Newer processing and prediction scripts
|-- data/raw/                 Small source fixtures and local inputs
|-- data/processed/           Local/generated datasets
|-- static/                   CSS and JavaScript assets
|-- templates/                Jinja templates
```

Generated datasets in `data/processed/` and large raw downloads are intentionally ignored by Git. Generate them locally or provide them through a data pipeline/object store in deployment.

## Local Setup

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Open https://agnidrishti-dh08.onrender.com

## Generate Local Data

The current prototype pipeline can be run in this order:

```powershell
python ml/risk_engine.py
python ml/georisk_engine.py
python prediction/escalation_predictor.py
python data/processing/advanced_event_classifier.py
python data/processing/population_analysis.py
python data/processing/weather_analysis.py
python data/processing/infrastructure_threat.py
python gis/hotspot_mapper.py
```

Some scripts depend on intermediate files created by earlier steps. For production, move this pipeline to a scheduled job and store generated datasets outside the web container.

## Run With Gunicorn

Linux, Render, Railway, or a container:

```bash
gunicorn --bind 0.0.0.0:$PORT --workers 2 app:app
```

The included `Dockerfile` and `Procfile` use this command.

## Deployment Notes

- Set `SECRET_KEY` and `DATABASE_URL` through the platform secret manager.
- Do not commit `.env`, credentials, `venv/`, caches, or large NASA downloads.
- Persistent generated data should use a database, object storage, or a mounted volume.
- The built-in Flask server is for local development only.
- Validate all data-processing jobs before starting the production web service.

## Prototype Data Disclosure

Population, weather, and historical trend values may be prototype/demo estimates when live sources are unavailable. Production deployments should label data freshness and source, and replace generated estimates with verified feeds.

## Validation

Basic application smoke test:

```powershell
.\venv\Scripts\python.exe -m py_compile app.py
.\venv\Scripts\python.exe -c "from app import app; print(app.test_client().get('/').status_code)"
```
