# STEP 15: DEPLOYMENT & CONTAINERIZATION REPORT

**Executive Summary:**
This report documents the architectural configuration, containerization, and cloud deployment topology for **Stage 15 (Deployment)** of the SalesSight AI commercial demand intelligence engine. In accordance with user specifications, the production deployment bypasses traditional enterprise AWS infrastructure in favor of a modern, cost-efficient, decoupled cloud topology:
1. **Frontend UI Tier:** **Streamlit Community Cloud (`https://share.streamlit.io/`)**.
2. **Backend ML REST API Tier:** **Render Web Service (`https://render.com`)**.
3. **Containerization Engine:** Production multi-stage Docker (`Dockerfile`), local multi-container orchestration (`docker-compose.yml`), and Infrastructure-as-Code (`render.yaml`).

---

## 1. Decoupled Production Architecture Overview

The system architecture cleanly separates presentation from analytical computation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER WEB BROWSER                                │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTPS (:443)
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               FRONTEND: Streamlit Community Cloud                      │
│                  (https://share.streamlit.io/)                         │
│  - Entry Point: app.py                                                 │
│  - Theme: .streamlit/config.toml (Deep Slate #0E1117, Electric Blue)   │
│  - Secret Config: API_URL="https://salessight-ai-backend.onrender.com" │
│  - Communicator: src/ui/client.py (requests / in-process fallback)     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ REST JSON Over HTTPS
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               BACKEND: Render Web Service (FastAPI)                    │
│                     (https://render.com)                               │
│  - Entry Point: uvicorn src.api.main:app --port $PORT                  │
│  - Health Probe: GET /health (Liveness / Readiness SLA < 50ms)         │
│  - Engine: ProductionForecastEngine & DemandIntelligenceEngine         │
│  - Serialized Artifacts: models/artifacts/production/*.joblib          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Backend Deployment on Render (`https://render.com`)

The FastAPI backend is deployed as a cloud Web Service on Render using the declarative Infrastructure-as-Code specification defined in [`render.yaml`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/render.yaml):

### 2.1 Render Infrastructure as Code (`render.yaml`)
```yaml
services:
  - type: web
    name: salessight-ai-backend
    runtime: python
    region: oregon
    plan: free
    branch: main
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: ENVIRONMENT
        value: production
      - key: LOG_LEVEL
        value: INFO
```

### 2.2 Step-by-Step Render Deployment Workflow
1. **GitHub Connection:** Push the `SalesSight AI` repository to GitHub.
2. **New Web Service:** Log in to [Render Dashboard](https://dashboard.render.com/) and select **New > Web Service**.
3. **Repository Linking:** Select the `SalesSight AI` GitHub repository.
4. **Environment Configuration:**
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
5. **Health Check Path:** Set to `/health`.
6. **Deploy:** Click **Create Web Service**. Render provisions the environment, installs dependencies, compiles wheel packages, and activates the service.
7. **Production URL:** Once deployed, Render assigns a public URL:
   `https://salessight-ai-backend.onrender.com`

---

## 3. Frontend Deployment on Streamlit Community Cloud

The executive dashboard is deployed on Streamlit Community Cloud (`https://share.streamlit.io/`):

### 3.1 Streamlit Configuration (`.streamlit/config.toml`)
Adheres strictly to the §16.1 Design Aesthetics:
```toml
[theme]
primaryColor = "#2563EB"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#161B22"
textColor = "#FFFFFF"
font = "sans serif"

[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = true
```

### 3.2 Step-by-Step Streamlit Cloud Deployment Workflow
1. **Access Portal:** Navigate to [share.streamlit.io](https://share.streamlit.io/) and authorize GitHub.
2. **New App:** Click **Create App**.
3. **Repository Settings:**
   - **Repository:** `<your-github-username>/SalesSight-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. **Secrets / Environment Variable Configuration:**
   - In **Advanced Settings > Secrets**, set:
     ```toml
     API_URL = "https://salessight-ai-backend.onrender.com"
     ```
5. **Deploy:** Click **Deploy!**. Streamlit Cloud builds the container, detects `requirements.txt`, applies theme settings from `.streamlit/config.toml`, and exposes the live application at:
   `https://<subdomain>.streamlit.app`

---

## 4. Production Containerization Architecture (Docker)

To support self-hosted container environments, local testing, and enterprise container registries, full Docker containerization is maintained without regression:

### 4.1 Production Multi-Stage Dockerfile (`Dockerfile`)
- **Stage 1 (Builder):** Uses `python:3.11-slim`, installs GCC, OpenMP (`libgomp1`), and compiles all requirements into `/root/.local`.
- **Stage 2 (Runner):** Minimal slim image, copies compiled wheels from builder, establishes a dedicated non-root security user (`appuser` in `appgroup`), and enforces least-privilege security.
- **Healthcheck:** Automatic liveness probe checking `/health`:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
      CMD curl -f http://localhost:8000/health || exit 1
  ```

### 4.2 Local Multi-Container Orchestration (`docker-compose.yml`)
Enables one-command local launching of both services:
```bash
docker-compose up --build
```
- **Service 1 (`api`):** Binds to port `8000:8000`, runs `uvicorn src.api.main:app`.
- **Service 2 (`web`):** Binds to port `8501:8501`, runs `streamlit run app.py`, depends on `api` health check, and receives `API_URL=http://api:8000`.

---

## 5. Verification & Unit Test Suite

The deployment configuration is validated by unit tests in [`tests/unit/test_step15_deployment.py`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/tests/unit/test_step15_deployment.py):
- `test_dockerfile_structure_and_security`: **PASSED** (Multi-stage, non-root user, healthcheck verified).
- `test_docker_compose_configuration`: **PASSED** (Service definitions, port bindings, and networking verified).
- `test_render_yaml_blueprint_specification`: **PASSED** (Render IaC syntax, build commands, and health probe verified).
- `test_streamlit_cloud_config_toml`: **PASSED** (Theme tokens and server parameters verified).
- `test_dockerignore_file_rules`: **PASSED** (Exclusion patterns verified).

Stage 15 deployment setup is verified and ready for cloud release.
