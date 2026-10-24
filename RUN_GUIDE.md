# SalesSight AI — Comprehensive Production Run Guide

This guide provides step-by-step instructions to set up, test, run, containerize, and deploy the **SalesSight AI** smart sales forecasting and commercial demand intelligence engine.

---

## 1. System Prerequisites

Ensure your host machine or cloud instance meets the following requirements:
- **Operating System:** Windows 10/11, macOS, or Linux (Ubuntu 20.04/22.04 LTS recommended).
- **Python Version:** Python `3.10`, `3.11`, or `3.12` (Python 3.11 recommended).
- **Package Manager:** `pip` (v23.0+) or `conda` (Miniconda / Anaconda).
- **Git:** Installed and configured.
- **Docker & Docker Compose:** Optional for containerized execution.

---

## 2. Environment Setup

### Option 2A: Using Conda (Recommended)

```bash
# 1. Open terminal / PowerShell and create a new environment
conda create -n salessight python=3.11 -y

# 2. Activate the environment
conda activate salessight

# 3. Navigate to the project root directory
cd "SalesSight AI"
```

### Option 2B: Using Python Standard Virtual Environment (`venv`)

```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3. Dependency Installation

Install the production dependencies. The `requirements.txt` file is pinned with a strict **NumPy 2.0 guardrail** (`numpy>=1.24.0,<2.0.0`) to prevent ABI incompatibility errors during compilation on Linux/Cloud:

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install production dependencies
pip install -r requirements.txt

# (Optional) Install development & test dependencies
pip install -r requirements-dev.txt
```

---

## 4. Running the Project Locally

The architecture consists of two decoupled services:
1. **Backend:** FastAPI REST ML Engine (`port 8000`)
2. **Frontend:** Streamlit Interactive Executive Dashboard (`port 8501`)

### Step 4.1: Start the Backend ML REST API

In your first terminal:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Liveness Probe:** Open [http://localhost:8000/health](http://localhost:8000/health) (returns `HTTP 200 OK` and model manifest status).
- **Interactive Swagger Documentation:** Open [http://localhost:8000/docs](http://localhost:8000/docs) to test `/forecast`, `/models`, and `/intelligence` endpoints.

### Step 4.2: Start the Frontend Streamlit Dashboard

In a second terminal (with the same environment activated):
```bash
streamlit run app.py
```
- The dashboard automatically opens at [http://localhost:8501](http://localhost:8501).
- The client in `src/ui/client.py` connects to `http://localhost:8000` by default, with automatic in-process fallback if the backend daemon is temporarily offline.

---

## 5. Running with Docker & Docker Compose (Containerized)

Both services can be built and orchestrated with a single command:

```bash
# Build and run containers in foreground
docker-compose up --build

# Or run in detached background mode
docker-compose up -d --build
```

- **Backend API:** Available at `http://localhost:8000` (Container: `salessight_api`)
- **Frontend Web UI:** Available at `http://localhost:8501` (Container: `salessight_web`)
- **Container Health Check:** Docker automatically waits for `salessight_api` to return healthy before launching `salessight_web`.

To stop containers:
```bash
docker-compose down
```

---

## 6. Cloud Deployment Instructions

### 6.1 Backend Deployment on Render (`https://render.com`)

The FastAPI inference service is preconfigured for zero-friction deployment on Render via [`render.yaml`](file:///c:/Users/mihir/Desktop/SalesSight%20AI/render.yaml):

1. **Push Code to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy SalesSight AI Backend & Frontend"
   git push origin main
   ```
2. **Open Render Dashboard:**
   - Log in at [dashboard.render.com](https://dashboard.render.com/).
   - Click **New > Web Service**.
3. **Connect Repository:**
   - Select your GitHub repository (`SalesSight AI`).
4. **Configure Service Settings:**
   - **Name:** `salessight-ai-backend`
   - **Region:** Any preferred region (e.g., Oregon, Ohio, Frankfurt).
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** `Free` (or Starter for commercial production).
5. **Configure Health Check:**
   - Under **Advanced Settings**, set **Health Check Path** to `/health`.
6. **Deploy:** Click **Create Web Service**.
7. **Copy Your Backend URL:**
   - Render assigns a URL like:
     `https://salessight-ai-backend.onrender.com`
   - Test by opening `https://salessight-ai-backend.onrender.com/health` in your browser.

---

### 6.2 Frontend Deployment on Streamlit Community Cloud (`https://share.streamlit.io/`)

1. **Access Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. **Create New App:**
   - Click **Create app** (or **New app**).
3. **Repository Settings:**
   - **Repository:** `<your-username>/SalesSight-AI`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** Customize subdomain if desired (e.g., `salessight-ai.streamlit.app`).
4. **Configure Secrets / Environment Variables (CRITICAL STEP):**
   - Click **Advanced Settings**.
   - Under **Secrets**, add the Render backend URL:
     ```toml
     API_URL = "https://salessight-ai-backend.onrender.com"
     BACKEND_URL = "https://salessight-ai-backend.onrender.com"
     ```
5. **Deploy:** Click **Deploy!**.
   - Streamlit Cloud provisions the container, automatically detects `requirements.txt`, applies `.streamlit/config.toml` dark theme, and connects to your live Render backend API.

---

## 7. Running the Unit Test Suite (74 Passing Tests)

To execute the entire 16-stage test suite:

```bash
# Run all unit tests with verbose output
pytest tests/unit/ -v

# Run tests for specific stages:
pytest tests/unit/test_step12_model_persistence.py -v   # Stage 12: Persistence & Checksums
pytest tests/unit/test_step13_ml_api.py -v             # Stage 13: FastAPI Endpoints
pytest tests/unit/test_step14_web_ui.py -v             # Stage 14: Streamlit UI
pytest tests/unit/test_step15_deployment.py -v         # Stage 15: Docker & Render
pytest tests/unit/test_step16_production.py -v         # Stage 16: Production QA
```

Expected result:
```
======================= 74 passed, 6 warnings in ~37s =======================
```

---

## 8. Troubleshooting & FAQ

### Q1: Streamlit frontend shows "Demonstration Mode Active" banner. Why?
> **Answer:** The underlying Northwind-style historical transaction dataset ends on `2014-09-29`. When selecting any date beyond `2014-12-28` (such as current calendar dates in 2026), Demonstration Mode must be enabled. The system maps the 2026 date onto historical seasonal day-of-week and month cycles, shifting display labels to 2026 while preserving authentic model inferences and residual bounds.

### Q2: What happens if Render backend spins down on the Free tier?
> **Answer:** Render's free tier spins down after 15 minutes of inactivity. When a new request arrives, Render automatically spins up within ~30 seconds. In the meantime, `src/ui/client.py` contains an automatic in-process fallback engine, so the dashboard continues to function without crashing!

### Q3: How do I export forecast predictions?
> **Answer:** On the dashboard, scroll to Row 3B and click **📥 Download RFC-4180 Forecast CSV**. The exported CSV file includes standardized audit comment headers, prediction values, and calibrated 80% confidence intervals.
