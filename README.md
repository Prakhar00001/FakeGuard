# FakeGuard

FakeGuard is a full-stack fake review detection prototype built with:
- Python + FastAPI for the prediction API
- TF-IDF text features plus linguistic heuristics
- An ensemble of Logistic Regression, Random Forest, and XGBoost (when available)
- A simple dashboard UI that can be served as a static site

## 🌐 Live Demo

**Deployed on Render:** [https://fakeguard-app.onrender.com](https://fakeguard-app.onrender.com)

> **Note:** Free Render instances spin down after inactivity. The first request may take 30-60 seconds to wake up.

## Prerequisites
- Python 3.10+
- Git
- Optional: Node.js only if you want to use the Vite-based workflow

## Run from scratch on Windows

```powershell
git clone <your-github-repo-url>
cd FakeGuard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Start the backend
```powershell
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

### Start the frontend
Open a second terminal:
```powershell
cd frontend
npm install
npm run dev
```

Then open:
- Frontend: http://127.0.0.1:3000/
- API docs: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial FakeGuard app"
git branch -M main
git remote add origin https://github.com/<your-username>/FakeGuard.git
git push -u origin main
```

## Deploy to Render

A Render configuration file is included in [render.yaml](render.yaml). 

1. Push this repo to your GitHub account
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and deploy

The app will be available at `https://fakeguard-app.onrender.com`.
