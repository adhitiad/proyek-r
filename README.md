# Proyek-R - Forex & IDX Signal Generator

Sistem backend FastAPI dan dashboard Streamlit untuk menghasilkan sinyal trading, backtest, optimasi, dan monitoring.

**Stack utama**
- FastAPI backend
- Streamlit dashboard
- MongoDB untuk storage
- Redis untuk job broker/queue

**Fitur utama**
- Generate sinyal (v5 dan v6)
- Backtest dan screening
- Model training dan AutoML
- Optimasi parameter
- Sentiment news
- Dashboard monitoring

**Arsitektur**
- Backend API: FastAPI berjalan di port 8000
- Dashboard: Streamlit berjalan di port 8501
- Storage: MongoDB
- Cache/queue: Redis

**Quick Start (Docker Compose)**
1. Jalankan semua service.

```bash
docker compose up --build
```

2. Akses aplikasi.
- API: http://localhost:8000
- Dashboard: http://localhost:8501

**Jalankan Lokal (Tanpa Docker)**
1. Buat virtual environment dan install dependency.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend\requirements.txt
```

2. Salin env contoh.

```bash
copy .env.example .env
```

3. Jalankan backend.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

4. Jalankan dashboard.

```bash
streamlit run backend\dashboard.py
```

**Konfigurasi Environment**
- Lokal: gunakan `./.env`
- Docker/Compose: gunakan `./backend/.env`

Contoh variabel penting:
- `MONGODB_URL`
- `DATABASE_NAME`
- `API_KEY` (opsional, jika diisi maka semua endpoint yang dilindungi butuh header `X-API-Key`)
- `API_URL` (dashboard menuju API)
- `REDIS_URL`

**Dokumentasi API**
Lihat `API.md` untuk detail endpoint dan contoh request.

**Catatan**
- Jika `API_KEY` diset, pastikan dashboard dan semua request API menyertakan header `X-API-Key`.
- Scheduler bisa dimatikan dengan `DISABLE_SCHEDULER=true`.
