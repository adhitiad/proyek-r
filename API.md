# API Docs

**Base URL**
- Lokal: http://localhost:8000
- Docker Compose: http://localhost:8000

**Authentication**
- Jika `API_KEY` di-set, maka endpoint yang dilindungi mewajibkan header `X-API-Key`.
- Jika `API_KEY` kosong, semua endpoint bisa diakses tanpa header.
 - JWT demo menggunakan `JWT_SECRET` untuk sign token.

**Header umum**
- `Content-Type: application/json`
- `X-API-Key: <API_KEY>` (opsional, sesuai konfigurasi)

**Health & Metrics**
| Method | Path | Auth | Keterangan |
| --- | --- | --- | --- |
| GET | `/` | No | Health sederhana |
| GET | `/metrics` | No | Prometheus metrics |

**Auth**
| Method | Path | Auth | Body | Keterangan |
| --- | --- | --- | --- | --- |
| POST | `/auth/jwt` | Yes | `sub`, `role` | Generate JWT demo |

**Signals (v5)**
| Method | Path | Auth | Query Params | Keterangan |
| --- | --- | --- | --- | --- |
| GET | `/signals/{symbol}` | No | `symbol` di path | Generate sinyal v5 |

**Backtest**
| Method | Path | Auth | Query Params | Keterangan |
| --- | --- | --- | --- | --- |
| POST | `/backtest/run` | Yes | `symbol`, `start_date`, `end_date`, `initial_capital`, `risk_per_trade`, `commission`, `slippage` | Jalankan backtest dan simpan hasil |
| POST | `/backtest/screen` | Yes | `symbols` (repeatable), `start_date`, `end_date`, `initial_capital`, `risk_per_trade`, `commission`, `slippage` | Screening multi-simbol |
| GET | `/backtest/metrics/{symbol}` | No | `start_date`, `end_date`, `initial_capital` | Metrik backtest detail |
| GET | `/backtest/trades/{symbol}` | No | `start_date`, `end_date`, `initial_capital`, `limit` | Riwayat trades |

**Evaluation**
| Method | Path | Auth | Keterangan |
| --- | --- | --- | --- |
| GET | `/evaluation/metrics` | No | Metrik performa dari koleksi trades |

**Model**
| Method | Path | Auth | Body / Query | Keterangan |
| --- | --- | --- | --- | --- |
| POST | `/model/train` | Yes | Query: `symbols`, `start_date`, `end_date`, `target_days`, `epochs` | Train model |
| POST | `/model/automl` | Yes | Body JSON: `symbols`, `start_date`, `end_date`, `target_days`, `trials` | Jalankan AutoML |
| GET | `/model/automl/latest` | No | - | Ambil hasil AutoML terbaru |
| GET | `/model/list` | No | - | List model |
| POST | `/model/activate` | Yes | Query: `model_path` | Set model aktif |
| GET | `/model/active` | No | - | Info model aktif |

**Optimize**
| Method | Path | Auth | Body / Query | Keterangan |
| --- | --- | --- | --- | --- |
| POST | `/optimize/start` | Yes | Query: `symbol`, `start_date`, `end_date`, `metric`, `initial_capital`, `commission`, `slippage` + Body JSON: `param_grid` | Mulai optimasi |
| GET | `/optimize/results/{opt_id}` | No | - | Hasil optimasi |

**Sentiment**
| Method | Path | Auth | Keterangan |
| --- | --- | --- | --- |
| GET | `/sentiment/{symbol}` | Yes | Analisis sentimen news |

**Telegram**
| Method | Path | Auth | Keterangan |
| --- | --- | --- | --- |
| POST | `/telegram/webhook` | No | Webhook Telegram bot |

**Level 6**
| Method | Path | Auth | Query Params | Keterangan |
| --- | --- | --- | --- | --- |
| GET | `/v6/signal/{symbol}` | Yes | `symbol` di path | Sinyal generasi v6 |
| POST | `/v6/scenario/generate` | Yes | `prompt`, `symbol` | Generate skenario pasar |
| POST | `/v6/rl/optimize` | Yes | `symbol` | Optimasi RL |
| GET | `/v6/macro/{symbol}` | Yes | `symbol` di path | Analisis makro |
| GET | `/v6/health` | No | - | Status health |

**WebSocket**
| Method | Path | Auth | Keterangan |
| --- | --- | --- | --- |
| WS | `/ws` | No | WebSocket sederhana untuk subscribe |

**Contoh Request**
```bash
curl "http://localhost:8000/signals/BBCA.JK"
```

```bash
curl -X POST "http://localhost:8000/backtest/run?symbol=BBCA.JK&start_date=2024-01-01&end_date=2024-12-31" \
  -H "X-API-Key: 123456"
```

```bash
curl -X POST "http://localhost:8000/optimize/start?symbol=BBCA.JK&start_date=2024-01-01&end_date=2024-12-31" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 123456" \
  -d "{\"param_grid\": {\"lookback_ob\": [20, 30], \"volume_mult_ob\": [1.5, 2.0]}}"
```

```bash
curl -X POST "http://localhost:8000/model/activate?model_path=models/model_v1.pkl" \
  -H "X-API-Key: 123456"
```

```bash
curl -X POST "http://localhost:8000/model/automl" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 123456" \
  -d '{"symbols": ["BBCA.JK", "BBRI.JK"], "start_date": "2024-01-01", "end_date": "2024-12-31", "target_days": 5}'
```

```bash
curl -X POST "http://localhost:8000/auth/jwt" \
  -H "Content-Type: application/json" \
  -d '{"sub": "demo@proyek-r.app", "role": "user"}'
```

**Catatan**: Jika `API_KEY` di-set di environment, tambahkan header `X-API-Key: <value>`

```bash
curl -X GET "http://localhost:8000/sentiment/BBCA.JK" \
  -H "X-API-Key: 123456"
```

```bash
# Jika API_KEY tidak di-set, header tidak diperlukan
curl -X GET "http://localhost:8000/v6/signal/BBCA.JK"

# Jika API_KEY di-set, gunakan header
curl -X GET "http://localhost:8000/v6/signal/BBCA.JK" \
  -H "X-API-Key: 123456"
```
