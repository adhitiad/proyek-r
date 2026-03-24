from fastapi import APIRouter, HTTPException
from app.core.data_collector import DataSourceManager
from app.core.signal_generator_v6 import  SignalGeneratorV5
from app.models.signal import SignalResponse

router = APIRouter(prefix="/signals", tags=["signals"])

@router.get("/{symbol}", response_model=SignalResponse)
async def get_signal(symbol: str):
    # Ambil data harga (saham IDX harus diakhiri .JK)
    if not symbol.endswith(".JK") and symbol.isalpha():
        symbol = f"{symbol}.JK"
    df = DataSourceManager().get_price_data(symbol, period="1mo", interval="1d")
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    generator = SignalGeneratorV6()  # modal contoh Rp 100 juta
    signal = generator.generate_signal(symbol, df)

    # Simpan ke MongoDB (opsional)
    # await signals_collection.insert_one(signal)

    return signal