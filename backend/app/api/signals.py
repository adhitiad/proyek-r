from fastapi import APIRouter, HTTPException
from app.core.signal_generator import signal_to_legacy_dict
from app.core.config import settings
from app.core.singletons import get_data_manager, get_signal_generator_v5
from app.models.signal import SignalResponse

router = APIRouter(prefix="/signals", tags=["signals"])

@router.get("/{symbol}", response_model=SignalResponse)
async def get_signal(symbol: str):
    # Ambil data harga (saham IDX harus diakhiri .JK)
    if not symbol.endswith(".JK") and symbol.isalpha():
        symbol = f"{symbol}.JK"
    df = await get_data_manager().get_price_data(
        symbol,
        period=settings.DATA_PERIOD,
        interval=settings.DATA_INTERVAL
    )
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    signal = await get_signal_generator_v5().generate_signal(symbol, df)

    # Simpan ke MongoDB (opsional)
    # await signals_collection.insert_one(signal)

    return signal_to_legacy_dict(signal)
