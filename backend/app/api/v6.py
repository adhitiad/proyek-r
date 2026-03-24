from fastapi import APIRouter, HTTPException, Depends
from app.core.signal_generator_v6 import SignalGeneratorV6
from app.core.database import db
from app.core.singletons import get_data_manager
from app.core.security import require_api_key

import os

router = APIRouter(prefix="/v6", tags=["level6"])

# Initialize Level 6 generator
generator = SignalGeneratorV6(groq_api_key=os.getenv("GROQ_API_KEY"))

@router.get("/signal/{symbol}", dependencies=[Depends(require_api_key)])
async def get_level6_signal(symbol: str):
    """Get Level 6 signal with generative & predictive capabilities"""
    data_manager = get_data_manager()
    
    df = await data_manager.get_price_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    
    signal = await generator.generate_signal(symbol, df)
    
    # Save to database
    db.level6_signals.insert_one(signal.__dict__)
    
    return signal

@router.post("/scenario/generate", dependencies=[Depends(require_api_key)])
async def generate_scenario(prompt: str, symbol: str):
    """Generate market scenario using GAN + LLM"""
    data_manager = get_data_manager()
    
    df = await data_manager.get_price_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    
    scenario = await generator.scenario_gen.generate_scenario(prompt, df)
    return scenario

@router.post("/rl/optimize", dependencies=[Depends(require_api_key)])
async def optimize_with_rl(symbol: str):
    """Optimize strategy using Deep Reinforcement Learning"""
    data_manager = get_data_manager()
    
    df = await data_manager.get_price_data(symbol, period="6mo")
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    
    result = await generator.optimize_with_rl(symbol, df)
    return result

@router.get("/macro/{symbol}", dependencies=[Depends(require_api_key)])
async def get_macro_impact(symbol: str):
    """Get global macro impact analysis"""
    data_manager = get_data_manager()
    
    df = await data_manager.get_price_data(symbol)
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")
    
    result = await generator.analyze_macro_impact(symbol, df)
    return result

@router.get("/health")
async def get_system_health():
    """Get system health status with anomaly detection"""
    return generator.self_healing.get_health_status()
