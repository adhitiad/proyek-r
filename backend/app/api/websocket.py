from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import manager

router = APIRouter()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Menerima pesan dari client (misal subscribe)
            data = await websocket.receive_text()
            # Di sini bisa implementasi subscription per symbol
            await websocket.send_text(f"Subscribed to {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)