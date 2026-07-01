from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("")
async def get_metrics():
    return JSONResponse(status_code=501, content={"detail": "not implemented yet"})


@router.get("/history")
async def get_metrics_history():
    return JSONResponse(status_code=501, content={"detail": "not implemented yet"})


@router.post("/evaluate")
async def evaluate():
    return JSONResponse(status_code=501, content={"detail": "not implemented yet"})
