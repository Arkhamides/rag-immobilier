from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat():
    return JSONResponse(status_code=501, content={"detail": "not implemented yet"})
