import logging
import os

import asgi_correlation_id
import uvicorn
from fastapi import APIRouter, FastAPI
from uvicorn.config import LOGGING_CONFIG

from app.dto.chat import ChatPayload
from app.dto.mem import AddPayload
from log import configure_logging
from mem_init import mem


app = FastAPI(on_startup=[configure_logging])
app.add_middleware(asgi_correlation_id.CorrelationIdMiddleware)
router = APIRouter()

logger = logging.getLogger(__name__)

@router.post("/memory/add")
async def add_memory(payload: AddPayload):
    return mem.add(payload.messages, payload.user_name, payload.namespace, payload.metadata, payload.filters)

@router.delete("/memory/{namespace}")
async def delete_memory(namespace: str):
    return mem.delete_all(namespace)

@router.post("/chat")
async def add_memory(payload: ChatPayload):
    return mem.chat(payload.system_prompt, payload.query, payload.user_name, payload.namespace)

app.include_router(router)


if __name__ == "__main__":
    LOGGING_CONFIG["handlers"]["access"]["filters"] = [asgi_correlation_id.CorrelationIdFilter()]
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s  - %(filename)s - %(funcName)s- Line: %(lineno)d - %(message)s"
    uvicorn.run("main:app", port=8080, log_level=os.environ.get("LOGLEVEL", "INFO").lower())