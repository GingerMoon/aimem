import json
import logging
from typing import Callable, Any
from fastapi.responses import Response

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool

from infra.logutil.context import *

logger = logging.getLogger('log_util')

class CustomResponse(Response):
    media_type = "application/json"

    def __init__(self, content: Any, status_code: int):
        super().__init__(content)
        self.body = json.dumps({"data": content}).encode("utf-8")
        self.status_code = status_code

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
            self, request: Request, call_next: Callable
    ):
        extend_log_ctx_tags({LOG_CORRELATION_ID:str(uuid.uuid4())})
        body = await request.body()
        request._body = body
        body_str = body.decode("utf-8")
        logging.info(f"Request: {request.method} {request.url} \n Body: {json.dumps(body_str)}")
        response = await call_next(request)
        response_body = [chunk async for chunk in response.body_iterator]
        logging.info(f"Response: {response.status_code} \n Body: {response_body[0].decode()}")
        response.body_iterator = iterate_in_threadpool(iter(response_body))
        return response