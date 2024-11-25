import json
import logging
import traceback
from typing import Callable, Any
from starlette.responses import JSONResponse

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool

from infra.logutil.context import *

logger = logging.getLogger('log_util')

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
            self, request: Request, call_next: Callable
    ):
        extend_log_ctx_tags({LOG_CORRELATION_ID:str(uuid.uuid4())})
        body = await request.body()
        request._body = body
        body_str = body.decode("utf-8")
        logging.info(f"Request: {request.method} {request.url} \n Body: {json.dumps(body_str)}")
        try:
            response = await call_next(request)
            response_body = [chunk async for chunk in response.body_iterator]
            logging.info(f"Response: {response.status_code} \n Body: {response_body[0].decode()}")
            response.body_iterator = iterate_in_threadpool(iter(response_body))
            return response
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return JSONResponse(status_code=500, content={"error": "An error occurred"})
