import uuid
from contextvars import ContextVar
from typing import Optional

LOG_CORRELATION_ID = "correlation_id"

LOG_FLOWNAME = "flow_name"
LOG_FLOWNAME_VEC_MEM_ADD = "vec_mem_add"
LOG_FLOWNAME_VEC_MEM_SEARCH = "vec_mem_search"
LOG_FLOWNAME_GRAPH_MEM_ADD = "graph_mem_add"
LOG_FLOWNAME_GRAPH_MEM_SEARCH = "graph_mem_search"

log_ctx_tags = ContextVar('log_ctx_tags', default={})

def extend_log_ctx_tags(tags: dict):
    tags = {**log_ctx_tags.get(), **tags}
    # if LOG_CORRELATION_ID not in tags:
    #     tags[LOG_CORRELATION_ID] = str(uuid.uuid4())
    log_ctx_tags.set(tags)

def get_log_ctx_tags():
    tags = log_ctx_tags.get()
    return tags