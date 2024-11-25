from logging import Filter
from typing import TYPE_CHECKING, Optional

from infra.logutil.context import *

if TYPE_CHECKING:
    from logging import LogRecord

class CtxTagsFilter(Filter):
    """Logging filter to attached correlation IDs to log records"""

    def __init__(self, name: str = '', uuid_length: Optional[int] = None, default_value: Optional[str] = None):
        super().__init__(name=name)
        self.uuid_length = uuid_length
        self.default_value = default_value

    def filter(self, record: 'LogRecord') -> bool:
        """
        Attach a correlation ID to the log record.

        Since the correlation ID is defined in the middleware layer, any
        log generated from a request after this point can easily be searched
        for, if the correlation ID is added to the message, or included as
        metadata.
        """
        tags = log_ctx_tags.get()
        for key in tags:
            value = tags[key]
            if key == LOG_CORRELATION_ID:
                value = value.replace('-', '')
            record.__dict__[key] = value
        return True


