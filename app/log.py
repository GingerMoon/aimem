import logging
from logging.config import dictConfig
import json

class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings after parsing the LogRecord.

    @param dict fmt_dict: Key: logging format attribute pairs. Defaults to {"message": "message"}.
    @param str time_format: time.strftime() format string. Default: "%Y-%m-%dT%H:%M:%S"
    @param str msec_format: Microsecond formatting. Appended at the end. Default: "%s.%03dZ"
    """
    def __init__(self, fmt_dict: dict = None, time_format: str = "%Y-%m-%dT%H:%M:%S", msec_format: str = "%s.%03dZ"):
        self.fmt_dict = fmt_dict if fmt_dict is not None else {"level": "levelname",
                                                               # "processName": "processName",
                                                               # "processID": "process",
                                                               # "threadName": "threadName",
                                                               # "threadID": "thread",
                                                               "timestamp": "asctime",
                                                               # "loggerName": "name",
                                                               "filename": "filename",
                                                               "funcName": "funcName",
                                                               "lineno": "lineno",
                                                               "correlation_id": "correlation_id",
                                                               "flow_name": "flow_name",
                                                               "message": "message",
                                                               }
        time_format = "%Y-%m-%dT%H:%M:%S"
        msec_format = """%s:%03d"""

        self.default_time_format = time_format
        self.default_msec_format = msec_format
        # self.fmt = '%(asctime)s.%(msecs)03d'
        self.datefmt = None

    def usesTime(self) -> bool:
        """
        Overwritten to look for the attribute in the format dict values instead of the fmt string.
        """
        return "asctime" in self.fmt_dict.values()

    def formatMessage(self, record) -> dict:
        """
        Overwritten to return a dictionary of the relevant LogRecord attributes instead of a string.
        KeyError is raised if an unknown attribute is provided in the fmt_dict.
        """
        result = {}
        for fmt_key, fmt_val in self.fmt_dict.items():
            if fmt_val in record.__dict__:
                result[fmt_key] = record.__dict__[fmt_val]
        return result

    def format(self, record) -> str:
        """
        Mostly the same as the parent's class method, the difference being that a dict is manipulated and dumped as JSON
        instead of a string.
        """
        record.message = record.getMessage()

        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        message_dict = self.formatMessage(record)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            message_dict["exc_info"] = record.exc_text

        if record.stack_info:
            message_dict["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(message_dict, default=str)

def configure_logging() -> None:
    dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'filters': {  # correlation ID filter must be added here to make the %(correlation_id)s formatter work
                'correlation_id': {
                    # '()': 'asgi_correlation_id.CorrelationIdFilter',
                    '()': 'infra.logutil.filter.CtxTagsFilter',
                    'uuid_length': 32,
                    'default_value': '-',
                },
            },
            'formatters': {
                'default': {
                    'class': 'logging.Formatter',
                    'datefmt': '%H:%M:%S',
                    # formatter decides how our console logs look, and what info is included.
                    # adding %(correlation_id)s to this format is what make correlation IDs appear in our logs
                    'format': '%(levelname)s:\t\b%(asctime)s %(name)s:%(lineno)d [%(correlation_id)s] %(message)s',
                },
                'json': {
                    'class': 'log.JsonFormatter',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    # Filter must be declared in the handler, otherwise it won't be included
                    'filters': ['correlation_id'],
                    'formatter': 'default',
                    'level': 'WARN',
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',  # Log to file
                    'filename': '../aimem.log.jsonl',  # Log file name
                    'filters': ['correlation_id'],
                    'formatter': 'json',
                    'maxBytes': 10 * 1024 * 1024,
                    'backupCount': 10,
                    'level': 'INFO',
                },
            },
            # Loggers can be specified to set the log-level to log, and which handlers to use
            'loggers': {
            #     # project logger
            #     'app': {'handlers': ['file'], 'level': 'DEBUG', 'propagate': True},
            #     # third-party package loggers
            #     'databases': {'handlers': ['console'], 'level': 'WARNING'},
                'httpx': {'handlers': ['console'], 'level': 'WARN'},
            #     'asgi_correlation_id': {'handlers': ['console'], 'level': 'WARNING'},
            },
            'root': {
                'handlers': ['file', 'console'],
                'level': 'DEBUG',
            },
        }
    )
