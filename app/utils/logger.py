import json
import logging
import os
import sys
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Avoid dumping args if message already formatted
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name if name else __name__)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        # Switch to JSON structured logs unless explicitly disabled via env
        if os.getenv("PLAIN_LOGS", "0") == "1":
            formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
