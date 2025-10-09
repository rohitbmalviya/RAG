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
        # Production: Log to file only (NO console output!)
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'app.log')
        
        # File handler for production logs
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        
        # Switch to JSON structured logs unless explicitly disabled via env
        if os.getenv("PLAIN_LOGS", "0") == "1":
            formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = JsonFormatter()
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Optional: Add console handler ONLY if DEBUG_CONSOLE=1 environment variable is set
        if os.getenv("DEBUG_CONSOLE", "0") == "1":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger