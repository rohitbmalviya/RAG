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
            "file": record.filename,
            "function": record.funcName,
            "line": record.lineno,
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
        
        # File handler for production logs (with immediate flush!)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Capture ALL levels
        
        # Switch to JSON structured logs unless explicitly disabled via env
        if os.getenv("PLAIN_LOGS", "0") == "1":
            formatter: logging.Formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = JsonFormatter()
        
        file_handler.setFormatter(formatter)
        
        # Force immediate flush (no buffering!)
        file_handler.flush = lambda: file_handler.stream.flush() if file_handler.stream else None
        
        logger.addHandler(file_handler)
        
        # Optional: Add console handler ONLY if DEBUG_CONSOLE=1 environment variable is set
        if os.getenv("DEBUG_CONSOLE", "0") == "1":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # Get level from config if available, otherwise use parameter
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                config_level = config.get('logging', {}).get('level', level)
                level = config_level
    except:
        pass  # Use provided level if config read fails
    
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    logger.propagate = False
    
    # Ensure all handlers flush immediately after each log
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            original_emit = handler.emit
            def emit_with_flush(record):
                original_emit(record)
                handler.flush()
            handler.emit = emit_with_flush
    
    return logger