# import logging
# import os

# def setup_logger(log_file="logs/agent.log"):
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure log directory exists

#     logger = logging.getLogger("AgentLogger")

#     # ✅ Prevent duplicate handlers
#     if not logger.hasHandlers():
#         logger.setLevel(logging.INFO)

#         file_handler = logging.FileHandler(log_file, encoding="utf-8")  # ✅ Set UTF-8 encoding
#         formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#         file_handler.setFormatter(formatter)

#         logger.addHandler(file_handler)

#     return logger

import logging
import os

def setup_logger(log_file="logs/agent.log"):
    """Setup a singleton logger that prevents duplicate handlers."""

    # ✅ Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    # ✅ Define a global logger
    logger = logging.getLogger("AgentLogger")

    # ✅ Remove any existing handlers (Prevents duplication)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # ✅ Create file handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # ✅ Add file handler
    logger.addHandler(file_handler)

    # ✅ Console handler (Optional for debugging)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
