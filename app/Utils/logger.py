# app/utils/logger.py

import logging

def setup_logger(log_file="logs/agent.log"):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger("AgentLogger")
