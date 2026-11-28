import time
import logging

from langchain.agents.middleware import AgentMiddleware

logger = logging.getLogger(name="")

class LoggingMiddleware(AgentMiddleware):
    pass