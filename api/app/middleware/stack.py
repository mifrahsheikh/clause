from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)

from app.core.config import Settings
from app.llm.models import get_chat_model, get_fallback_model
from app.middleware.observability import LoggingMiddleware


def default_stack(settings: Settings) -> list[AgentMiddleware]:
    stack = [
        ModelCallLimitMiddleware(
            run_limit=settings.max_model_call, exit_behavior="end"
        ),
        ToolCallLimitMiddleware(run_limit=settings.max_tool_call),
        SummarizationMiddleware(
            model=get_chat_model(settings),
            trigger=("tokens", settings.summary_trigger_tokens),
            keep=("messages", settings.summary_keep_messages),
        ),
        LoggingMiddleware(),
    ]
    fallback = get_fallback_model(settings)
    if fallback:
        stack.append(ModelFallbackMiddleware())
    return stack
