
from langchain.chat_models import init_chat_model
from app.core.config import Settings

from langchain_core.language_models import BaseChatModel


def get_chat_model(settings:Settings) -> BaseChatModel:
  return init_chat_model(
        model_provider=settings.chat_provider,
        model=settings.chat_model,
        api_key=settings.chat_api_key,
            # temperature=self.settings.chat_temperature,
        )

def get_fallback_model(settings: Settings) -> None:
  pass