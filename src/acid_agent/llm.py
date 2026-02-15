from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from acid_agent.config import AppConfig


def build_llm(config: AppConfig) -> BaseChatModel | None:
    if not config.openai_api_key:
        return None

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    os.environ.setdefault("OPENAI_API_KEY", config.openai_api_key)
    return ChatOpenAI(model=config.openai_model, temperature=0)
