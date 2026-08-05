"""Helpers for constructing chat models used by the project."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel


def build_chat_model(provider: str, model_name: str, temperature: float) -> BaseChatModel:
    """Build a chat model for the requested provider."""
    normalized_provider = provider.strip().lower()

    if normalized_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=temperature)

    if normalized_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name, temperature=temperature)

    raise ValueError("provider must be one of: ollama, openai")
