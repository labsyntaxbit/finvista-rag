"""LLM response generation with RAG context and citations."""

from typing import List, Dict, Any

import requests
from openai import OpenAI

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are FinVista Capital's Enterprise Financial Intelligence Assistant.
You help financial analysts and consultants by answering questions based on enterprise documents
including financial reports, annual reports, analyst reports, compliance manuals, and market research.

Rules:
1. Answer ONLY based on the provided context. If the context does not contain enough information, say so clearly.
2. Be precise, professional, and concise in your financial analysis.
3. Reference specific data points from the context when available.
4. Do not fabricate numbers, dates, or facts not present in the context.
5. Maintain a professional tone suitable for corporate financial consulting."""


class ResponseGenerator:
    """Generates grounded AI responses using retrieved context."""

    def __init__(self):
        self.provider = settings.llm_provider.lower()

    def generate(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append(
            {
                "role": "user",
                "content": f"""Context from enterprise documents:
---
{context}
---

Question: {query}

Provide a detailed, context-grounded answer. Reference specific information from the documents.""",
            }
        )

        if self.provider == "openai":
            return self._generate_openai(messages)
        if self.provider == "ollama":
            return self._generate_ollama(messages)
        return self._generate_mock(query, context)

    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""

    def _generate_ollama(self, messages: List[Dict[str, str]]) -> str:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={"model": settings.ollama_model, "messages": messages, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def _generate_mock(self, query: str, context: str) -> str:
        """Fallback generator for demo/testing without API keys."""
        logger.info("Using mock LLM provider (set LLM_PROVIDER=openai for production)")
        context_preview = context[:500] + "..." if len(context) > 500 else context
        return (
            f"**FinVista Intelligence Response** (Mock Mode)\n\n"
            f"Based on the retrieved enterprise documents, here is the analysis for your query: "
            f"*{query}*\n\n"
            f"The relevant context from indexed financial documents indicates the following key points:\n\n"
            f"{context_preview}\n\n"
            f"---\n"
            f"*Note: Running in mock mode. Set `LLM_PROVIDER=openai` and provide `OPENAI_API_KEY` "
            f"in your `.env` file for full AI-generated responses.*"
        )
