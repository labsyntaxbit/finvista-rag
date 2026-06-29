"""Multi-turn conversation memory management."""

from typing import List, Dict, Any
from collections import deque

from app.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """Maintains conversation history for multi-turn interactions."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: deque = deque(maxlen=max_turns * 2)

    def add_user_message(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str, citations: List[Dict[str, Any]] | None = None) -> None:
        self.history.append(
            {
                "role": "assistant",
                "content": content,
                "citations": citations or [],
            }
        )

    def get_history(self) -> List[Dict[str, Any]]:
        return list(self.history)

    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        return [{"role": m["role"], "content": m["content"]} for m in self.history]

    def clear(self) -> None:
        self.history.clear()
        logger.info("Conversation history cleared")

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.history if m["role"] == "user")
