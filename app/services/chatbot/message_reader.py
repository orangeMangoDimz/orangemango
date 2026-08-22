"""Read text, role, and tool calls from heterogeneous LangChain messages."""

from __future__ import annotations

from typing import Any

from app.config.const.chatbot import PDF_UPLOAD_MARKER


class MessageReader:
    """Stateless accessors over dict-shaped or object-shaped messages."""

    @staticmethod
    def message_text(message: Any) -> str:
        if isinstance(message, dict):
            content: Any = message.get("content", "")
        else:
            content = getattr(message, "content", message)
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            return "".join(text_parts)
        return str(content)

    @staticmethod
    def normalize_user_turn_text(content: str) -> str:
        text: str = (content or "").strip()
        if not text:
            return ""
        return text.replace(PDF_UPLOAD_MARKER, "").strip()

    @staticmethod
    def message_role(message: Any) -> str:
        if isinstance(message, dict):
            explicit_role: str = str(message.get("role") or "").strip().lower()
            if explicit_role:
                return explicit_role
            msg_type: str = str(message.get("type") or "").strip().lower()
            if msg_type in {"human", "user"}:
                return "user"
            if msg_type in {"ai", "assistant"}:
                return "assistant"
            if msg_type == "tool":
                return "tool"
            return "user"
        msg_type = str(getattr(message, "type", "") or "").strip().lower()
        if msg_type == "tool":
            return "tool"
        return "user"

    @staticmethod
    def message_tool_calls(message: Any) -> Any:
        if isinstance(message, dict):
            return message.get("tool_calls")
        return getattr(message, "tool_calls", None)
