from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from time import monotonic
from typing import Any
from uuid import UUID

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import BaseModel

from app.config.const.api_res import (
    CHAT_STREAM_ERROR,
    MESSAGE_STATUS_COMPLETED,
    MESSAGE_STATUS_FAILED,
)
from app.data.schema.response import AcceptedMessageResponse
from app.db.session import Database, postgres_checkpointer_url
from app.logger import log_exception, logger
from app.models.chat_model import ChatModel
from app.repositories.chat_repository import (
    ChatEvent,
    ChatPersistenceError,
    ChatRequestBusyError,
    ChatRepository,
    ThreadNotFoundError,
)
from app.repositories.cv_repository import CvRepository
from app.services.cv_service import CvService

_MAX_SERIALIZED_TEXT = 4000
_MAX_SERIALIZED_ITEMS = 64
_MAX_SERIALIZATION_DEPTH = 8


class ChatThreadBusyError(RuntimeError):
    """Raised when a thread already has an active graph run."""


class ChatThreadNotFoundError(LookupError):
    """Raised when an event stream references an unknown thread."""


def _bounded_text(value: Any) -> str:
    text = str(value)
    if len(text) <= _MAX_SERIALIZED_TEXT:
        return text
    return text[: _MAX_SERIALIZED_TEXT - 1].rstrip() + "…"


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_SERIALIZATION_DEPTH:
        return _bounded_text(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, BaseMessage):
        return {
            "type": value.type,
            "content": _json_safe(value.content, depth=depth + 1),
        }
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"), depth=depth + 1)
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_SERIALIZED_ITEMS]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _json_safe(item, depth=depth + 1)
            for item in list(value)[:_MAX_SERIALIZED_ITEMS]
        ]
    return _bounded_text(value)


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
    return "".join(text_parts)


class ChatService:
    """Coordinate thread-scoped LangGraph runs and public chat events."""

    def __init__(
        self,
        repository: ChatRepository,
        *,
        database: Database | None = None,
        provider: str = "openai",
        model_name: str = "unknown",
        chat_model: ChatModel | None = None,
        cv_service: CvService | None = None,
    ) -> None:
        self._repository = repository
        self._database = database
        self._provider = provider
        self._model_name = model_name
        self._chat_model = chat_model
        self._cv_service = cv_service
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._startup_lock = asyncio.Lock()
        self._checkpointer_stack: AsyncExitStack | None = None

    @classmethod
    def from_environment(cls) -> ChatService:
        model = ChatModel.from_env()
        database = Database.from_environment()
        repository = ChatRepository(
            graph=None,
            session_factory=database.session_factory,
        )
        return cls(
            repository,
            database=database,
            provider=model.provider,
            model_name=model.model_name,
            chat_model=model,
            cv_service=CvService(
                event_repository=repository,
                repository=CvRepository(
                    session_factory=database.session_factory,
                ),
            ),
        )

    @property
    def cv_service(self) -> CvService:
        if self._cv_service is None:
            raise ChatPersistenceError("CV service is not configured")
        return self._cv_service

    async def startup(self) -> None:
        """Initialize the durable LangGraph checkpointer once per service."""
        if self._repository.graph is not None:
            return
        if self._chat_model is None:
            raise ChatPersistenceError("Chat model is not configured")

        async with self._startup_lock:
            if self._repository.graph is not None:
                return

            stack = AsyncExitStack()
            try:
                checkpointer = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(postgres_checkpointer_url())
                )
                await checkpointer.setup()

                from studio.chatbot.graph import build_graph

                self._repository.graph = build_graph(
                    checkpointer=checkpointer,
                    chat_model=self._chat_model,
                )
                self._checkpointer_stack = stack
            except Exception as exc:
                await stack.aclose()
                raise ChatPersistenceError(
                    "Unable to initialize durable chat history"
                ) from exc

    async def accept_message(
        self,
        thread_id: str,
        message: str,
    ) -> AcceptedMessageResponse:
        await self.startup()
        try:
            request_id = await self._repository.begin_run(
                thread_id,
                message=message,
                provider=self._provider,
                model=self._model_name,
            )
        except ChatRequestBusyError as exc:
            raise ChatThreadBusyError(thread_id) from exc

        try:
            task = asyncio.create_task(self._run_graph(thread_id, message, request_id))
        except BaseException:
            await self._repository.end_run(thread_id)
            try:
                await self._repository.fail_request(
                    request_id,
                    content="",
                    error_message=CHAT_STREAM_ERROR,
                )
            except ChatPersistenceError:
                log_exception(
                    "Unable to persist failed chat request",
                    exc_info=True,
                    thread_id=thread_id,
                    request_id=str(request_id),
                )
            raise

        self._tasks[thread_id] = task
        logger.info(
            "Accepted chat message thread_id={thread_id} request_id={request_id}",
            thread_id=thread_id,
            request_id=str(request_id),
        )
        return AcceptedMessageResponse(
            thread_id=thread_id,
            request_id=request_id,
            status="accepted",
        )

    async def ensure_thread(self, thread_id: str) -> None:
        if not await self._repository.has_thread(thread_id):
            raise ChatThreadNotFoundError(thread_id)

    async def events(
        self,
        thread_id: str,
        last_event_id: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        try:
            async for event in self._repository.subscribe(
                thread_id,
                last_event_id=last_event_id,
            ):
                yield event
        except ThreadNotFoundError as exc:
            raise ChatThreadNotFoundError(thread_id) from exc

    async def _run_graph(
        self,
        thread_id: str,
        message: str,
        request_id: UUID,
    ) -> None:
        response_parts: list[str] = []
        response_persisted = False
        started_at = monotonic()
        try:
            await self._repository.mark_processing(request_id)
            config = {"configurable": {"thread_id": thread_id}}
            graph_input = {
                "messages": [{"role": "user", "content": message}],
            }

            async for mode, payload in self._repository.graph.astream(
                graph_input,
                config=config,
                stream_mode=["messages", "updates"],
                version="v1",
                durability="sync",
            ):
                if mode == "updates":
                    if not isinstance(payload, dict):
                        continue
                    for node_name, update in payload.items():
                        await self._repository.publish(
                            thread_id,
                            "state",
                            {
                                "thread_id": thread_id,
                                "request_id": str(request_id),
                                "node": node_name,
                                "data": _json_safe(update),
                            },
                        )
                    continue

                if mode != "messages" or not isinstance(payload, tuple):
                    continue
                if len(payload) != 2:
                    continue

                chunk, metadata = payload
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("langgraph_node") != "respond":
                    continue

                content = _chunk_text(chunk)
                if content:
                    response_parts.append(content)
                    await self._repository.publish(
                        thread_id,
                        "message",
                        {
                            "thread_id": thread_id,
                            "request_id": str(request_id),
                            "content": content,
                        },
                    )

            await self._repository.complete_request(
                request_id,
                content="".join(response_parts),
                latency_ms=max(0, round((monotonic() - started_at) * 1000)),
            )
            response_persisted = True
            await self._repository.publish(
                thread_id,
                "done",
                {
                    "thread_id": thread_id,
                    "request_id": str(request_id),
                    "status": MESSAGE_STATUS_COMPLETED,
                },
            )
        except asyncio.CancelledError:
            if not response_persisted:
                try:
                    await self._repository.fail_request(
                        request_id,
                        content="".join(response_parts),
                        error_message=CHAT_STREAM_ERROR,
                        request_status="cancelled",
                    )
                except ChatPersistenceError:
                    log_exception(
                        "Unable to persist cancelled chat request",
                        exc_info=True,
                        thread_id=thread_id,
                        request_id=str(request_id),
                    )
            raise
        except Exception as exc:
            log_exception(
                "Chat graph run failed",
                exc=exc,
                thread_id=thread_id,
                request_id=str(request_id),
            )
            if not response_persisted:
                try:
                    await self._repository.fail_request(
                        request_id,
                        content="".join(response_parts),
                        error_message=CHAT_STREAM_ERROR,
                    )
                except ChatPersistenceError:
                    log_exception(
                        "Unable to persist failed chat request",
                        exc_info=True,
                        thread_id=thread_id,
                        request_id=str(request_id),
                    )
            try:
                await self._repository.publish(
                    thread_id,
                    "error",
                    {
                        "thread_id": thread_id,
                        "request_id": str(request_id),
                        "message": CHAT_STREAM_ERROR,
                    },
                )
                await self._repository.publish(
                    thread_id,
                    "done",
                    {
                        "thread_id": thread_id,
                        "request_id": str(request_id),
                        "status": MESSAGE_STATUS_FAILED,
                    },
                )
            except ThreadNotFoundError as publish_exc:
                log_exception(
                    "Unable to publish chat graph failure",
                    exc=publish_exc,
                    thread_id=thread_id,
                    request_id=str(request_id),
                )
        finally:
            await self._repository.end_run(thread_id)
            current_task = asyncio.current_task()
            if self._tasks.get(thread_id) is current_task:
                self._tasks.pop(thread_id, None)

    async def shutdown(self) -> None:
        if self._cv_service is not None:
            await self._cv_service.shutdown()
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        if self._checkpointer_stack is not None:
            await self._checkpointer_stack.aclose()
            self._checkpointer_stack = None
        if self._database is not None:
            await self._database.dispose()
