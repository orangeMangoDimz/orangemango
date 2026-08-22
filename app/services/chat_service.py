from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from time import monotonic
from typing import Any
from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config.const.api_res import (
    CHAT_STREAM_ERROR,
    MESSAGE_STATUS_COMPLETED,
    MESSAGE_STATUS_FAILED,
)
from app.data.schema.response import AcceptedMessageResponse
from app.db.session import Database, postgres_checkpointer_url
from app.graph.chatbot.graph import build_graph
from app.logger import log_exception, logger
from app.models.chat_model import ChatConfigurationError, ChatModel
from app.repositories.chat_repository import (
    ChatEvent,
    ChatPersistenceError,
    ChatRepository,
    ChatRequestBusyError,
    ThreadNotFoundError,
)
from app.repositories.cv_repository import CvRepository
from app.services.cv_service import CvService


class ChatThreadBusyError(RuntimeError):
    """Raised when a thread already has an active graph run."""


class ChatThreadNotFoundError(LookupError):
    """Raised when an event stream references an unknown thread."""


def _chunk_text(chunk: Any) -> str:
    content: Any = getattr(chunk, "content", "")
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
        provider: str,
        model_name: str,
        chat_model: ChatModel | None = None,
        cv_service: CvService | None = None,
    ) -> None:
        if not provider.strip():
            raise ChatConfigurationError("Chat provider is not configured")
        if not model_name.strip():
            raise ChatConfigurationError("Chat model name is not configured")

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
    def from_env(cls) -> ChatService:
        model: ChatModel = ChatModel.from_env()
        database: Database = Database.from_env()
        repository: ChatRepository = ChatRepository(
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
            # Prevent concurrent startup calls from creating duplicate graphs.
            if self._repository.graph is not None:
                return

            stack: AsyncExitStack = AsyncExitStack()
            try:
                # Open and prepare the database-backed checkpointer.
                checkpointer: AsyncPostgresSaver = await stack.enter_async_context(
                    AsyncPostgresSaver.from_conn_string(postgres_checkpointer_url())
                )
                await checkpointer.setup()

                # Build the graph with durable conversation history.
                self._repository.graph = build_graph(
                    checkpointer=checkpointer,
                    chat_model=self._chat_model,
                )
                self._checkpointer_stack = stack
            except Exception as exc:
                # Release resources if startup fails partway through.
                await stack.aclose()
                raise ChatPersistenceError(
                    "Unable to initialize durable chat history"
                ) from exc

    async def accept_message(
        self,
        thread_id: str,
        message: str,
    ) -> AcceptedMessageResponse:
        # Comppile the graph
        await self.startup()
        try:
            # Generate request ID
            request_id: UUID = await self._repository.begin_run(
                thread_id,
                message=message,
                provider=self._provider,
                model=self._model_name,
            )
        except ChatRequestBusyError as exc:
            raise ChatThreadBusyError(thread_id) from exc

        try:
            # Call the LLM graph, but as async
            task: asyncio.Task[None] = asyncio.create_task(
                self._run_graph(thread_id, message, request_id)
            )
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
        response_persisted: bool = False
        started_at: float = monotonic()
        try:
            # Mark the request as running
            await self._repository.mark_processing(request_id)
            # Build the input for the graph
            graph_input: dict[str, Any] = await self._build_graph_input(
                thread_id,
                message,
            )

            # Run the graph and forward its output to SSE subscribers
            response_parts = await self._stream_graph_and_publish_events(
                thread_id,
                request_id,
                graph_input,
            )

            # Save the complete response after streaming finishes
            await self._persist_completed_response(
                request_id,
                response_parts,
                started_at,
            )
            response_persisted = True
            # Tell subscribers that the run is complete
            await self._publish_completed_event(
                thread_id,
                request_id,
            )
        except asyncio.CancelledError:
            await self._handle_cancelled_run(
                thread_id,
                request_id,
                response_parts,
                response_persisted,
            )
            raise
        except Exception as exc:
            await self._handle_failed_run(
                thread_id,
                request_id,
                response_parts,
                response_persisted,
                exc,
            )
        finally:
            await self._cleanup_finished_run(thread_id)

    async def _build_graph_input(
        self,
        thread_id: str,
        message: str,
    ) -> dict[str, Any]:
        graph_input: dict[str, Any] = {
            "messages": [{"role": "user", "content": message}],
        }
        cv_context: dict[str, Any] | None = await self._load_thread_cv_context(
            thread_id
        )
        if cv_context is not None:
            # Add saved CV context when this thread has one
            graph_input.update(cv_context)
        else:
            # Give the graph an empty CV context when none is saved
            graph_input.update(
                {
                    "cv": {
                        "documents": [],
                        "needs_extraction": False,
                        "review": None,
                    }
                }
            )
        return graph_input

    async def _stream_graph_and_publish_events(
        self,
        thread_id: str,
        request_id: UUID,
        graph_input: dict[str, Any],
    ) -> list[str]:
        response_parts: list[str] = []
        # Call the LLM uisng existing graph with langgraph
        async for mode, payload in self._repository.graph.astream(
            graph_input,
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
            version="v1",
            durability="sync",
        ):
            if mode == "updates":
                # Consume backend state updates without sending them to SSE
                continue

            # Keep only final response text for message events
            content: str = self._extract_response_text(mode, payload)
            if content:
                response_parts.append(content)
                await self._publish_message_chunk(
                    thread_id,
                    request_id,
                    content,
                )
        return response_parts

    @staticmethod
    def _extract_response_text(mode: object, payload: object) -> str:
        if mode != "messages" or not isinstance(payload, tuple):
            return ""
        if len(payload) != 2:
            return ""

        chunk, metadata = payload
        if not isinstance(metadata, dict):
            return ""
        if metadata.get("langgraph_node") != "respond":
            return ""
        return _chunk_text(chunk)

    async def _publish_message_chunk(
        self,
        thread_id: str,
        request_id: UUID,
        content: str,
    ) -> None:
        await self._repository.publish(
            thread_id,
            "message",
            {
                "thread_id": thread_id,
                "request_id": str(request_id),
                "content": content,
            },
        )

    async def _persist_completed_response(
        self,
        request_id: UUID,
        response_parts: list[str],
        started_at: float,
    ) -> None:
        await self._repository.complete_request(
            request_id,
            content="".join(response_parts),
            latency_ms=max(0, round((monotonic() - started_at) * 1000)),
        )

    async def _publish_completed_event(
        self,
        thread_id: str,
        request_id: UUID,
    ) -> None:
        await self._repository.publish(
            thread_id,
            "done",
            {
                "thread_id": thread_id,
                "request_id": str(request_id),
                "status": MESSAGE_STATUS_COMPLETED,
            },
        )

    async def _handle_cancelled_run(
        self,
        thread_id: str,
        request_id: UUID,
        response_parts: list[str],
        response_persisted: bool,
    ) -> None:
        if response_persisted:
            return

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

    async def _handle_failed_run(
        self,
        thread_id: str,
        request_id: UUID,
        response_parts: list[str],
        response_persisted: bool,
        exc: Exception,
    ) -> None:
        log_exception(
            "Chat graph run failed",
            exc=exc,
            thread_id=thread_id,
            request_id=str(request_id),
        )
        if not response_persisted:
            # Save any partial response before reporting failure
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
            # Tell subscribers that the run failed
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

    async def _cleanup_finished_run(self, thread_id: str) -> None:
        # Release the thread and remove the finished task
        await self._repository.end_run(thread_id)
        current_task: asyncio.Task[None] | None = asyncio.current_task()
        if self._tasks.get(thread_id) is current_task:
            self._tasks.pop(thread_id, None)

    async def _load_thread_cv_context(
        self,
        thread_id: str,
    ) -> dict[str, Any] | None:
        if self._cv_service is None:
            return None
        return await self._cv_service.load_latest_valid_thread_context(thread_id)

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
