from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession

from app.config.const.chat import MAX_EVENT_HISTORY
from app.db.models import ChatMessage as ChatMessageRecord
from app.db.models import ChatRequest as ChatRequestRecord
from app.db.models import ChatResponse as ChatResponseRecord
from app.db.models import (
    ChatThread,
)
from app.db.models import CvExtraction as CvExtractionRecord

EventType = Literal["state", "message", "cv_extraction", "error", "done"]


class ThreadNotFoundError(LookupError):
    """Raised when repository operations reference an unknown thread."""


class ChatRequestBusyError(RuntimeError):
    """Raised when the database already has an active request for a thread."""


class ChatPersistenceError(RuntimeError):
    """Raised when a chat request cannot be persisted."""


@dataclass(frozen=True)
class ChatEvent:
    event_id: int
    event_type: EventType
    data: dict[str, Any]


@dataclass
class ThreadRecord:
    events: deque[ChatEvent] = field(
        default_factory=lambda: deque(maxlen=MAX_EVENT_HISTORY)
    )
    subscribers: set[asyncio.Queue[ChatEvent]] = field(default_factory=set)
    active: bool = False
    next_event_id: int = 1
    request_id: UUID | None = None


class ChatRepository:
    """Persist chat runs/transcripts and coordinate live in-process events."""

    def __init__(
        self,
        *,
        graph: Any,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.graph = graph
        self._session_factory = session_factory
        self._threads: dict[str, ThreadRecord] = {}
        self._lock = asyncio.Lock()

    async def begin_run(
        self,
        thread_id: str,
        *,
        message: str,
        provider: str,
        model: str,
    ) -> UUID:
        async with self._lock:
            record = self._threads.setdefault(thread_id, ThreadRecord())
            if record.active:
                raise ChatRequestBusyError(thread_id)

            now = datetime.now(UTC)
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        thread = await session.get(ChatThread, thread_id)
                        if thread is None:
                            session.add(
                                ChatThread(
                                    id=thread_id,
                                    created_at=now,
                                    updated_at=now,
                                )
                            )
                            await session.flush()
                        else:
                            thread.status = "active"
                            thread.updated_at = now

                        active_cv = await session.execute(
                            select(CvExtractionRecord.id)
                            .where(
                                CvExtractionRecord.thread_id == thread_id,
                                CvExtractionRecord.status == "processing",
                            )
                            .limit(1)
                        )
                        if active_cv.scalar_one_or_none() is not None:
                            raise ChatRequestBusyError(thread_id)

                        request = ChatRequestRecord(
                            thread_id=thread_id,
                            message=message,
                            provider=provider,
                            model=model,
                            status="accepted",
                            created_at=now,
                        )
                        session.add(request)
                        await session.flush()
                        next_sequence = await self._next_message_sequence(
                            session,
                            thread_id,
                        )
                        session.add(
                            ChatMessageRecord(
                                thread_id=thread_id,
                                request_id=request.id,
                                role="user",
                                content=message,
                                sequence=next_sequence,
                                created_at=now,
                            )
                        )
            except ChatRequestBusyError:
                raise
            except IntegrityError as exc:
                if "uq_active_chat_request_per_thread" in str(exc):
                    raise ChatRequestBusyError(thread_id) from exc
                raise ChatPersistenceError("Unable to persist chat request") from exc
            except SQLAlchemyError as exc:
                raise ChatPersistenceError("Unable to persist chat request") from exc

            record.events.clear()
            record.active = True
            record.request_id = request.id
            return request.id

    async def begin_activity(self, thread_id: str) -> None:
        """Reserve a thread for a non-chat operation such as CV extraction."""
        async with self._lock:
            record = self._threads.setdefault(thread_id, ThreadRecord())
            if record.active:
                raise ChatRequestBusyError(thread_id)

            now = datetime.now(UTC)
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        thread = await session.get(ChatThread, thread_id)
                        if thread is None:
                            session.add(
                                ChatThread(
                                    id=thread_id,
                                    created_at=now,
                                    updated_at=now,
                                )
                            )
                            await session.flush()
                        else:
                            thread.status = "active"
                            thread.updated_at = now

                        active_chat = await session.execute(
                            select(ChatRequestRecord.id)
                            .where(
                                ChatRequestRecord.thread_id == thread_id,
                                ChatRequestRecord.status.in_(
                                    ("accepted", "processing")
                                ),
                            )
                            .limit(1)
                        )
                        if active_chat.scalar_one_or_none() is not None:
                            raise ChatRequestBusyError(thread_id)

                        active_cv = await session.execute(
                            select(CvExtractionRecord.id)
                            .where(
                                CvExtractionRecord.thread_id == thread_id,
                                CvExtractionRecord.status == "processing",
                            )
                            .limit(1)
                        )
                        if active_cv.scalar_one_or_none() is not None:
                            raise ChatRequestBusyError(thread_id)
            except ChatRequestBusyError:
                raise
            except SQLAlchemyError as exc:
                raise ChatPersistenceError("Unable to reserve chat thread") from exc

            record.events.clear()
            record.active = True
            record.request_id = None

    async def has_thread(self, thread_id: str) -> bool:
        async with self._lock:
            if thread_id in self._threads:
                return True

        try:
            async with self._session_factory() as session:
                return await session.get(ChatThread, thread_id) is not None
        except SQLAlchemyError as exc:
            raise ChatPersistenceError("Unable to load chat thread") from exc

    async def history(self, thread_id: str) -> list[dict[str, str]]:
        """Return the ordered transcript for internal history/replay use."""
        try:
            async with self._session_factory() as session:
                result = await session.execute(
                    select(ChatMessageRecord)
                    .where(ChatMessageRecord.thread_id == thread_id)
                    .order_by(ChatMessageRecord.sequence)
                )
                return [
                    {"role": message.role, "content": message.content}
                    for message in result.scalars().all()
                ]
        except SQLAlchemyError as exc:
            raise ChatPersistenceError("Unable to load chat history") from exc

    async def publish(
        self,
        thread_id: str,
        event_type: EventType,
        data: dict[str, Any],
    ) -> ChatEvent:
        async with self._lock:
            record = self._threads.get(thread_id)
            if record is None:
                raise ThreadNotFoundError(thread_id)

            event = ChatEvent(record.next_event_id, event_type, data)
            record.next_event_id += 1
            record.events.append(event)
            if event_type == "done":
                record.active = False
            for queue in record.subscribers:
                queue.put_nowait(event)
            return event

    async def end_run(self, thread_id: str) -> None:
        async with self._lock:
            record = self._threads.get(thread_id)
            if record is not None:
                record.active = False
                record.request_id = None

    async def mark_processing(self, request_id: UUID) -> None:
        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    request = await session.get(ChatRequestRecord, request_id)
                    if request is None:
                        raise ChatPersistenceError("Chat request was not found")
                    request.status = "processing"
                    request.started_at = now
        except ChatPersistenceError:
            raise
        except SQLAlchemyError as exc:
            raise ChatPersistenceError("Unable to update chat request") from exc

    async def complete_request(
        self,
        request_id: UUID,
        *,
        content: str,
        latency_ms: int | None,
    ) -> None:
        await self._finish_request(
            request_id,
            request_status="completed",
            response_status="completed",
            content=content,
            error_message=None,
            latency_ms=latency_ms,
        )

    async def fail_request(
        self,
        request_id: UUID,
        *,
        content: str,
        error_message: str,
        request_status: str = "failed",
    ) -> None:
        response_status = "partial" if content else "failed"
        await self._finish_request(
            request_id,
            request_status=request_status,
            response_status=response_status,
            content=content or None,
            error_message=error_message,
            latency_ms=None,
        )

    async def _finish_request(
        self,
        request_id: UUID,
        *,
        request_status: str,
        response_status: str,
        content: str | None,
        error_message: str | None,
        latency_ms: int | None,
    ) -> None:
        now = datetime.now(UTC)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    request = await session.get(ChatRequestRecord, request_id)
                    if request is None:
                        raise ChatPersistenceError("Chat request was not found")

                    request.status = request_status
                    request.finished_at = now
                    request.error_message = error_message
                    session.add(
                        ChatResponseRecord(
                            request_id=request_id,
                            content=content,
                            status=response_status,
                            latency_ms=latency_ms,
                            error_message=error_message,
                        )
                    )

                    if content and content.strip():
                        next_sequence = await self._next_message_sequence(
                            session,
                            request.thread_id,
                        )
                        session.add(
                            ChatMessageRecord(
                                thread_id=request.thread_id,
                                request_id=request_id,
                                role="assistant",
                                content=content,
                                sequence=next_sequence,
                                created_at=now,
                            )
                        )

                    thread = await session.get(ChatThread, request.thread_id)
                    if thread is not None:
                        thread.updated_at = now
        except ChatPersistenceError:
            raise
        except IntegrityError as exc:
            raise ChatPersistenceError("Unable to persist chat response") from exc
        except SQLAlchemyError as exc:
            raise ChatPersistenceError("Unable to persist chat response") from exc

    async def subscribe(
        self,
        thread_id: str,
        *,
        last_event_id: int | None = None,
    ) -> AsyncIterator[ChatEvent]:
        async with self._lock:
            record = self._threads.get(thread_id)
            if record is not None:
                replay = [
                    event
                    for event in record.events
                    if last_event_id is None or event.event_id > last_event_id
                ]
                queue: asyncio.Queue[ChatEvent] = asyncio.Queue()
                record.subscribers.add(queue)
                active = record.active
                finished = (
                    not active
                    and bool(record.events)
                    and record.events[-1].event_type == "done"
                )

        if record is None:
            persisted_history = await self.history(thread_id)
            if not persisted_history:
                raise ThreadNotFoundError(thread_id)

            event_id = 1
            for message in persisted_history:
                if message["role"] != "assistant":
                    continue
                if last_event_id is None or event_id > last_event_id:
                    yield ChatEvent(
                        event_id,
                        "message",
                        {
                            "thread_id": thread_id,
                            "content": message["content"],
                        },
                    )
                event_id += 1
                if last_event_id is None or event_id > last_event_id:
                    yield ChatEvent(
                        event_id,
                        "done",
                        {
                            "thread_id": thread_id,
                            "status": "completed",
                        },
                    )
                event_id += 1
            return

        try:
            for event in replay:
                yield event

            if finished:
                return

            while True:
                event = await queue.get()
                yield event
                if event.event_type == "done":
                    return
        finally:
            async with self._lock:
                record = self._threads.get(thread_id)
                if record is not None:
                    record.subscribers.discard(queue)

    @staticmethod
    async def _next_message_sequence(
        session: AsyncSession,
        thread_id: str,
    ) -> int:
        result = await session.execute(
            select(
                func.coalesce(
                    func.max(ChatMessageRecord.sequence),
                    -1,
                )
            ).where(ChatMessageRecord.thread_id == thread_id)
        )
        return int(result.scalar_one()) + 1
