# Chat Database Schema

The database stores one API chat request and its final LLM response. Streaming
SSE tokens remain in memory and are not persisted individually.

```mermaid
erDiagram
    CHAT_THREADS ||--o{ CHAT_REQUESTS : contains
    CHAT_REQUESTS ||--o| CHAT_RESPONSES : produces

    CHAT_THREADS {
        varchar_128 id PK
        varchar status
        timestamptz created_at
        timestamptz updated_at
    }

    CHAT_REQUESTS {
        uuid id PK
        varchar_128 thread_id FK
        text message
        varchar provider
        varchar model
        varchar status
        timestamptz created_at
        timestamptz started_at
        timestamptz finished_at
        varchar error_code
        text error_message
    }

    CHAT_RESPONSES {
        uuid id PK
        uuid request_id FK UK
        text content
        varchar status
        varchar finish_reason
        integer input_tokens
        integer output_tokens
        integer total_tokens
        integer latency_ms
        varchar provider_request_id
        timestamptz created_at
        text error_message
    }
```

## Tables

### `chat_threads`

Stores the client-provided conversation identifier.

- `id`: `VARCHAR(128)` primary key
- `status`: `active` or `archived`
- `created_at`, `updated_at`: UTC timestamps

### `chat_requests`

Stores each `POST /message` request and its processing lifecycle.

- `id`: application-generated UUID
- `thread_id`: foreign key to `chat_threads.id`
- `message`: validated user message, maximum 10,000 characters
- `provider`, `model`: LLM configuration used for the run
- `status`: `accepted`, `processing`, `completed`, `failed`, or `cancelled`
- `error_message`: sanitized internal failure detail

### `chat_responses`

Stores the final assembled LLM response for a request.

- `request_id`: unique foreign key to `chat_requests.id`
- `content`: final response text; may be empty for failed responses
- `status`: `completed`, `partial`, or `failed`
- Token counts, latency, finish reason, and provider request ID are optional
  provider metadata.

## Constraints and indexes

```sql
CREATE INDEX ix_chat_requests_thread_created
    ON chat_requests (thread_id, created_at DESC);

CREATE INDEX ix_chat_threads_updated
    ON chat_threads (updated_at DESC);

CREATE UNIQUE INDEX uq_active_chat_request_per_thread
    ON chat_requests (thread_id)
    WHERE status IN ('accepted', 'processing');
```

The partial unique index prevents concurrent active requests for the same
thread, including when multiple API instances are running.
