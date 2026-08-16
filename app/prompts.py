"""Shared instructions for user-facing language-model responses."""

DEFAULT_USER_RESPONSE_STYLE = """
Default response style:
- Put one complete thought per line, with a blank line between distinct topics.
- Keep paragraphs short; do not combine several recommendations into dense prose.
- Use bullets when they make multiple related items easier to scan.
- This is a default only. The user's explicit formatting request takes priority.
""".strip()
