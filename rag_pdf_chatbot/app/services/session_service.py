"""In-memory chat session/history management."""

import uuid
from datetime import datetime, timezone

from app.core.exceptions import SessionNotFoundError
from app.models.schemas import ChatTurn


class SessionService:
    """Stores per-session conversation history in memory.

    Single responsibility: session lifecycle and history bookkeeping only.
    Swappable for a Redis/DB-backed implementation later without touching
    the RAG orchestration logic, since callers only depend on this class's
    public methods.
    """

    def __init__(self, max_history_turns: int) -> None:
        """Initialize with the max number of user/assistant turn pairs kept per session."""
        self._sessions: dict[str, list[ChatTurn]] = {}
        self._max_history_turns = max_history_turns

    def create_session(self) -> str:
        """Create a new empty session and return its id."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        return session_id

    def ensure_session(self, session_id: str) -> str:
        """Return session_id if valid, otherwise raise SessionNotFoundError."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(f"Session '{session_id}' does not exist.")
        return session_id

    def get_history(self, session_id: str) -> list[ChatTurn]:
        """Return the stored turns for a session."""
        self.ensure_session(session_id)
        return self._sessions[session_id]

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a turn and trim history to the configured max length."""
        self._sessions[session_id].append(
            ChatTurn(role=role, content=content, timestamp=datetime.now(timezone.utc))
        )
        max_len = self._max_history_turns * 2
        if len(self._sessions[session_id]) > max_len:
            self._sessions[session_id] = self._sessions[session_id][-max_len:]

    def clear_session(self, session_id: str) -> None:
        """Delete a session's history."""
        self.ensure_session(session_id)
        del self._sessions[session_id]

    def session_exists(self, session_id: str) -> bool:
        """Return whether a session id is currently tracked."""
        return session_id in self._sessions
