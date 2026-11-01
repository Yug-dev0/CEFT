"""Multi-turn conversational memory and query reformulation for Ceft.

Enables interactive follow-up questions in the Ask interface by maintaining
session history and contextualizing anaphoric pronouns (e.g., 'who did that?',
'which slide showed it?', 'did that work?') with entities and topics from
previous turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any
from uuid import uuid4

_FOLLOWUP_PRONOUNS = re.compile(
    r"\b(it|that|this|they|them|he|she|the fix|the issue|the problem|the change|the slide|the diagram|the outage|the meeting|who|why|how)\b",
    re.IGNORECASE,
)


@dataclass
class ChatTurn:
    turn_id: str
    query: str
    reformulated_query: str
    answer: str
    grounded: bool
    sources: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ChatSession:
    session_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turns: list[ChatTurn] = field(default_factory=list)


class ChatSessionManager:
    """In-memory session manager with automatic context resolution."""

    def __init__(self, max_history_per_session: int = 20) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._max_history = max_history_per_session

    def get_or_create(self, session_id: str | None = None) -> ChatSession:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        sid = session_id or str(uuid4())
        session = ChatSession(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        return bool(self._sessions.pop(session_id, None))

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "turn_count": len(s.turns),
                "last_query": s.turns[-1].query if s.turns else None,
            }
            for s in sorted(self._sessions.values(), key=lambda s: s.created_at, reverse=True)
        ]

    def reformulate_query(self, session: ChatSession, new_query: str) -> str:
        """Contextualize follow-up questions using prior turns."""
        clean_q = new_query.strip()
        if not session.turns:
            return clean_q

        last_turn = session.turns[-1]
        has_followup_cue = bool(_FOLLOWUP_PRONOUNS.search(clean_q)) or len(clean_q.split()) <= 4

        if not has_followup_cue:
            return clean_q

        # Extract context keywords/entities from previous query and sources
        last_q_terms = [
            w for w in re.findall(r"[A-Za-z0-9_-]+", last_turn.query)
            if len(w) > 3 and w.lower() not in {"what", "when", "where", "which", "could", "would", "should"}
        ]
        context_hint = " ".join(last_q_terms[:3])

        # If question is very short or references 'it'/'that', blend the context
        if context_hint and context_hint.lower() not in clean_q.lower():
            return f"{clean_q} ({context_hint})"
        return clean_q

    def add_turn(
        self,
        session_id: str,
        query: str,
        reformulated_query: str,
        answer: str,
        grounded: bool,
        sources: list[dict[str, Any]],
    ) -> ChatTurn:
        session = self.get_or_create(session_id)
        turn = ChatTurn(
            turn_id=str(uuid4()),
            query=query,
            reformulated_query=reformulated_query,
            answer=answer,
            grounded=grounded,
            sources=sources,
        )
        session.turns.append(turn)
        if len(session.turns) > self._max_history:
            session.turns.pop(0)
        return turn


_GLOBAL_CHAT_MANAGER = ChatSessionManager()


def get_chat_manager() -> ChatSessionManager:
    return _GLOBAL_CHAT_MANAGER

