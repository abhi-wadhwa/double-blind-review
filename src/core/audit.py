"""Audit trail for tracking all platform actions.

Every significant operation (anonymization, assignment, review
submission, aggregation, etc.) is recorded with a timestamp, actor,
action type, and structured details.  The trail is append-only and
can be exported for compliance or debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: str = ""
    action: str = ""
    actor: str = ""
    entity_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "actor": self.actor,
            "entity_id": self.entity_id,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class AuditTrail:
    """Append-only audit log for platform operations.

    Parameters
    ----------
    actor:
        Default actor identity for log entries (e.g., ``"system"``
        or a user ID).
    """

    def __init__(self, actor: str = "system") -> None:
        self._entries: list[AuditEntry] = []
        self._actor = actor

    @property
    def entries(self) -> list[AuditEntry]:
        """Return a copy of all audit entries."""
        return list(self._entries)

    @property
    def count(self) -> int:
        """Return the number of entries in the log."""
        return len(self._entries)

    def log(
        self,
        action: str,
        entity_id: str = "",
        details: Optional[dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> AuditEntry:
        """Append a new entry to the audit trail.

        Parameters
        ----------
        action:
            The action name (e.g., ``"anonymize"``, ``"assign"``,
            ``"submit_review"``, ``"aggregate"``).
        entity_id:
            The primary entity the action applies to.
        details:
            Arbitrary structured data about the action.
        actor:
            Override the default actor for this entry.

        Returns
        -------
        AuditEntry
            The created entry.
        """
        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            actor=actor or self._actor,
            entity_id=entity_id,
            details=details or {},
        )
        self._entries.append(entry)
        return entry

    def filter(
        self,
        action: Optional[str] = None,
        entity_id: Optional[str] = None,
        actor: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[AuditEntry]:
        """Filter audit entries by criteria.

        All parameters are optional.  Only entries matching *all*
        specified criteria are returned.
        """
        results = self._entries
        if action:
            results = [e for e in results if e.action == action]
        if entity_id:
            results = [e for e in results if e.entity_id == entity_id]
        if actor:
            results = [e for e in results if e.actor == actor]
        if since:
            results = [e for e in results if e.timestamp >= since]
        return results

    def export_json(self) -> str:
        """Export the entire trail as a JSON string."""
        return json.dumps(
            [e.to_dict() for e in self._entries],
            indent=2,
        )

    def export_jsonl(self) -> str:
        """Export the trail as newline-delimited JSON (JSON Lines)."""
        return "\n".join(e.to_json() for e in self._entries)

    def clear(self) -> None:
        """Clear all entries (use only in testing)."""
        self._entries.clear()

    def summary(self) -> dict[str, int]:
        """Return a count of entries by action type."""
        counts: dict[str, int] = {}
        for entry in self._entries:
            counts[entry.action] = counts.get(entry.action, 0) + 1
        return counts
