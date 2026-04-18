from __future__ import annotations

from dataclasses import replace

from .types import PatchRecord


class PatchBank:
    def __init__(self) -> None:
        self._temporary: dict[str, PatchRecord] = {}
        self._durable: dict[str, PatchRecord] = {}
        self._counter = 0
        self._recent_rollbacks: set[str] = set()

    def clone(self) -> "PatchBank":
        other = PatchBank()
        other._temporary = {key: replace(value) for key, value in self._temporary.items()}
        other._durable = {key: replace(value) for key, value in self._durable.items()}
        other._counter = self._counter
        other._recent_rollbacks = set(self._recent_rollbacks)
        return other

    @staticmethod
    def scope_key(subject: str, relation: str) -> str:
        return f"{subject}::{relation}"

    def create_temporary_patch(
        self,
        subject: str,
        relation: str,
        answer: str,
        creation_trigger: str,
        acceptance_score: float,
        activation_policy: dict | None = None,
    ) -> PatchRecord:
        self._counter += 1
        patch_id = f"patch-{self._counter}"
        scope_key = self.scope_key(subject, relation)
        patch = PatchRecord(
            patch_id=patch_id,
            scope_key=scope_key,
            answer=answer,
            creation_trigger=creation_trigger,
            acceptance_score=acceptance_score,
            durability_status="temporary",
            rollback_metadata={"rollback_ready": True},
            activation_policy=activation_policy or {"scope": scope_key},
            subject=subject,
            relation=relation,
            temporary=True,
        )
        self._temporary[scope_key] = patch
        return patch

    def get_temporary(self, subject: str, relation: str, current_timestamp: int | None = None) -> PatchRecord | None:
        scope_key = self.scope_key(subject, relation)
        patch = self._temporary.get(scope_key)
        if patch is None:
            return None
        if current_timestamp is None:
            return patch
        expires_at = patch.activation_policy.get("expires_at")
        if expires_at is not None and current_timestamp > int(expires_at):
            self._temporary.pop(scope_key, None)
            return None
        return patch

    def get_durable(self, subject: str, relation: str) -> PatchRecord | None:
        return self._durable.get(self.scope_key(subject, relation))

    def promote(self, subject: str, relation: str) -> PatchRecord | None:
        scope_key = self.scope_key(subject, relation)
        patch = self._temporary.get(scope_key)
        if not patch:
            return None
        promoted = replace(patch, durability_status="durable", temporary=False)
        self._durable[scope_key] = promoted
        return promoted

    def clear_temporary(self, subject: str, relation: str) -> None:
        self._temporary.pop(self.scope_key(subject, relation), None)

    def rollback(self, subject: str, relation: str) -> bool:
        scope_key = self.scope_key(subject, relation)
        removed = self._durable.pop(scope_key, None)
        if removed:
            self._recent_rollbacks.add(scope_key)
            return True
        return False

    def recently_rolled_back(self, subject: str, relation: str) -> bool:
        return self.scope_key(subject, relation) in self._recent_rollbacks
