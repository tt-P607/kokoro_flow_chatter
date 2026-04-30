"""KFC 场景状态模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


SceneCertainty = Literal["unknown", "weak", "confirmed"]
SceneEvidenceKind = Literal[
    "user_message",
    "history_message",
    "tool_result",
    "setting",
    "inference",
]


@dataclass(slots=True)
class SceneEvidence:
    """单条场景证据。"""

    source: str
    content: str
    kind: SceneEvidenceKind = "user_message"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "source": self.source,
            "content": self.content,
            "kind": self.kind,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneEvidence:
        """从字典反序列化。"""
        kind = str(data.get("kind", "user_message") or "user_message")
        if kind not in {
            "user_message",
            "history_message",
            "tool_result",
            "setting",
            "inference",
        }:
            kind = "user_message"

        confidence = data.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)):
            confidence = 1.0

        return cls(
            source=str(data.get("source", "") or ""),
            content=str(data.get("content", "") or ""),
            kind=kind,
            confidence=float(confidence),
        )


@dataclass(slots=True)
class SceneState:
    """当前会话的显式场景状态。"""

    certainty: SceneCertainty = "unknown"
    location_type: str = "unknown"
    social_channel: str = ""
    device_assumption_allowed: bool = False
    evidence: list[SceneEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典。"""
        return {
            "certainty": self.certainty,
            "location_type": self.location_type,
            "social_channel": self.social_channel,
            "device_assumption_allowed": self.device_assumption_allowed,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneState:
        """从字典反序列化。"""
        certainty = str(data.get("certainty", "unknown") or "unknown")
        if certainty not in {"unknown", "weak", "confirmed"}:
            certainty = "unknown"

        raw_evidence = data.get("evidence", [])
        evidence_items = [
            SceneEvidence.from_dict(item)
            for item in raw_evidence
            if isinstance(item, dict)
        ]

        return cls(
            certainty=certainty,
            location_type=str(data.get("location_type", "unknown") or "unknown"),
            social_channel=str(data.get("social_channel", "") or ""),
            device_assumption_allowed=bool(data.get("device_assumption_allowed", False)),
            evidence=evidence_items,
        )