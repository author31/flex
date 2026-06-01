"""Domain layer — entities, value objects, enums, rules. Pure: no I/O, no framework.

Depends on nothing in this package. numpy is used only as a pure compute container
for mask geometry (no I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class DomainError(Exception):
    """Base for domain rule violations (mapped to HTTP 422 by infrastructure)."""


class EmptyPrompt(DomainError):
    pass


class EmptyMask(DomainError):
    pass


class InvalidEditParams(DomainError):
    pass


class InvalidPreset(DomainError):
    pass


class InvalidTransition(DomainError):
    pass


class InvalidRequest(DomainError):
    """One-of / required-field violations on an edit request."""


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #


class FacialRegion(StrEnum):
    FACE = "face"
    EYES = "eyes"
    MOUTH = "mouth"
    EYEBROWS = "eyebrows"
    CUSTOM = "custom"


class ExpressionPreset(StrEnum):
    SMILE = "smile"
    ANGRY = "angry"
    SAD = "sad"
    SURPRISED = "surprised"


class JobStatus(StrEnum):
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EditParams:
    """Inpainting parameters with validated ranges (SDXL inpaint needs strength<1.0)."""

    strength: float = 0.99
    steps: int = 25
    guidance: float = 8.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if not (0.0 < self.strength < 1.0):
            raise InvalidEditParams("strength must be in (0, 1)")
        if not (10 <= self.steps <= 50):
            raise InvalidEditParams("steps must be in [10, 50]")
        if not (1.0 <= self.guidance <= 20.0):
            raise InvalidEditParams("guidance must be in [1, 20]")
        if self.seed is not None and self.seed < 0:
            raise InvalidEditParams("seed must be >= 0")


@dataclass(frozen=True)
class Mask:
    """Binary edit mask. White (>0) = edit, black (0) = preserve."""

    data: np.ndarray  # 2D array, shape (H, W)

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def is_empty(self) -> bool:
        return not bool(np.any(self.data))

    def area(self) -> int:
        return int(np.count_nonzero(self.data))

    def bbox(self) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the non-zero region. Raises if empty."""
        if self.is_empty():
            raise EmptyMask("cannot compute bbox of an empty mask")
        ys, xs = np.nonzero(self.data)
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


@dataclass
class CharacterImage:
    id: str
    width: int
    height: int


@dataclass
class EvaluationRecord:
    clip_similarity_in_mask: float
    edit_success: bool
    edit_success_score: float
    identity_preserved: bool
    latency_ms: int


@dataclass
class EditResult:
    id: str
    job_id: str
    result_image_id: str


@dataclass
class EditJob:
    id: str
    image_id: str
    region: FacialRegion
    prompt: str
    params: EditParams
    status: JobStatus = JobStatus.PROCESSING
    result_image_id: str | None = None
    evaluation: EvaluationRecord | None = None
    error: str | None = None

    def mark_completed(self, result_image_id: str, evaluation: EvaluationRecord) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise InvalidTransition(f"cannot complete a {self.status.value} job")
        self.status = JobStatus.COMPLETED
        self.result_image_id = result_image_id
        self.evaluation = evaluation

    def mark_failed(self, error: str) -> None:
        if self.status is not JobStatus.PROCESSING:
            raise InvalidTransition(f"cannot fail a {self.status.value} job")
        self.status = JobStatus.FAILED
        self.error = error


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #


def validate_prompt(prompt: str) -> str:
    cleaned = prompt.strip()
    if not cleaned:
        raise EmptyPrompt("prompt must not be empty")
    return cleaned


def validate_mask(mask: Mask) -> Mask:
    if mask.is_empty():
        raise EmptyMask("a non-empty region/mask is required")
    return mask


_EXPRESSION_PHRASES: dict[ExpressionPreset, str] = {
    ExpressionPreset.SMILE: "a happy smiling expression, smiling mouth, cheerful",
    ExpressionPreset.ANGRY: "an angry expression, furrowed eyebrows, frowning",
    ExpressionPreset.SAD: "a sad expression, downturned mouth, teary eyes",
    ExpressionPreset.SURPRISED: "a surprised expression, wide open eyes, raised eyebrows",
}

_REGION_FOCUS: dict[FacialRegion, str] = {
    FacialRegion.FACE: "on the face",
    FacialRegion.EYES: "focused on the eyes",
    FacialRegion.MOUTH: "focused on the mouth",
    FacialRegion.EYEBROWS: "focused on the eyebrows",
    FacialRegion.CUSTOM: "in the selected region",
}


def preset_to_prompt(preset: ExpressionPreset, region: FacialRegion) -> str:
    """Map a preset + region to an engine prompt (US3 / FR-006, FR-007)."""
    if preset not in _EXPRESSION_PHRASES:
        raise InvalidPreset(f"unsupported preset: {preset}")
    return f"{_EXPRESSION_PHRASES[preset]}, {_REGION_FOCUS[region]}"


SUPPORTED_PRESETS: tuple[ExpressionPreset, ...] = tuple(ExpressionPreset)
SUPPORTED_REGIONS: tuple[FacialRegion, ...] = tuple(FacialRegion)
# Regions the segmenter produces automatically (eyebrows is derived).
AUTO_REGIONS: tuple[FacialRegion, ...] = (
    FacialRegion.FACE,
    FacialRegion.EYES,
    FacialRegion.MOUTH,
    FacialRegion.EYEBROWS,
)
