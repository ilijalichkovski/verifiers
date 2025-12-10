"""Trainer variant that surfaces textual feedback from environments."""

from __future__ import annotations

from .orchestrator import Batch
from .trainer import RLTrainer


class TextFeedbackRLTrainer(RLTrainer):
    """RL trainer that records textual feedback alongside rollouts."""

    def get_textual_feedback(self, batch: Batch) -> list[str] | None:
        """Return textual feedback captured during rollout generation."""
        feedbacks = getattr(batch, "textual_feedbacks", None)
        if not feedbacks:
            return None
        return list(feedbacks)
