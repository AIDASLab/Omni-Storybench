"""Inference executors and runtime dispatch for Omni StoryBench judges."""

from src.inference.dispatch import initialize_inference, run_batch_inference


__all__ = [
    "initialize_inference",
    "run_batch_inference",
]
