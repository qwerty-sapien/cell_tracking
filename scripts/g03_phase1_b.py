#!/usr/bin/env python3
"""Kaggle entrypoint B for the G-03 resume proof."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from g03_resume_torch import run_b


if __name__ == "__main__":
    run_b()
