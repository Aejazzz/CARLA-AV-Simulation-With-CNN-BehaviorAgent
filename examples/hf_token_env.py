"""Hugging Face Hub token from environment (never hard-code secrets).

Reads, in order:
  - ``HF_TOKEN``
  - ``HUGGING_FACE_HUB_TOKEN``

Set before running (PowerShell)::

    $env:HF_TOKEN = "hf_*****"

Or run ``huggingface-cli login`` once (stores token in user cache).
"""
from __future__ import print_function

import os


def get_hf_hub_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
