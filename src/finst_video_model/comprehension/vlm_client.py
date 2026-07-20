"""
Generic OpenRouter chat-completions wrapper for asking a video-understanding
VLM a text question about a locally rendered video.

Sends the video as a base64 data URL alongside a text question via
POST /api/v1/chat/completions (per claude/openrouter/video_input.md), rather
than the async submit/poll/download video-generation API used by
finst_video_model.generation.client (this is a synchronous chat call, not a
video generation job).

This module is intentionally generic: it knows nothing about any particular
experiment's question text or model choice. Each experiment script (in
`scripts/`) owns its own question text and model name, then calls
`ask_about_video` here.
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before querying a VLM."
        )
    return key


def _video_data_uri(video_path: str) -> str:
    with open(video_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:video/mp4;base64,{b64}"


def ask_about_video(video_path: str, question: str, model: str, timeout_s: float = 180.0) -> str:
    """Sends `video_path` + `question` to `model` (an OpenRouter chat model
    slug with video input support, e.g. "google/gemini-2.5-flash") via
    /chat/completions. Returns the model's raw text response."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "video_url",
                        "video_url": {"url": _video_data_uri(video_path)},
                    },
                ],
            }
        ],
    }
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
