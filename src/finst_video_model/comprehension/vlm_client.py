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

Enforces OpenRouter's rate limit (20 requests/minute, per their docs) via a
sliding-window limiter shared across all callers in this process, including
concurrent ones (e.g. run_comprehension_batch.py's thread pool) -- this way
the cap holds regardless of how many trials a caller runs in parallel,
rather than relying on --concurrency alone to stay under it.

The limiter is per-process, though, so it has no memory of requests made by
a different, prior process invocation (e.g. a quick one-off smoke test run
moments before a real batch) -- OpenRouter's actual account-level window can
still be partway used up even though this process's own deque starts empty.
To absorb that (and any other transient 429), ask_about_video retries on a
429 response with a short backoff rather than failing the trial outright.
"""

import base64
import os
import threading
import time
from collections import deque

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_REQUESTS_PER_MINUTE = 20


class _RateLimiter:
    """Sliding-window limiter: blocks callers so no more than `max_per_minute`
    acquisitions happen in any trailing 60s window, across all threads."""

    def __init__(self, max_per_minute: int):
        self.max_per_minute = max_per_minute
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            while True:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= 60.0:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_per_minute:
                    self._timestamps.append(now)
                    return
                time.sleep(60.0 - (now - self._timestamps[0]))


_rate_limiter = _RateLimiter(MAX_REQUESTS_PER_MINUTE)


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


MAX_429_RETRIES = 3


def ask_about_video(
    video_path: str, question: str, model: str, timeout_s: float = 180.0,
    provider: dict | None = None, reasoning: dict | None = None,
    return_usage: bool = False,
) -> str | tuple[str, dict | None]:
    """Sends `video_path` + `question` to `model` (an OpenRouter chat model
    slug with video input support, e.g. "google/gemini-2.5-flash") via
    /chat/completions. Returns the model's raw text response.

    `provider` and `reasoning`, when given, are passed through verbatim as
    the OpenRouter request body's `provider`/`reasoning` fields (e.g.
    `provider={"only": ["google-vertex"], "allow_fallbacks": False}` to pin
    a specific provider, `reasoning={"effort": "minimal"}` to cap reasoning
    spend on models that default to heavy internal reasoning even for a
    short answer -- some endpoints reject `"effort": "none"` outright with
    a 400, "reasoning is mandatory," so `"minimal"` is the practical floor,
    and even that isn't a hard token cap: OpenRouter's own docs note actual
    reasoning-token counts for Gemini are decided internally by Google
    regardless of the requested effort level). Omitted (the default) for
    both, matching prior behavior exactly.

    If `return_usage`, returns `(text, usage_dict_or_None)` instead of just
    `text` -- `usage_dict` is the response's raw `usage` object (token
    counts and `cost` in USD), useful for a batch runner to log actual
    per-trial spend rather than relying on a pre-run estimate.
    """
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
    if provider is not None:
        payload["provider"] = provider
    if reasoning is not None:
        payload["reasoning"] = reasoning

    for attempt in range(MAX_429_RETRIES + 1):
        _rate_limiter.acquire()
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_s,
        )
        if response.status_code == 429 and attempt < MAX_429_RETRIES:
            retry_after = float(response.headers.get("Retry-After", 15.0))
            time.sleep(retry_after)
            continue
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        return (text, data.get("usage")) if return_usage else text
