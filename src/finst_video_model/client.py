"""
OpenRouter REST client for video generation models.

Talks to models exclusively through OpenRouter's async video API (submit ->
poll -> download), per claude/openrouter/video_generation.md, rather than
any model provider's own SDK.

This module is intentionally generic: it knows nothing about any particular
experiment's stimulus, prompt, or model choice. Each experiment script (in
`scripts/`) owns its own prompt text, stimulus generation, and model name,
then calls `run_trial` here to launch it.
"""

import base64
import json
import os
import time

import requests
from dotenv import load_dotenv

from finst_video_model.config import TrialConfig

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "expired"}


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running video "
            "generation."
        )
    return key


def _image_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def submit_video_job(prompt: str, image_path: str, cfg: TrialConfig, model: str) -> dict:
    """POSTs a video generation job to OpenRouter for `model` (an OpenRouter
    model slug, e.g. "google/veo-3.1"). Returns the submit response:
    {"id", "polling_url", "status"}."""
    payload = {
        "model": model,
        "prompt": prompt,
        "frame_images": [
            {
                "type": "image_url",
                "image_url": {"url": _image_data_uri(image_path)},
                "frame_type": "first_frame",
            }
        ],
        "duration": cfg.clip_duration_s,
        "size": f"{cfg.image_width}x{cfg.image_height}",
        # Stimuli are visually silent; skip audio generation to avoid
        # audio-safety-filter false positives some models exhibit.
        "generate_audio": False,
    }
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/videos",
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def poll_job(polling_url: str, interval_s: float = 15.0, timeout_s: float = 600.0) -> dict:
    """Polls `polling_url` until the job reaches a terminal status or
    `timeout_s` elapses. Returns the final poll response."""
    headers = {"Authorization": f"Bearer {_api_key()}"}
    deadline = time.monotonic() + timeout_s

    while True:
        response = requests.get(polling_url, headers=headers, timeout=30)
        response.raise_for_status()
        status = response.json()

        if status["status"] in TERMINAL_STATUSES:
            return status

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Video generation did not finish within {timeout_s:.0f}s "
                f"(last status: {status['status']})"
            )

        time.sleep(interval_s)


def download_video(status: dict, out_path: str) -> str:
    """Downloads a completed job's video to `out_path`."""
    content_url = status["unsigned_urls"][0]
    headers = {"Authorization": f"Bearer {_api_key()}"}
    response = requests.get(content_url, headers=headers, timeout=120)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def run_trial(cfg: TrialConfig, trial_dir: str, prompt: str, frame0_path: str, model: str) -> dict:
    """Launches one trial: submits an already-built `prompt` + already-
    rendered `frame0_path` image to `model` via OpenRouter, polls to
    completion, and downloads the resulting video.

    Failures (safety blocks, timeouts, malformed responses) are caught and
    recorded in the returned dict rather than raised, so a future batch
    runner can log and continue instead of crashing.

    Callers are responsible for creating `trial_dir` and generating the
    stimulus + prompt into it (see `stimulus_gen.py` and each experiment
    script). Writes `video.mp4` and `generation.json` into `trial_dir`.
    """
    os.makedirs(trial_dir, exist_ok=True)

    result = {
        "trial_id": cfg.trial_id,
        "model": model,
        "prompt": prompt,
        "frame0_path": frame0_path,
    }

    try:
        submitted = submit_video_job(prompt, frame0_path, cfg, model)
        result["job_id"] = submitted["id"]
        final_status = poll_job(submitted["polling_url"])
        result["status"] = final_status["status"]
        result["usage"] = final_status.get("usage")

        if final_status["status"] == "completed":
            video_path = f"{trial_dir}/video.mp4"
            try:
                download_video(final_status, video_path)
                result["video_path"] = video_path
            except (requests.RequestException, KeyError) as e:
                result["download_error"] = str(e)
        else:
            result["error"] = final_status.get("error", "Unknown failure")
    except (requests.RequestException, KeyError, TimeoutError, RuntimeError) as e:
        result["status"] = "error"
        result["error"] = str(e)

    gen_path = f"{trial_dir}/generation.json"
    with open(gen_path, "w") as f:
        json.dump(result, f, indent=2)
    result["generation_path"] = gen_path

    return result
