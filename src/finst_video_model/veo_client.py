"""
OpenRouter REST client for Veo 3.1 video generation.

Talks to Veo exclusively through OpenRouter's async video API (submit ->
poll -> download), per claude/openrouter/video_generation.md. This project
does NOT use the Google Gemini SDK.
"""

import base64
import json
import os
import time

import requests

from finst_video_model.config import TrialConfig
from finst_video_model.prompts import build_prompt
from finst_video_model.stimulus_gen import generate_stimulus

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
VEO_MODEL = "google/veo-3.1"
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


def submit_video_job(prompt: str, image_path: str, cfg: TrialConfig) -> dict:
    """POSTs a video generation job to OpenRouter. Returns the submit
    response: {"id", "polling_url", "status"}."""
    payload = {
        "model": VEO_MODEL,
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
        # Stimulus has no audio content, and audio-safety filters are a
        # known Veo 3.1 false-positive source -- skip audio generation.
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
    response = requests.get(content_url, timeout=120)
    response.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(response.content)
    return out_path


def run_trial(cfg: TrialConfig, out_dir: str) -> dict:
    """Proof-of-concept pipeline for one trial: generate the frame-0
    stimulus + prompt, submit to Veo via OpenRouter, poll to completion, and
    download the resulting video.

    Failures (safety blocks, timeouts, malformed responses) are caught and
    recorded in the returned dict rather than raised, so a future batch
    runner can log and continue instead of crashing.

    Writes `video_<trial_id>.mp4` and `generation_<trial_id>.json` into
    `out_dir`, alongside the `frame0_<trial_id>.png` /
    `ground_truth_<trial_id>.json` pair from `stimulus_gen.py`.
    """
    os.makedirs(out_dir, exist_ok=True)
    ground_truth = generate_stimulus(cfg, out_dir)
    prompt = build_prompt(cfg)

    result = {
        "trial_id": cfg.trial_id,
        "prompt": prompt,
        "frame0_path": ground_truth["frame0_path"],
    }

    try:
        submitted = submit_video_job(prompt, ground_truth["frame0_path"], cfg)
        result["job_id"] = submitted["id"]
        final_status = poll_job(submitted["polling_url"])
        result["status"] = final_status["status"]
        result["usage"] = final_status.get("usage")

        if final_status["status"] == "completed":
            video_path = f"{out_dir}/video_{cfg.trial_id}.mp4"
            download_video(final_status, video_path)
            result["video_path"] = video_path
        else:
            result["error"] = final_status.get("error", "Unknown failure")
    except (requests.RequestException, KeyError, TimeoutError, RuntimeError) as e:
        result["status"] = "error"
        result["error"] = str(e)

    gen_path = f"{out_dir}/generation_{cfg.trial_id}.json"
    with open(gen_path, "w") as f:
        json.dump(result, f, indent=2)
    result["generation_path"] = gen_path

    return result


if __name__ == "__main__":
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    result = run_trial(cfg, "data")
    print(json.dumps(result, indent=2))
