"""
Runs one real Veo 3.1 trial via OpenRouter: generates the frame-0 stimulus
image, builds the prompt, submits the job, polls to completion, and
downloads the resulting video.

Usage:
    uv run python scripts/run_trial.py
"""

import json

from finst_video_model.config import TrialConfig
from finst_video_model.veo_client import run_trial


def main():
    cfg = TrialConfig(n_circles=6, n_cued=2, seed=42)
    result = run_trial(cfg, "data")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
