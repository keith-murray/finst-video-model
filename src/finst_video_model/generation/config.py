"""
Trial configuration for the Veo-3.1 FINST/MOT-style capacity experiment.

Each TrialConfig fully specifies one stimulus + prompt pair. Keep every
parameter that could plausibly affect the result here (not hardcoded
downstream) so sweeps and logging stay honest.
"""

from dataclasses import dataclass, field, asdict
import json
import uuid


@dataclass
class TrialConfig:
    # --- Core independent variables ---
    n_circles: int = 6          # total circles in the display (capacity axis)
    n_cued: int = 2             # number of circles cued red at t=0

    # --- Timing (all in seconds; must sum to <= clip duration Veo generates) ---
    clip_duration_s: int = 8    # Veo 3.1 supports 4, 6, or 8s per generation call
    cue_flash_s: float = 1.0    # how long the initial red cue is visible
    tracking_s: float = 6.0     # de-cued phase, all circles identical gray
    recue_s: float = 1.0        # final phase where cued circles turn red again

    # --- Stimulus geometry ---
    image_width: int = 1280
    image_height: int = 720
    circle_radius: int = 28
    min_center_distance_factor: float = 2.6  # * radius, for non-overlap + margin

    # --- Circular-track geometry (only used by the circular-track stimulus) ---
    track_radius: int = 250          # radius of the circular path, centered in frame
    track_line_width: int = 3
    track_start_angle_deg: float = -90.0  # angle of circle index 0; -90 = top of frame
    track_color: tuple = (160, 160, 160)

    # --- Colors (RGB) ---
    background_color: tuple = (235, 235, 235)
    cued_color: tuple = (220, 40, 40)     # red
    neutral_color: tuple = (120, 120, 120)  # gray

    # --- Motion description fed to the prompt (not simulated by us) ---
    physics_description: str = (
        "moving in straight lines at constant speed, bouncing elastically "
        "off the frame edges and off each other"
    )

    # --- Misc / bookkeeping ---
    seed: int = 0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def validate(self):
        assert self.n_cued <= self.n_circles, "n_cued cannot exceed n_circles"
        total = self.cue_flash_s + self.tracking_s + self.recue_s
        assert abs(total - self.clip_duration_s) < 1e-6, (
            f"Phase durations ({total}s) must sum to clip_duration_s "
            f"({self.clip_duration_s}s)"
        )
        assert self.clip_duration_s in (4, 6, 8), (
            "Veo 3.1 only supports 4, 6, or 8 second generations per call"
        )

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
