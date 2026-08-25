"""
Trial configuration for the "debug circular" stimulus: the simplest possible
motion-perception check ahead of returning to the pylyshyn/smooth_pursuit
arms (see `claude/2026_08/2026_08_25/TODO.md`). n_objects crosses sit
equally spaced around one circle and rotate together as a rigid ring at a
single constant angular velocity/direction for the whole trial -- there's no
random walk, no per-object heading, and no minimum-separation logic to
resolve, since equal spacing plus rigid rotation makes collisions impossible
by construction. The goal isn't to replicate FINST/MOT capacity limits; it's
to isolate whether a locally-hosted Qwen3.8-27B (with full control over the
raw frames it receives, not an OpenRouter-preprocessed mp4) can perceive
rotation direction and magnitude at all.

Siloed from `finst_video_model.pylyshyn.config` /
`finst_video_model.smooth_pursuit.config` (separate, same-named
`TrialConfig`), per this project's existing convention of one dataclass per
task since the fields don't overlap enough to share one.
"""

from dataclasses import dataclass, field, asdict
import json
import uuid


@dataclass
class TrialConfig:
    # --- Core independent variables ---
    n_objects: int = 3          # crosses equally spaced around the circle
    rotation_deg: float = 90.0  # total rotation over tracking_s (the "speed"
                                 # axis to sweep -- larger = faster)
    clockwise: bool = True      # rotation direction

    # Whether the cross highlighted red in the probe phase is the same
    # physical object cued red in the cue phase -- the two conditions
    # needed for a d' computation later, mirroring
    # pylyshyn/smooth_pursuit's probe_on_target.
    probe_on_target: bool = True

    # --- Timing (seconds) ---
    fps: int = 10
    cue_flash_s: float = 1.0     # objects stationary; one shown red
    tracking_s: float = 8.0      # rigid ring rotation; all objects white
    probe_flash_s: float = 1.0   # frozen at final position; one shown red

    # --- Stimulus geometry ---
    image_width: int = 384
    image_height: int = 384
    circle_radius: int = 140    # radius of the path objects travel along
    object_size: int = 14        # cross half-extent
    arm_thickness: int = 5       # cross arm thickness

    # --- Colors (RGB) ---
    background_color: tuple = (0, 0, 0)
    cued_color: tuple = (255, 0, 0)
    neutral_color: tuple = (255, 255, 255)

    # --- Misc / bookkeeping ---
    seed: int = 0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def total_duration_s(self) -> float:
        return self.cue_flash_s + self.tracking_s + self.probe_flash_s

    @property
    def total_frames(self) -> int:
        return int(round(self.total_duration_s * self.fps))

    def validate(self):
        assert self.n_objects >= 2, (
            "n_objects must be at least 2 (need a target and, for "
            "probe_on_target=False trials, a distinct distractor)"
        )
        assert self.rotation_deg >= 0, (
            "rotation_deg is a magnitude; use `clockwise` to set direction"
        )

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
