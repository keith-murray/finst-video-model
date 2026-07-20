"""
Trial configuration for the VLM (video-to-text) FINST/MOT capacity
experiment -- the "comprehension arm".

Unlike the Veo generation arm (see `finst_video_model.generation.config`), we fully
simulate and render the video ourselves, so we have exact ground truth for
every circle at every frame -- no reverse engineering required. The model's
job is purely to answer a text question about which circles were cued.
"""

from dataclasses import dataclass, field, asdict
import json
import uuid


@dataclass
class TrialConfig:
    # --- Core independent variables ---
    n_circles: int = 6          # total circles (capacity axis)
    n_cued: int = 2             # number of circles cued red at t=0
    force_path_crossing: bool = False  # bias initial velocities so cued
                                        # circles' paths cross distractors'
                                        # (set True for the crossing/"swap"
                                        # condition)

    # --- Timing (seconds) ---
    fps: int = 24
    cue_flash_s: float = 1.0    # initial red cue period
    tracking_s: float = 8.0     # de-cued phase, all circles identical gray

    # Whether to append a frozen, letter-labeled final frame (needed for the
    # VLM text-question task). Disable for uses that only need the raw
    # cue -> tracking motion, e.g. probing a frozen encoder's activations
    # directly against ground truth instead of asking a model a question.
    use_label_phase: bool = True
    label_s: float = 2.0        # frozen final frame with letter labels

    # --- Motion ---
    speed_px_s: float = 140.0   # constant circle speed, pixels/second

    # --- Stimulus geometry ---
    image_width: int = 1280
    image_height: int = 720
    circle_radius: int = 28
    min_center_distance_factor: float = 2.6  # * radius, for initial non-overlap

    # --- Colors (RGB) ---
    background_color: tuple = (235, 235, 235)
    cued_color: tuple = (220, 40, 40)       # red
    neutral_color: tuple = (120, 120, 120)  # gray
    label_text_color: tuple = (255, 255, 255)

    # --- Misc / bookkeeping ---
    seed: int = 0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def total_duration_s(self) -> float:
        label_s = self.label_s if self.use_label_phase else 0.0
        return self.cue_flash_s + self.tracking_s + label_s

    @property
    def total_frames(self) -> int:
        return int(round(self.total_duration_s * self.fps))

    def validate(self):
        assert self.n_cued <= self.n_circles, "n_cued cannot exceed n_circles"
        assert self.n_circles <= 26, "only 26 letters (A-Z) available for labels"
        if self.use_label_phase:
            assert self.label_s > 0, "label_s must be > 0 when use_label_phase is set"

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
