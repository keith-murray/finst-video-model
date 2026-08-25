"""
Trial configuration for the "smooth pursuit" VLM (video-to-text) FINST/MOT
capacity experiment -- the original comprehension-arm task, reformatted
(per `claude/2026_08_14/TODO.md`'s "Updating the old task") to mirror the
Pylyshyn reproduction's (`finst_video_model.pylyshyn`) report
format while keeping this task's defining feature: one constant heading per
circle for the whole trial ("smooth" motion), rather than Pylyshyn's
frequently-redirecting random walk. Differences from the original version
of this task:
  - Circles are stationary during the cue phase (they used to already be
    moving), matching Pylyshyn's "cueing happens before motion starts."
  - A continuous minimum-separation constraint now runs during the tracking
    phase too (previously only enforced at initial placement), so identity
    is never ambiguous during motion, not just at t=0.
  - The end-of-trial report is a single freeze-frame True/False probe (one
    circle recolored red, "was this circle cued at the start?") instead of
    a frozen letter-labeled frame asking for all cued letters at once.

We fully simulate and render the video ourselves, so we have exact ground
truth for every circle at every frame -- no reverse engineering required.

Siloed from `finst_video_model.pylyshyn.config` (a separate,
same-named `TrialConfig`) for the same reason `pylyshyn` is siloed from this
module's old location: the fields don't overlap enough to share one
dataclass. The probe/motion timing fields below are further constrained by
the downstream VLM's video ingestion (OpenRouter backends, e.g. Gemini,
effectively downsample to ~1 fps regardless of our render fps) -- see the
comment on each affected field.
"""

from dataclasses import dataclass, field, asdict
import json
import uuid

# Assumed effective sampling period of the downstream VLM's video ingestion.
# This is an external fact about the model (we don't control or observe its
# exact rate/phase), not a render parameter -- fields below that are sized
# around it reference this constant in their comments.
ASSUMED_VLM_SAMPLE_PERIOD_S = 1.0


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
    fps: int = 20
    cue_flash_s: float = 2.0    # initial red cue period; circles are
                                 # stationary throughout this phase
    tracking_s: float = 6.0     # de-cued phase, all circles identical gray

    # Duration of the end-of-trial probe: motion freezes at the end of
    # tracking_s, one circle (probed_index) is recolored cued_color, and
    # the model is asked whether it was cued at the start. Long enough to
    # be guaranteed caught by at least one VLM sample regardless of unknown
    # sampling phase: for periodic sampling of period T, an event of
    # duration >= T can't be slotted entirely between two samples. Using a
    # conservative T_max ~= 1.3*ASSUMED_VLM_SAMPLE_PERIOD_S (the true rate
    # is only approximate) and rounding up for margin gives 2.0s.
    probe_flash_s: float = 2.0

    # Whether the probed circle in this trial was actually cued at t=0 --
    # the two conditions needed to compute d' later (hit rate from True
    # trials, false-alarm rate from False ones), mirroring
    # `pylyshyn.config.TrialConfig.probe_on_target`.
    probe_on_target: bool = True

    # --- Motion ---
    speed_px_s: float = 64.0   # constant circle speed, pixels/second,
                                 # unchanged for the whole tracking phase --
                                 # this single-heading-per-trial motion is
                                 # what distinguishes this task from
                                 # pylyshyn's frequently-redirecting walk

    # --- Stimulus geometry ---
    image_width: int = 384
    image_height: int = 384
    circle_radius: int = 16
    min_center_distance_factor: float = 2.6  # * radius; enforced both at
                                               # initial placement and
                                               # continuously during motion

    # --- Colors (RGB) ---
    background_color: tuple = (235, 235, 235)
    cued_color: tuple = (220, 40, 40)       # red; also used for the
                                             # end-of-trial probe highlight
    neutral_color: tuple = (120, 120, 120)  # gray

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
        assert self.n_cued <= self.n_circles, "n_cued cannot exceed n_circles"
        assert self.n_cued >= 1, "n_cued must be at least 1"
        assert self.probe_flash_s > 0, "probe_flash_s must be > 0"

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
