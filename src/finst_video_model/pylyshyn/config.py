"""
Trial configuration for the Pylyshyn-faithful FINST/MOT reproduction (see
`claude/2026_08_12/TODO.md` for the original experiment excerpts this is
adapted from).

Siloed from `finst_video_model.config` because the design
differs in ways that don't fit the same fields: the field size is fixed at
n_objects (not swept), motion is a continuously-redirecting random walk
rather than one constant heading, and the report is a single True/False
judgment about a brief mid-trial probe flash rather than an end-of-clip
letter identification.

min_probe_delay_s/min_post_probe_s below are further constrained by the
original paper's own probe-timing requirements, not just by convenience --
see the comment on that field. (Earlier versions of this file also derived
redirect/probe-flash timing from an assumed downstream VLM sampling rate;
that derivation was superseded 2026-08-26 -- see below.)

Visual design (resolution, cross size, colors, and color-based cueing/
probing rather than blinking/shape-changing) was ported over from
`finst_video_model.debug_circular.config` on 2026-08-26
(claude/2026_08/2026_08_26/TODO.md's "Part 3") once the debug_circular work
confirmed a duration-stretched mp4 encoding gets OpenRouter to retain far
more frames -- this stimulus should look nearly identical to
debug_circular's except for the motion, which is unchanged (continuously-
redirecting random walk, not debug_circular's rigid rotation).

Timing was further slowed down and matched to debug_circular's exact
cue/track/fps numbers on the same date, per explicit user request (fps=10,
cue_s=1.0, tracking_s=9.0 -> 100 total frames, probe_flash_s=1.0) -- this
superseded the original ASSUMED_VLM_SAMPLE_PERIOD_S=1.0-driven redirect/
probe-margin derivation below (kept only where it still reflects the
original Pylyshyn paper's own fidelity requirements, e.g.
min_probe_delay_s/min_post_probe_s). redirect_min_s/max_s and
speed_min/max_px_s are a first-pass "slow it down" guess (redirect period
2.0s, speed roughly halved) explicitly pending the user's visual feedback
on generated samples, not a re-derived-from-geometry value like the
original 40/90 (later 31/65) bounds were.
"""

from dataclasses import dataclass, field, asdict
import json
import uuid


@dataclass
class TrialConfig:
    # --- Core independent variables ---
    # Field size fixed at 10 in the original; reduced to 3 to start (see
    # module docstring's "Part 3") -- kept as a swept-capable field (not
    # hardcoded) so future work can scale it back up.
    n_objects: int = 3
    n_cued: int = 1        # number of targets cued (colored red), 1-5 in the original

    # --- Timing (seconds) ---
    # fps/cue_s/tracking_s match debug_circular's own cue_flash_s=1.0/
    # tracking_s=8.0 numbers as closely as the mid-tracking-probe design
    # allows -- cue_s=1.0 + tracking_s=9.0 at fps=10 gives exactly 100
    # total frames, same as debug_circular's 100-frame clips.
    fps: int = 10
    cue_s: float = 1.0                # objects stationary; cued subset colored red
    tracking_s: float = 9.0          # motion phase, probe flash embedded partway through
    # Matches debug_circular's probe_flash_s=1.0 exactly (was 2.0, derived
    # from a now-superseded VLM-sampling-margin guarantee -- see module
    # docstring).
    probe_flash_s: float = 1.0

    # Probe must occur at least this far into tracking_s, and tracking_s
    # must continue at least this long after it ends, matching the
    # original's "at least 3 seconds after the start of the animation" /
    # "at least 4s after the target-flash" constraints (unchanged --
    # grounded in the original paper's own fidelity requirement, not in the
    # redirect cadence). With tracking_s=9.0 this leaves only a 2s window
    # for t_probe to be drawn from (see _choose_probe in stimulus_gen.py) --
    # revisit if that proves too narrow once real samples are eyeballed.
    min_probe_delay_s: float = 3.0
    min_post_probe_s: float = 4.0

    # Whether the single probe in this trial lands on a cued target or a
    # distractor -- the two conditions needed to compute d' (hit rate from
    # probe_on_target=True trials, false-alarm rate from False ones).
    probe_on_target: bool = True

    # --- Motion ---
    # redirect_min_s == redirect_max_s makes physics.py's rng.uniform(...)
    # deterministically return that value, so each object holds one
    # (direction, speed) draw per redirect interval -- one explicit,
    # controllable step rather than an unpredictable random-walk sum.
    # Direction is still one of 8 compass headings, as in the original.
    #
    # 2.0s here (was 1.0s, tied to the now-superseded
    # ASSUMED_VLM_SAMPLE_PERIOD_S assumption -- see module docstring) is a
    # first-pass "slow the motion down" guess per explicit user request
    # 2026-08-26, pending feedback on generated samples -- not re-derived
    # from any sampling-rate assumption.
    redirect_min_s: float = 2.0
    redirect_max_s: float = 2.0
    # Roughly halved from the geometry-derived 31/65 (see git history for
    # that derivation) as part of the same 2026-08-26 "slow it down" pass --
    # also a first-pass guess pending sample feedback, not re-derived from
    # field geometry (which hasn't changed).
    speed_min_px_s: float = 16.0
    speed_max_px_s: float = 33.0

    # --- Stimulus geometry (ported from debug_circular.config) ---
    image_width: int = 384
    image_height: int = 384
    object_size: int = 14       # cross half-extent
    arm_thickness: int = 5      # cross arm thickness
    min_center_distance_factor: float = 2.6  # * object_size, min separation

    # --- Colors (RGB, ported from debug_circular.config) ---
    background_color: tuple = (0, 0, 0)
    cued_color: tuple = (255, 0, 0)      # cued crosses during cue_s; probed cross during its flash
    neutral_color: tuple = (255, 255, 255)

    # --- Misc / bookkeeping ---
    seed: int = 0
    trial_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def total_duration_s(self) -> float:
        return self.cue_s + self.tracking_s

    @property
    def total_frames(self) -> int:
        return int(round(self.total_duration_s * self.fps))

    def validate(self):
        assert self.n_cued <= self.n_objects, "n_cued cannot exceed n_objects"
        assert self.n_cued >= 1, "n_cued must be at least 1"
        assert self.tracking_s >= self.min_probe_delay_s + self.min_post_probe_s, (
            "tracking_s must be long enough to fit min_probe_delay_s before "
            "the probe and min_post_probe_s after it"
        )
        assert self.redirect_min_s <= self.redirect_max_s
        assert self.speed_min_px_s <= self.speed_max_px_s
        assert self.probe_flash_s <= self.min_post_probe_s, (
            "probe_flash_s must fit within min_post_probe_s so the flash "
            "never runs past the reserved post-probe tracking margin"
        )

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
