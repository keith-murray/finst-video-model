"""
Trial configuration for the Pylyshyn-faithful FINST/MOT reproduction (see
`claude/2026_08_12/TODO.md` for the original experiment excerpts this is
adapted from).

Siloed from `finst_video_model.comprehension.config` because the design
differs in ways that don't fit the same fields: the field size is fixed at
n_objects (not swept), objects are stationary and blink during cueing rather
than moving red circles, motion is a continuously-redirecting random walk
rather than one constant heading, and the report is a single True/False
judgment about a brief mid-trial probe flash rather than an end-of-clip
letter identification.

The cue/probe/motion timing fields below are further constrained by the
downstream VLM's video ingestion (OpenRouter backends, e.g. Gemini,
effectively downsample to ~1 fps regardless of our render fps) -- not just
by Pylyshyn-fidelity -- see the comment on each affected field.
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
    n_objects: int = 10   # total field size, fixed at 10 in the original
    n_cued: int = 3        # number of targets cued (flashed), 1-5 in the original

    # --- Timing (seconds) ---
    fps: int = 24
    cue_s: float = 8.0                # objects stationary; cued subset blinks
    # Blinks at 4x ASSUMED_VLM_SAMPLE_PERIOD_S (i.e. much faster than the
    # VLM's sampling rate, not matched to it) so each ~1s-spaced sample
    # lands on an effectively independent random on/off phase -- a
    # dithered 50/50 draw decorrelated from sampling phase. Matching the
    # blink rate to the sampling rate instead would risk aliasing (a
    # sampled frame could phase-lock onto "always on" or "always off",
    # making a cued object indistinguishable from a steady distractor).
    # With cue_s=8.0 (~8 samples), the chance any one cued object's
    # samples all land on the same state is 2*(0.5)^8 ~= 0.8%.
    cue_blink_period_s: float = 0.125  # on/off half-period during cue_s
    tracking_s: float = 10.0         # motion phase (7-15s in the original)
    # Long enough to be guaranteed caught by at least one sample regardless
    # of unknown sampling phase: for periodic sampling of period T, an
    # event of duration >= T can't be slotted entirely between two
    # samples. Using a conservative T_max ~= 1.3*ASSUMED_VLM_SAMPLE_PERIOD_S
    # (since the true rate is only approximate) and rounding up for margin
    # gives 2.0s; at the nominal rate this generically catches two samples.
    probe_flash_s: float = 2.0

    # Probe must occur at least this far into tracking_s, and tracking_s
    # must continue at least this long after it ends, matching the
    # original's "at least 3 seconds after the start of the animation" /
    # "at least 4s after the target-flash" constraints. Expressed in
    # ASSUMED_VLM_SAMPLE_PERIOD_S terms, these are now ~3 and ~4 tracking
    # hops (see redirect_min_s/redirect_max_s below).
    min_probe_delay_s: float = 3.0
    min_post_probe_s: float = 4.0

    # Whether the single probe in this trial lands on a cued target or a
    # distractor -- the two conditions needed to compute d' (hit rate from
    # probe_on_target=True trials, false-alarm rate from False ones).
    probe_on_target: bool = True

    # --- Motion ---
    # The VLM only ever perceives two static endpoint positions per sample
    # interval, not the path between them -- it can't tell "one 1s straight
    # leg" from "several sub-second legs summing to the same net
    # displacement" (unlike the original's "every few hundred
    # milliseconds" redirect rate, which assumed a human perceiving
    # continuous motion). So instead, both bounds are fixed at
    # ASSUMED_VLM_SAMPLE_PERIOD_S: redirect_min_s == redirect_max_s makes
    # physics.py's rng.uniform(...) deterministically return that value,
    # so each object holds one (direction, speed) draw per assumed VLM
    # sample interval -- one explicit, controllable step instead of an
    # unpredictable random-walk sum. Direction is still one of 8 compass
    # headings, as in the original.
    redirect_min_s: float = ASSUMED_VLM_SAMPLE_PERIOD_S
    redirect_max_s: float = ASSUMED_VLM_SAMPLE_PERIOD_S
    # Since each redirect interval is exactly ASSUMED_VLM_SAMPLE_PERIOD_S,
    # speed_px_s is numerically the per-sample hop distance in pixels.
    # Bounds are sized against field geometry (image_width/height,
    # n_objects below): mean nearest-neighbor distance for n_objects=10
    # scattered over a 1280x720 field, treated as ~Poisson, is
    # 0.5*sqrt(area/n) ~= 152px. 40px (~2.2x object_size) is unambiguously
    # a real hop, not noise; 90px (~59% of 152px) leaves ~40% margin
    # against ambiguous frame-to-frame nearest-neighbor matches between
    # samples (with _resolve_separations in physics.py as a second layer
    # of defense in denser local configurations). Recompute if n_objects
    # or the image dimensions change.
    speed_min_px_s: float = 40.0
    speed_max_px_s: float = 90.0

    # --- Stimulus geometry ---
    image_width: int = 1280
    image_height: int = 720
    object_size: int = 18       # cross half-extent / probe square half-extent
    arm_thickness: int = 7      # cross arm thickness
    min_center_distance_factor: float = 2.6  # * object_size, min separation

    # --- Colors (RGB) ---
    background_color: tuple = (15, 15, 15)     # original: black background
    object_color: tuple = (235, 235, 235)      # original: white crosses
    probe_color: tuple = (235, 235, 235)       # solid white square flash

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
