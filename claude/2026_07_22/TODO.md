# Understanding video length and increasing cued circles

## Understanding effect of video length

### Context

We've learned a lot about the model's ability to track objects already from the `results/e8489fbd/accuracy_plot_speed_px_s.png` plot generated yesterday. We've investigated the effect of speed and number of circles. The next natural variable to change is video length.

### Video length actionable

Let's design another sweep, with all the same parameters as before, to test different video lengths. We still want the cued seconds at the beginning to be 2 and the label seconds at the end to be 2. For the video duration, let's try 4, 6, 8, 10, 12 seconds. Let's also fix number of circles to be 5 with only 1 cue. We should use the same speeds as before.

### Video length run command

uv run python scripts/run_comprehension_batch.py \
    --n-circles 5 --n-cued 1 \
    --tracking-s 4 6 8 10 12 \
    --speed-px-s 70.0 87.5 105.0 122.5 140.0 \
    --cue-flash-s 2 --label-s 2 \
    --fps 4 \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --models qwen/qwen3.5-flash-02-23 \
    --concurrency 10

### Results

Unsurprisingly, we see a monotonic decrease in accuracy as tracking seconds go up.

## Next task

We'll skip the increasing cued circles task, as implied in the title. Let's just make a two panel figure that is the left panel of `results/e8489fbd/accuracy_plot_speed_px_s.png` and the left panel of `results/ce753d46/accuracy_plot_tracking_s_speed_px_s.png`. This is to be a summary figure for my PI of the two big sweeps we've run.
