# New model and stimulus directions

## Context

We have experimented with getting video generation models like Veo to perform a FINST-like task; however, the results have been mixed. One limitation is that motion generation in Veo is not great for multiple objects. Looking at Google's own documentation about Veo, they acknowledge this upfront. Hence, we want to generate the motion ourselves.

This means that we cannot use video generation models that only use the initial frame and a text prompt. We have two options:

1. Use video generation models that can extend existing video. There are very few of these, and they are still in the beta phase, but it's worth a shot.
2. Use video-to-text models. The task will look very similar to the original FINST task. I.e. we can ask the model to output the location circles it was tasked to track, or we can flash circles at the end of the video and ask the model to report if those circles were in the original subset of circles it was asked to track.

Both of these options require that we generate our own videos with circles that move according to our own physcis.

## Actionable

If already asked Claude to generate some python code that could generate these videos. You will find it in the `claude_gen_files`. You do not need to use this code, you may use it as inspiration. All we need to do is to add or modify python files in `src/finst_video_model/` to generate the video.

I'm unsure if OpenRouter accepts video or even hosts models that can take video, so we will leave the implementation of that step for another time.

Also, the `claude/2026_07_20/claude_gen_files/stimulus_gen.py` file has a label phase at the end. Let's keep this, but it should be a hyperparameter of whether or not we use this label phase.

## Results

After trying Kling AI's video extend feature, I'm getting the sense that video extension has to develop more before we can plausibly use it. This means that for now, we will abandon all video generation and only stick to video comprehension. The models are much better at video-to-text, so it's more worthwhile to study.
