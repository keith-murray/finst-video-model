# Saving stimuli as numpy arrays

## Context

Given the poor performance of these models on the smooth pursuit and pylyshyn tasks, I am investigating the hypothesis that this is due to downsampling and compression artifacts from the server side of unpacking the mp4. To mitigate this possibility, I am locally hosting the Qwen3.8-27B model to manually adjust sampling fps rates.

## Compression from mp4 files

I want to understand if there is some artifact being introduced from the mp4 file format. To test this hypothesis, let's save stimuli as numpy arrays and mp4s. I have been testing the smooth pursuit task and have been using `scripts/smooth_pursuit/generate_samples.py` to generate simple stimuli. Let's edit this file and the associated files to also have a flag to save the stimuli as numpy arrays (as well as mp4s).

You will notice that I have changed the default parameters on `src/finst_video_model/comprehension/smooth_pursuit/config.py`. This is intentional so as to make the files and stimuli more digestible by the vLLM interface.
