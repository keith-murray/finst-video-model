# Experimenting on Qwen3.8-27B-nothink with varying levels of thinking

## Context

In the past few sessions, we have hosted a local version of Qwen3.8-27B to run experiments of video analysis on a stimulus of crosses moving in a circle. We just ran a preliminary experiment on a sweep of these stimuli where the crosses move at varying speeds. Of note is that Qwen3.8-27B was set to "no thinking" mode. The results are unsurprising, the model returned the same answer (False) for stimulus. I added the log file for the run for your reference.

## Part 1: Understanding the effect of thinking

Today, we will look at the results of this experiment, and then vary the level of thinking for the model. From previous experiments on a non-locally hosted Qwen3.8-27B, we saw that reasoning effort had a large impact on the model's ability to perform the task.

We can vary thinking via the following arguments

```Python
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": False,
        "reasoning_effort": "medium",
    }
```

For `"reasoning_effort"`, we can use `"low"`, `"medium"`, or `"xhigh"`. No `"high"` oddly enough. I'm leaning towards not preserve thinking for the batch sweep to lower the amount of text being stored in the log, but let's run a couple of profiles with preserve thinking to see what the output is.

## Part 2: Dealing with unreasonably long wait times

While we have built out a pipeline to run inference on models offline in order to control the exact videos they are seeing, we are now running into issues with wait time on the cluster. The profiling jobs above have been submitted, but it will take a while before they are completed.

In the meantime, I'm curious as to if we can use OpenRouter again, but get around the downsampling issue by artificially making the videos 2 fps with a duration of 50 seconds. We are still giving the model 100 frames, but spaced out over 50 seconds so OpenRouter keeps all of our frames.

Instead of modifying `scripts/debug_circular/generate_samples.py` like we've done before (along with adding the requisite functions in `src/finst_video_model/debug_circular`), let's generate some sample stimuli in a new script. Make sure these new stimuli are saved in a separate folder in `data/debug_circular`.

## Part 3: Part 2 actually worked

In part 2, we experimented with artificially stretching the videos so that models on openrouter would get access to all 100 frames. This worked. Now let's revisit our work on the pylyshyn stimuli.

First, we want to port over most of the parameters from the debug circular stimulus into the pylyshyn stimulus. This includes cueing shapes by coloring them red instead of blinking or changing their shape. We want to have the same 384x384 sized stimulus. We want to have the same sized crosses. We will also only use three crosses to begin with, but we should allow for more in the future. Essentially, the pylyshyn stimulus should look nearly identical to the debug circular stimulus, expect for the motion, this should remain the same.

Second, we want to test Qwen3.8, like last Pylyshyn Sweep, but the parameters are probe_on_target, model size, reasoning effort (none, low, medium), and stimuli variant (native and stretched). Let's have 8 seeds per condition, so that each model variant has 16 samples (8 where probe is on target, 8 where probe is not on target).

## Part 4: We now have a regime where Qwen3.8-max works

The results from part 3 are very insightful. While Qwen3.8-27B has some kinks to workout with how much of a reasoning budget it needs to solve the task, Qwen3.8-max works. I believe Qwen3.8-max works simply because it is a bigger model. For one final experiment, let's test Qwen3.5-122B-A10B with no reasoning. Qwen3.8-max is required to have reasoning, but I'm ultimately curious if a model without reasoning can solve the task. Can it just intake the video and the prompt and output True or False solely from the first pass.
