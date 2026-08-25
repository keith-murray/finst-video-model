# Building out FINST-like experiment

## Context

From the last TODO (specifically `claude/2026_08_12/TODO.md`), we saw that our current smooth object tracking task differs from the original FINST experiment. The main differences are the following:

1. Cueing happens before motion starts, not during it
2. The motion itself is far more unpredictable than ours
3. Pylyshyn explicitly prevents identity ambiguity via a minimum-sepration constrain that we don't enforce during motion
4. The response format is a continuous monitoring task, not a single end-of-trial report
5. The experiment fixes the total field at 10 objects and varies the cued subset size
6. The objects are crosses and flash to squares

## Building the new task

Let's not change the smooth object tracking task that we have. Instead, let's build out another task for the pylyshyn experiment. 

The only thing we'll have to change is how the model reports. VLMs can't do continuous reporting, so we will only flash one object and then ask the model to report True or False on whether the flashed object was a cued object or not. We can assess the performance of models via computing a d' value.

### Organizing these new files

Add a subdirectory in `src/finst_video_model/comprehension` for these pylyshyn experiments. We want to reuse the same vlm client and any other relevant functions in `src/finst_video_model/comprehension`, but we want the stimulus gen for the pylyshyn experiments to be silo-ed off.

### Checking the stimulus

Create a script in `scripts/pylyshyn` to generate a host of sample videos in `data`.

## Updating the old task

Now let's change the organization and substance of the old task to mirror the pylyshyn task, keeping the smooth object motion. Let's have

1. Cueing happens before motion starts, not during it (keep the red cueing)
2. explicitly prevent identity ambiguity via a minimum-sepration constrain
3. Have a cueing phase at the end (stop motion) and ask the model to output whether the cued object was cued in the beginning

### Organizing these files

Let's move these files to the `src/finst_video_model/comprehension/smooth_pursuit` subdirectory.

### Checking the stimulus

Create a script in `scripts/smooth_pursuit` to generate a host of sample videos in `data`.

## Let's run some simple experiments

Let's run an experiment with on both the pylyshyn and smooth pursuit paradigms. Let's try Gemini 3.7 flash (Google Vertex provider is half off at the moment for this model, so let's try and see if we can specify the Google Vertex provider).

### Pylyshyn

Let's keep the number of crosses cued at 1/10. This should be the simplest set up. Let's had 20 seeds. I want a confusion matrix and d' plots in one figure at the end.

### Smooth pursuit

Let's have the number of circles cued at 1/5. This should be the simplest set up. Let's had 20 seeds. I want a confusion matrix and d' plots in one figure at the end.
