# Testing all models on hard stimuli

## Context

Last session, we saw that we could close the performance gap between the local cluster and openrouter by using mp4 files. Moreover, we also saw that performance on openrouter significantly decreased by making the stimulus harder. Specifically, we saw that Pylyshyn stimuli where the nearest neighbor heuristic failed were especially hard for gemma-4-31b-it on both local cluster and openrouter.

## Part 1: Let's evaluate all models

In light of this finding, let's return to openrouter models and test them on 100 videos of the Pylyshyn task (balanced with 50 true and 50 false) with three objects (and only one cue). Let's test many of the same models before with stretched videos and no reasoning. Let's also try to approximate the cost of this sweep as best as possible. The final figure should be a bar graph that compares the performance across all models.

## Part 2: Let's turn on thinking

In the previous part, we came across a disappointing results: none of the models get above 80% accuracy on the hard stimuli. However, this was with thinking turned off. Just to show the effect of thinking, let's turn it on and compare the results. We want to use the medium thinking setting across all models. I leave it to you to determine a fair way to balance thinking token budgets.

## Part 3: Organize files

With the previous result, we will most likely pause this project indefinitely. Before we do that, we need to do some house keeping. Let's organize the files in `results/pylyshyn` and `scripts/pylyshyn`. Opening these directories gives people whiplash with the number of files and subdirectories. Let's just add some more subdirectories to make this manageable.

## Part 4: Rerunning the sweep today, but with even harder stimulus

Some after work thoughts. Let's rerun the sweep (both with and without reasoning). Drop qwen3.5 for `google/gemini-3.8-flash` and drop glm-5v-turbo for `qwen/qwen3.8-27b`. Also, let's sample stimuli to have the nearest neighbor heuristic always be wrong. Estimate the cost of the sweep before running. I'm expecting it to be between $9 and 10$.
