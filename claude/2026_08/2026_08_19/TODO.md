# Testing Qwen 3.8

## Context

The previous experiments have shown that Gemini 3.7 can perform above chance accuracy at the Pylyshyn and smooth pursuit tasks. However, Gemini is not open source, so we can't perform mechanistic interpretability on the model performing the task. Therefore, we will conduct experiments on Qwen 3.8, the latest open source VLMs to determine if they can perform the tasks.

## The two models

We have two models

1. qwen/qwen3.8-max
2. qwen/qwen3.8-27b

The `qwen3.8-max` model is expected to perform better, but the weights are not released yet. Thus, we will test both `qwen3.8-max` and `qwen3.8-27b` models.

## The task

We will focus on the Pylyshyn task today. We will vary the number of objects and the number of cues. The number of objects will be an even number in $\{2,4,6,8,10\}$ and the number of cues will start at 1 and increase by 2 until the number of cues is great than half the number of objects. In other words:

1. 2 objects will have 1 cue
2. 4 objects will have 1 cue
3. 6 objects will have 1 and 3 cues
4. 8 objects will have 1 and 3 cues
5. 10 objects will have 1, 3, and 5 cues

Each condition should have 20 seeds, 10 matching and 10 non-matching.

## First step

Let's estimate how much it would cost to run these experiments on Openrouter.

## Summary plot

Once we have the data I want a summary plot that has percent error on the y-axis and number of cued objects on the x axis. The two different models will be different lines. There should be a separate panel for each of the total number of objects.
