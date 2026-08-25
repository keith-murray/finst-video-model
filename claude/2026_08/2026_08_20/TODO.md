# Empirically testing the effect of reasoning on Qwen in the Pylyshyn task

## Context

Yesterday, we tested the `qwen3.8-max` and `qwen3.8-27b` models on the Pylyshyn task across a variety of conditions. We saw that reasoning significantly aided `qwen3.8-27b` in solving the task. Let's see if this holds for `qwen3.8-max` and for higher levels of reasoning.

## Experiment

Let's test both `qwen3.8-max` and `qwen3.8-27b` models across a range of reasoning budgets on the simplest version of the Pylyshyn task: where there are 2 objects and only one is cued.

The final figure should be a line plot with two lines (one for `qwen3.8-max` and `qwen3.8-27b` respectively), the x-axis is reasoning budget and the y axis is percent error.
