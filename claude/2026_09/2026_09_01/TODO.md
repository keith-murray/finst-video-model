# Testing `qwen3.6-plus` on the circular debug task

## Context

In the last session, we saw that `gemma-4-31b-it` did very well in the Pylyshyn task, yet failed in certain regimes of the circular debug task. This is confusing because the Pylyshyn task should be much harder. Potentially the model is defaulting to a "nearest neighbor" heuristic, which just happens to perform well in the Pylyshyn task. As a sanity check, let's test the `qwen3.6-plus` on the circular debug task.

## Task 1

Let's reproduce `results/debug_circular/angle_sweep/accuracy_by_angle.png` using the `qwen3.6-plus` model.

## Task 2

Following from the work in task 1, we can see that the V-shape phenomena in the circular debug plots is quite interesting. It appears in `gemma-4-31b-it` and `qwen3.6-plus`, models with high visual reasoning abilities. One concern of mine is that this is indicative of a "nearest neighbor" heuristic, and that models in the Pylyshyn task are engaging using this heuristic to solve the task.

For the results in `results/pylyshyn/speed_sweep_models/accuracy_summary.png`, I want to know how well the "nearest neighbor" heuristic would perform. Is it possible to get results from this heuristic via the videos we already have?

The figure I would like is to replot the `qwen3.6-plus` and `gemma-4-31b-it` results from `results/pylyshyn/speed_sweep_models/accuracy_summary.png` with dashed red horizontal lines indicating performance of the heuristic (just like we have horizontal lines indicating chance).

## Task 3

Just as I suspected. The "nearest neighbor" heuristic performs quite well on the task. To disambiguate genuine object tracking from this heuristic, let's scale the number of objects up to 4 and 5. We'll keep evaluating `qwen3.6-plus` and `gemma-4-31b-it` and let's only focus on the slow redirect and slow speed pylyshyn stimulus conditions.
