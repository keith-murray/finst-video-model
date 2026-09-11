# Testing Gemma-4-31b-it model locally on Pylyshyn

## Context

Yesterday, we built a pipeline for running gemma-4-31b-it locally (really on the department cluster). We tested it on the debug circular task with a 40 degree rotation. We found that there is a gap between performance on openrouter and performance on the department cluster. We tried to debug many potential issues and did not find the culprit. There are many more avenues to search, but for today, we are going to run inference on the local gemma4 model on the pylyshyn task.

## Running inference on the Pylyshyn task

I want to make a similar figure as `results/pylyshyn/nobjects_sweep/accuracy_by_nobjects_fast.png`, but for the locally hosted gemma4 model. Don't worry about stretched versus native, we'll just do native. We will rsync the stimuli on to the cluster, run the experiment, aggregate the data, pull the data to this machine, and then create a figure that compares the openrouter and local results against the nearest neighbor heuristic.

## Testing different hosts of gemma4

Note that it is the next day, but we are continuing this session.

Following from the work in the previous task, let's test different hosts for gemma-4-31b-it on openrouter to see if there is a difference in the performance across these hosts. Let's test circular debug (40, 80, and 120 degrees) and pylyshyn (3, 4, and 5 objects, fast, with nearest neighbor heuristic indicated). On the figures, let's just note the host, and then in the summary file, we can note the quantization and any other information about the hosts.
