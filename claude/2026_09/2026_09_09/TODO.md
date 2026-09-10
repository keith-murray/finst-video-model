# Running gemma-4-31b-it tests locally

## Context

From our experiments on openrouter yesterday, we saw that the `gemma-4-31b-it` model was only ingesting 32 frames. This leaves open the question of whether this small number of frames contributed to the model's superior ability to analyze videos. To test this hypothesis, we need to locally host the model and vary the number of frames ingested.

In a separate Claude interaction, I downloaded `gemma-4-31b-it` on my department's cluster and built a pipeline to feed in numpy arrays with the option of sampling any number of different frames. Refer to the `claude/skills/cluster/gemma4` directory for those files.

## Testing `gemma-4-31b-it` with different number frames on the circular debug task

In the `results/debug_circular/angle_sweep/accuracy_by_angle.png` figure, we saw that `gemma-4-31b-it` achieves fairly high performance on the circular debug task with a rotation degree of $40^\circ$. Let's test the locally hosted (department hosted would be more accurate) `gemma-4-31b-it` model with a sweep of different number of frames being sampled.

This control makes sense because the task is solvable, in principle, with any amount of frames. Hence, if the number of frames cause the model's performance to worsen, we would see if here despite the task being solvable.

Let's generate 40 different stimuli (20 match and 20 non-match) (just numpy so I can generate them on the cluster) and then each sweep instance will run inference on the same 40 stimuli.
