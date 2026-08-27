# Polishing results from yesterday

## Context

Yesterday we did a lot of great work. We figured out how to send all 100 frames to openrouter via stretching the videos (accomplished by manipulating the mp4 metadata) and we saw that Qwen3.6-plus could solve our tasks with no reasoning. Yay! Today, we want to fill in some of the models we didn't test yesterday, and understand the effect of stimulus speed on Qwen3.6-plus.

## Tasks

Let't complete one task at a time. And reminder, this is all with the new pylyshyn stimulus.

## Task 1: Filling out stretch sweep

From yesterday, we have this fantastic plot `results/pylyshyn/stretch_sweep/accuracy_summary.png` that shows the effect of reasoning effort and video stretching on a host of models. Let's round it out by including qwen3.8-27b xhigh reasoning. Let's also remake the accuracy figure so that all bars are the same width. At the moment each panel is dedicated to a model, and each panel is the same width, causing bars that are quite wide. Just make it one panel where the model is specific via the x-axis tick label.

## Task 2: Understanding the effect of stimulus speed on Qwen3.6-plus

I have a hypothesis that the stimulus speed, which we slowed down yesterday, has an effect on Qwen3.6-plus ability to solve the task. Let's try another sweep that changes the following parameters:

- `redirect_s`: yesterday we extended it from 1s to 2s. Let's try 1s again.
- `speed_px_s`: yesterday we cut the speed. Let's double the speed.

I'm thinking the final result is a 2x2 panel figure where each panel has one `redirect_s` and one `speed_px_s` condition. In other words, one panel is (slow, slow), another is (fast, slow), etc. (where the condition means (`redirect_s`, `speed_px_s`))

## Task 3: Trying other models

The sweep above is an excellent method for gauging model performance. We can even try this on other models to understand if they can complete the task. Let's bump up the number of matching and non-matching seeds to 10 each for 20 seeds per condition. Try `z-ai/glm-5.3-flash` with no reasoning.

## Task 4: Follow up on Gemma work

Task 3 reveal something very surprising: gemma-4-31B solve the task incredibly well. This is shocking to me. I want to test it on the debug circular stimulus. Moreover, I want to test it for naive and stretched videos across angles of {40, 80, 120, 160, 200, 240, 280, 320, 360} with 20 seeds in each condition (10 for true, 10 for false). I want a figure that plots the performance as a line, with two separate lines for naive and stretched.

I hypothesize that we won't learn much from the result, but it is a good sanity check. Estimate the openrouter cost before running.
