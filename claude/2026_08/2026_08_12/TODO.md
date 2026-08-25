# Coming back after 3 weeks

## Context

I have put this project on hold for the past 3 weeks to focus on presenting another project at a conference. I am now coming back to this project with the goal of wrapping it up as soon as possible. 

## What needs to be accomplished

There are two questions we need to answer:

1. Can video language models track 1 object amongst a field of distractors?
2. Can video language models perform Pylyshyn's motion tracking experiment?

### First question

We have a nice handle on the first question for one model, but we need more models. For Qwen3.5 flash, the answer looks to be that it cannot reliably track 1 object out of 5 for longer than 10 seconds. We need to perform this experiment at scale for a variety of video language models on OpenRouter.

### Second question

We were originally motivated to understand motion tracking in video language models by Pylyshyn's FINST theory. Regardless of what our answer is for the first question, we want to reproduce Pylyshyn's experiment in video language models. First, we need to understand what his experiment was.

#### Pylyshyn's experiment

Here are some exerpts:

> Subjects were shown a display consisting of 10 stationary plus (+) signs and were told to note the subset of from 1 to 5 that were flashing. After 10 seconds the flashing stopped and all 10 objects began mocing. The subjects' task was to track the subset that had been flashing, without moving their eyes, and to indicate whenever one of those target objects 'flashed', i.e. briefly changed into a solid square shape. Whenever this happened, subjects were instructed to press a response key. A detailed description of this display and the timing is given below.
> An animated sequence of frames (generated in advance) was displayed by an Apply II+ microcomputer on a 50cm Sony monochrome monitor at a viewing distance of 120 cm. The display consisted of a white fixation square subtending a visual angle of 0.42 deg. This fixation square was presented in the center of the screen with a black background, the background subtending a visual angle of 21.5 deg. The animated stimuli always consisted of ten moving white crosses. A randomly chosen subset of from one to five of the total field of ten objects was designated as targets. The remainder were designated as distractors. Each object subtended a visual angle of approximately 0.42 geg and moved with a velocity and direction that was changed at random every few hundred milliseconds. The velocities of the objects ranged from 1.25 to 9.4 deg/s. The directions were chosen from among 8 equal divisions of the compass. The random motion of the objects was subject to the restriction that no two objects could be closer than 0.75 deg apart, so that the continuity of their identity was never ambiguous (as it would be if they were allowed to collide). In generating the animation sequence to meet this restriction, trajectories were generated for each object. After each frame, the location of objects was tested to determine whether any two objects were too close together. If they were the last few frames of the sequence were rejected and the generation was restarted at that point with a new random choice of directions and/or velocities. When an object was about to go off the edge of the screen its motion was reflected ('bounced') off that edge.
> Each trial consisted of a variable number of animation frames with the duration of the animated portion of the display ranging from 7 to 15 seconds. After a predetermined time (at least 3 seconds after the start of the animation), a solid white square, subtending 0.6 deg of visual angle, was flashed over (i.e. replaced) one of the moving objects for about 83 ms. Sometimes the 'flash' occurred over a target and sometimes over a distractor. The target-flash condition (to which subject had to respond) was sometimes the first flash in a trial and at other times was preceded by from 1 to 3 distractor-flash conditions (each of these 4 conditions occurring equally often), in order to ensure that subjects did not merely respond to any flash. There was only one target-flash condition in each trial. The trial ended after the subject responded to that flash, or at the end of the animation sequence which occurred at least 4s after the target-flash.

#### Thoughts

We can't have the VLMs perform a reaction like the participants, but we can think of various ways to change the experiment to accomodate VLMs.

## One final thought

There is nothing that needs to be done today, jsut ideation. Going forward, let's try to organize the repo around these 2 questions and developing the experiments.

