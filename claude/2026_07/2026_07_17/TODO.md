# Project onboarding

## Context

I am working on a rotation project in my first year of a Neuroscience PhD program. Question for my rotation project is simple:

> Do video generation models use a FINST like mechanism for object tracking?

For context, FINST is a theory from Zenon Pylyshyn of visual attention and indexing. In Pylyshyn's 1988 paper "Tracking multiple independent targets: Evidence for a parallel tracking mechanism", he outlined a behavioral task and a variety of behavioral signatures surrounding his theory.

In the first step of this project, we seek to see if we can reproduce this task and behavioral signatures in Google Deepmind's Veo 3.1 model. However, since we are looking at video generation models, we have to get creative as to how we can adapt the task and what the behavioral signatures look like.

## Actionable

I have previously ideated with Claude about what the task can look like. Now I want to see a proof of concept that we can generate an image and prompt, use openrouter to interface with Veo, and download the generated video. We will leave the automated evaluation of the generated video for another time.

I have included markdowns from openrouter about how to interface with the api.

Note that this project uses uv for dependency managing and project packaging.
