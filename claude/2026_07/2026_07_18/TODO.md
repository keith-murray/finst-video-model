# Organization and new prompt

## Organization

### Context

Yesterday we submitted a job to OpenRouter to generate a video. We generated many artifacts in the `data/` folder. Let's make things more organized.

### Actionable

Let's have it so that when we submit a job to OpenRouter, all the artifacts generated are in a subfolder in data. Organize the existing run into a subfolder as well.

## New Prompt

### Context

Yesterday we used OpenRouter to generate a video of balls moving to test the FINST theory behaviorally. The video included many halluciations. Let's simplify things.

### Actionable

First, let's have the trajectory that the circles travel on be further specified by us. I want the circles to move in a circle. Let's have a large circle be in the background of the image, this will be the track. The circles will be initialized on the track and then the model will be instructed to move them clockwise on the track.

Second, let's just have three circles where one of them is initially colored. The model should then, like last time, initially color all the circles grey, move them acording to the circular trajectory above, stop the circles after some amount of time, and then recolor the initial red circle.

### Results

Veo was able to follow instructions for generating the movement of the circles, but for 3 circles the wrong one was labeled at the end. For 2 circles, the correct one was labeled at the end.

## De-emphasize Veo in the repo

### Context

We originally started out this repo with attempting to understand if Veo can perform a FINST-like task. In principle, we want to test any video generation model on OpenRouter. 

## Actionable

In the `src/finst_video_model/veo_client.py` file, remove references to the VEO model. These should be made in the scripts from which the jobs are being sent from. Also remane the file to just `client.py`.

In the README.md, rewrite it to be more general to video generation models.
