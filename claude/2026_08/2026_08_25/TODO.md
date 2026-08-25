# Building a debugging stimulus

## Context

In a separate project, I have been able to successfully host Qwen3.8-27B on 2 A100 GPUs on a cluster that I have access through via my department. This allows us to control exactly the what the stimulus the model sees is instead of guessing at how a preprocessor might be downsampling an mp4 file.

In this session, I want to design a simple stimulus to test the video reasoning capabilities of Qwen3.8-27B. We will return to the smooth pursuit and pylyshyn stimuli, but first we want to make the stimuli as simple as possible to understand if the model can reason about videos at all.

## First task

Let's begin by depreciating and removing the `src/finst_video_model/generation` source files entirely. We will then move files in `src/finst_video_model/comprehension` outside of the `comprehension` folder to the `src/finst_video_model` directly. This will require us updating many python files in `scripts/`.

## Second task

In this step, we will design the stimulus. I'm thinking of white crosses on a black background, like in the pylyshyn stimulus. However, the crosses are arranged, equally spaced, along a circle and will move clockwise or counter-clockwise around the circle with a particular speed.

There will still be three phases:

1. Crosses are stationary and one is colored red to indicate the target
2. Crosses move according to specified circular trajectory and speed
3. Crosses stop moving and one is colored red

The first and third phase will be 1 second, and the second phase will be 8 seconds.

For the size of the video and the frame rate, let's have the video be 400 by 400 pixels and have the frame rate be 10 fps.

For the colors, let's have the black be (0,0,0), red be (255,0,0), and black be (255, 255, 255).

For the speed of the stimulus and the direction, we want to have a variety of stimulus speeds to test the performance of the model. We will also have equal numbers of clockwise and counter-clockwise stimuli.

For now, let's just make the source files and `scripts/debug_circular/generate_samples.py` script file to generate sample stimuli. We will save the stimuli as both mp4s and numpy arrays. The model will receive numpy arrays, but we will use the mp4s to debug. The file name for this work to be saved in (in the data, results, scripts, and src folders) is `debug_circular`.

## Third task

Great, we will now move on to the third task, which is submitting jobs on the department cluster to run inference on the stimuli that we will generate. The Qwen3.8-27B pipeline is a bit involved, but I've provided context in `claude/skills/cluster` about how I've gotten it to work. The prompt will be different since the stimuli is different, and the data directory will be a bit different of course.

The model is known to overthink, so let's make sure we change the return statement of `prepare_input` to

```Python
        return {
            "prompt": text,
            "multi_modal_data": {"video": (frames, video_metadata)},
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"do_sample_frames": False},
        }
```

Eventually we will test different levels of thinking via `"chat_template_kwargs": {"reasoning_effort": "medium"}` where we can vary the reasoning effort. For now, we just want the model to return True or False.

As for the sweep, we will have the following conditions to vary

1. `rotation_deg`
2. `clockwise`
3. `probe_on_target`

Each set of conditions should get 10 seeds. The most impactful variable is `rotation_deg`. We want to try ${20,40,60,80,100}$ degrees.

A big unknown is how much time was should ask for to run this job. We should have a profiling job to determine the average amount of time to run one seed. Given that a seed requires 2 A100 GPUs, we will have to run seeds sequentially.
